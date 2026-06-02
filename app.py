"""
Turf Analyzer v5 - Plateforme professionnelle de pronostics PMU

v5 nouveautés :
  15. XGBoost-like avec régularisation L2 + subsampling + early stopping
  16. Multi-paris : placé, couplé gagnant/placé, tiercé
  17. Scraping Geny pour terrain/météo/pronostics presse
  18. Système d'alertes value bets (notifications navigateur)
  19. Dashboard analytique ROI (par hippo, discipline, dans le temps)
  20. Base SQLite persistante (remplace JSON)
"""

from flask import Flask, jsonify, render_template, request
from datetime import datetime, timedelta
import requests
import math
import os
import pickle
from functools import lru_cache
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

from lib.ml_models import (GradientBoosting, RandomForest, Ensemble,
                            fit_isotonic, apply_calibration, load_model_from_dict)
from lib.xgb_like import XGBoostLike
from lib.neural_net import MLPClassifier
from lib.automl import (log_loss, roc_auc, brier_score, calibration_curve,
                         evaluate_model, cross_validate, random_search,
                         StackingEnsemble, feature_importance_perturbation)
from lib.kelly import kelly_amount, kelly_fraction, expected_value, expected_roi
from lib.features_v4 import (build_pedigree_stats, get_pedigree_score,
                              get_corde_score, get_equipment_score,
                              detect_profile, get_profile_match_score)
from lib.multi_paris import proba_place_simple, best_combinations
from lib.walk_forward import generate_windows, aggregate_fold_metrics, fmt_window
from lib.calibration import Calibrator
from lib import db
from lib import geny_scraper

app = Flask(__name__)

PMU_BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/61/programme"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TurfAnalyzer/5.0)"}

CACHE_DIR = os.environ.get("CACHE_DIR", "/tmp/turf_cache")
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
except Exception:
    CACHE_DIR = "/tmp/turf_cache"
    os.makedirs(CACHE_DIR, exist_ok=True)

# Caches v5
STATS_CACHE_FILE = os.path.join(CACHE_DIR, "stats_team_v5.pkl")
HORSE_STATS_FILE = os.path.join(CACHE_DIR, "horse_stats_v5.pkl")
ELO_CACHE_FILE = os.path.join(CACHE_DIR, "elo_v5.pkl")
ELO_HIST_FILE = os.path.join(CACHE_DIR, "elo_hist_v5.pkl")
HORSE_RACES_FILE = os.path.join(CACHE_DIR, "horse_races_v5.pkl")
PEDIGREE_FILE = os.path.join(CACHE_DIR, "pedigree_v5.pkl")
ML_MODEL_FILE = os.path.join(CACHE_DIR, "ml_model_v5.pkl")
CALIBRATION_FILE = os.path.join(CACHE_DIR, "calibration_v5.pkl")
OLD_BETS_JSON = os.path.join(CACHE_DIR, "bets_v4.json")  # pour migration

# Migration v4 → v5 au démarrage
try:
    n = db.migrate_from_json(OLD_BETS_JSON)
    if n > 0:
        print(f"[Migration] {n} paris migrés depuis bets_v4.json → SQLite")
except Exception as e:
    print(f"[Migration] {e}")

WINDOW_SHORT = 30
HISTORY_DAYS = 180

# v7.1 — réglages issus de l'optimisation walk-forward (sans look-ahead)
# Calibration sur un hold-out (anti-optimisme) + poids du blend heuristique/ML.
CALIB_METHOD = "isotonic"   # "isotonic" | "platt"
CALIB_HOLDOUT_FRAC = 0.25   # part du train réservée à la calibration
ML_BLEND_WEIGHT = 0.55      # poids du ML dans chance = (1-w)*heur + w*ml


# ============================================================
#  PMU API
# ============================================================
def fmt_date(d):
    return d.strftime("%d%m%Y")


@lru_cache(maxsize=256)
def get_programme(date_str):
    r = requests.get(f"{PMU_BASE}/{date_str}", headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=1024)
def get_participants(date_str, r_num, c_num):
    url = f"{PMU_BASE}/{date_str}/R{r_num}/C{c_num}/participants"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


def get_participants_live(date_str, r_num, c_num):
    url = f"{PMU_BASE}/{date_str}/R{r_num}/C{c_num}/participants"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()


@lru_cache(maxsize=1024)
def get_performances(date_str, r_num, c_num):
    url = f"{PMU_BASE}/{date_str}/R{r_num}/C{c_num}/performances-detaillees/pretty"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"participants": []}


# ============================================================
#  Cache helpers
# ============================================================
def load_pickle(path, max_age_hours=24):
    if os.path.exists(path):
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
                age = datetime.now() - datetime.fromisoformat(data["saved_at"])
                if age.total_seconds() < max_age_hours * 3600:
                    return data["payload"]
        except Exception:
            pass
    return None


def save_pickle(path, payload):
    try:
        with open(path, "wb") as f:
            pickle.dump({"saved_at": datetime.now().isoformat(), "payload": payload}, f)
    except Exception as e:
        print(f"Save error {path}: {e}")


# ============================================================
#  Construction des stats (v4 → v5 mêmes calculs)
# ============================================================
def _empty_bucket():
    return {"c": 0, "v": 0, "p": 0}


def _fetch_course_full(args):
    date_str, r_num, c_num, discipline, hippodrome, delta_days, type_corde = args
    try:
        parts = get_participants(date_str, r_num, c_num)
        return (parts, discipline, hippodrome, delta_days, date_str, type_corde)
    except Exception:
        return None


def compute_all_stats(max_days=HISTORY_DAYS, ref_date=None, use_cache=True):
    """Calcule toutes les stats (Elo, forme, drivers, pedigree...) sur les
    `max_days` jours PRÉCÉDANT `ref_date`.

    ref_date : datetime de référence. None => aujourd'hui (comportement v6).
               En walk-forward on passe une date passée pour n'utiliser QUE les
               données antérieures à la course évaluée (anti look-ahead).
    use_cache: le cache pickle monolithique n'est valide que pour "aujourd'hui".
               En walk-forward (ref_date passée) on le désactive automatiquement.
    """
    is_today = ref_date is None
    if not is_today:
        use_cache = False  # le cache global ne vaut que pour "aujourd'hui"

    if use_cache:
        cached_t = load_pickle(STATS_CACHE_FILE)
        cached_h = load_pickle(HORSE_STATS_FILE)
        cached_e = load_pickle(ELO_CACHE_FILE)
        cached_eh = load_pickle(ELO_HIST_FILE)
        cached_hr = load_pickle(HORSE_RACES_FILE)
        cached_p = load_pickle(PEDIGREE_FILE)
        if all([cached_t, cached_h, cached_e, cached_eh, cached_hr, cached_p]):
            return cached_t, cached_h, cached_e, cached_eh, cached_hr, cached_p

    team_stats = {
        "drivers": defaultdict(_empty_bucket),
        "drivers_short": defaultdict(_empty_bucket),
        "drivers_disc": defaultdict(lambda: defaultdict(_empty_bucket)),
        "drivers_hippo": defaultdict(lambda: defaultdict(_empty_bucket)),
        "entraineurs": defaultdict(_empty_bucket),
        "entraineurs_short": defaultdict(_empty_bucket),
        "entraineurs_disc": defaultdict(lambda: defaultdict(_empty_bucket)),
    }
    horse_stats = {
        "global": defaultdict(_empty_bucket),
        "with_driver": defaultdict(lambda: defaultdict(_empty_bucket)),
        "hippo": defaultdict(lambda: defaultdict(_empty_bucket)),
        "disc": defaultdict(lambda: defaultdict(_empty_bucket)),
    }
    elo = defaultdict(lambda: 1500.0)
    elo_hist = defaultdict(lambda: deque(maxlen=10))
    horse_races = defaultdict(list)
    pedigree_data = []
    elo_K = 16

    tasks = []
    today = ref_date if ref_date is not None else datetime.now()
    for delta in range(1, max_days + 1):
        d = today - timedelta(days=delta)
        date_str = fmt_date(d)
        try:
            prog = get_programme(date_str)
        except Exception:
            continue
        for r in prog["programme"]["reunions"]:
            hippo = r["hippodrome"]["libelleCourt"]
            for c in r["courses"]:
                if c.get("arriveeDefinitive"):
                    tasks.append((date_str, r["numOfficiel"], c["numOrdre"],
                                  c.get("discipline", ""), hippo, delta,
                                  c.get("corde", "")))

    with ThreadPoolExecutor(max_workers=30) as ex:
        results = list(ex.map(_fetch_course_full, tasks))

    valid = sorted([r for r in results if r], key=lambda x: -x[3])

    for parts_data, discipline, hippo, delta_days, date_str, type_corde in valid:
        is_short = delta_days <= WINDOW_SHORT
        partants = [p for p in parts_data.get("participants", [])
                    if p.get("statut") == "PARTANT"]
        finishers = sorted(
            [p for p in partants if (p.get("ordreArrivee") or 0) > 0],
            key=lambda p: p["ordreArrivee"]
        )
        all_horses_in_race = [p.get("nom") for p in partants if p.get("nom")]
        race_ts = (today - timedelta(days=delta_days)).timestamp()

        for p in partants:
            driver = p.get("driver")
            entr = p.get("entraineur")
            cheval = p.get("nom")
            pere = p.get("nomPere")
            mere = p.get("nomMere")
            place = p.get("ordreArrivee", 0) or 0
            won = 1 if place == 1 else 0
            placed = 1 if 1 <= place <= 3 else 0

            pedigree_data.append({"cheval": cheval, "pere": pere, "mere": mere, "place": place})

            if driver:
                team_stats["drivers"][driver]["c"] += 1
                team_stats["drivers"][driver]["v"] += won
                team_stats["drivers"][driver]["p"] += placed
                if is_short:
                    team_stats["drivers_short"][driver]["c"] += 1
                    team_stats["drivers_short"][driver]["v"] += won
                    team_stats["drivers_short"][driver]["p"] += placed
                if discipline:
                    team_stats["drivers_disc"][driver][discipline]["c"] += 1
                    team_stats["drivers_disc"][driver][discipline]["v"] += won
                    team_stats["drivers_disc"][driver][discipline]["p"] += placed
                if hippo:
                    team_stats["drivers_hippo"][driver][hippo]["c"] += 1
                    team_stats["drivers_hippo"][driver][hippo]["v"] += won
                    team_stats["drivers_hippo"][driver][hippo]["p"] += placed

            if entr:
                team_stats["entraineurs"][entr]["c"] += 1
                team_stats["entraineurs"][entr]["v"] += won
                team_stats["entraineurs"][entr]["p"] += placed
                if is_short:
                    team_stats["entraineurs_short"][entr]["c"] += 1
                    team_stats["entraineurs_short"][entr]["v"] += won
                    team_stats["entraineurs_short"][entr]["p"] += placed
                if discipline:
                    team_stats["entraineurs_disc"][entr][discipline]["c"] += 1
                    team_stats["entraineurs_disc"][entr][discipline]["v"] += won
                    team_stats["entraineurs_disc"][entr][discipline]["p"] += placed

            if cheval:
                horse_stats["global"][cheval]["c"] += 1
                horse_stats["global"][cheval]["v"] += won
                horse_stats["global"][cheval]["p"] += placed
                if driver:
                    horse_stats["with_driver"][cheval][driver]["c"] += 1
                    horse_stats["with_driver"][cheval][driver]["v"] += won
                    horse_stats["with_driver"][cheval][driver]["p"] += placed
                if hippo:
                    horse_stats["hippo"][cheval][hippo]["c"] += 1
                    horse_stats["hippo"][cheval][hippo]["v"] += won
                    horse_stats["hippo"][cheval][hippo]["p"] += placed
                if discipline:
                    horse_stats["disc"][cheval][discipline]["c"] += 1
                    horse_stats["disc"][cheval][discipline]["v"] += won
                    horse_stats["disc"][cheval][discipline]["p"] += placed

                adversaires = [h for h in all_horses_in_race if h != cheval]
                horse_races[cheval].append((race_ts, hippo, adversaires))

        if len(finishers) >= 2:
            for i, winner in enumerate(finishers):
                for loser in finishers[i+1:]:
                    wn = winner.get("nom")
                    ln = loser.get("nom")
                    if not wn or not ln:
                        continue
                    rw, rl = elo[wn], elo[ln]
                    expected_w = 1 / (1 + 10 ** ((rl - rw) / 400))
                    elo[wn] = rw + elo_K * (1 - expected_w)
                    elo[ln] = rl + elo_K * (0 - (1 - expected_w))
            for f in finishers:
                n = f.get("nom")
                if n:
                    elo_hist[n].append(elo[n])

    pere_stats, mere_stats = build_pedigree_stats(pedigree_data)
    pedigree = {"peres": pere_stats, "meres": mere_stats}

    def freeze(d):
        if isinstance(d, defaultdict):
            return {k: freeze(v) for k, v in d.items()}
        return d

    team_out = {k: freeze(v) for k, v in team_stats.items()}
    horse_out = {k: freeze(v) for k, v in horse_stats.items()}
    elo_out = dict(elo)
    elo_hist_out = {k: list(v) for k, v in elo_hist.items()}
    horse_races_out = {k: v for k, v in horse_races.items()}

    # On ne persiste le cache que pour le run "aujourd'hui" : un fold de
    # walk-forward (ref_date passée) ne doit JAMAIS écraser le cache prod.
    if is_today:
        save_pickle(STATS_CACHE_FILE, team_out)
        save_pickle(HORSE_STATS_FILE, horse_out)
        save_pickle(ELO_CACHE_FILE, elo_out)
        save_pickle(ELO_HIST_FILE, elo_hist_out)
        save_pickle(HORSE_RACES_FILE, horse_races_out)
        save_pickle(PEDIGREE_FILE, pedigree)

    return team_out, horse_out, elo_out, elo_hist_out, horse_races_out, pedigree


# ============================================================
#  Scoring helpers (inchangé)
# ============================================================
def get_bucket_score(bucket, max_score=100, min_courses=5):
    if not bucket or bucket["c"] < min_courses:
        return None
    c, v, p = bucket["c"], bucket["v"], bucket["p"]
    tv, tp = v / c, p / c
    confiance = min(1.0, c / 30)
    raw = tv * 200 + tp * 60
    return min(max_score, raw * confiance + 30 * (1 - confiance))


def get_team_score_multi(name, kind, team_stats, discipline=None, hippodrome=None):
    if not team_stats or not name:
        return 50
    if kind == "drivers":
        gb = team_stats["drivers"].get(name)
        sb = team_stats["drivers_short"].get(name)
        db_b = team_stats["drivers_disc"].get(name, {}).get(discipline) if discipline else None
        hb = team_stats["drivers_hippo"].get(name, {}).get(hippodrome) if hippodrome else None
    else:
        gb = team_stats["entraineurs"].get(name)
        sb = team_stats["entraineurs_short"].get(name)
        db_b = team_stats["entraineurs_disc"].get(name, {}).get(discipline) if discipline else None
        hb = None
    s_g = get_bucket_score(gb) or 50
    s_s = get_bucket_score(sb, min_courses=3)
    s_d = get_bucket_score(db_b, min_courses=3)
    s_h = get_bucket_score(hb, min_courses=3)
    parts = [(s_g, 0.35)]
    if s_s is not None: parts.append((s_s, 0.30))
    if s_d is not None: parts.append((s_d, 0.20))
    if s_h is not None: parts.append((s_h, 0.15))
    tw = sum(w for _, w in parts)
    return sum(s * w for s, w in parts) / tw


def get_horse_score(cheval, driver, hippodrome, discipline, horse_stats):
    if not horse_stats or not cheval:
        return 50
    s_g = get_bucket_score(horse_stats["global"].get(cheval)) or 50
    s_d = get_bucket_score(horse_stats["with_driver"].get(cheval, {}).get(driver),
                           min_courses=2) if driver else None
    s_h = get_bucket_score(horse_stats["hippo"].get(cheval, {}).get(hippodrome),
                           min_courses=2) if hippodrome else None
    s_di = get_bucket_score(horse_stats["disc"].get(cheval, {}).get(discipline),
                            min_courses=2) if discipline else None
    parts = [(s_g, 0.40)]
    if s_d is not None: parts.append((s_d, 0.25))
    if s_h is not None: parts.append((s_h, 0.20))
    if s_di is not None: parts.append((s_di, 0.15))
    tw = sum(w for _, w in parts)
    return sum(s * w for s, w in parts) / tw


def get_elo_score(cheval, elo_ratings, all_horses_in_race):
    if not elo_ratings or not cheval:
        return 50
    my_elo = elo_ratings.get(cheval, 1500)
    elos = [elo_ratings.get(h, 1500) for h in all_horses_in_race if h]
    if len(elos) < 2:
        return 50
    e_min, e_max = min(elos), max(elos)
    if e_max == e_min:
        return 50
    return (my_elo - e_min) / (e_max - e_min) * 100


def get_age_sexe_score(age, sexe):
    if not age:
        return 50
    if age <= 2: pts = 35
    elif age == 3: pts = 60
    elif age == 4: pts = 75
    elif age == 5: pts = 85
    elif age == 6: pts = 85
    elif age == 7: pts = 75
    elif age == 8: pts = 60
    elif age == 9: pts = 50
    else: pts = 40
    if sexe == "HONGRES": pts += 3
    return min(100, pts)


def get_repos_score(cheval, today_ts, horse_races):
    races = horse_races.get(cheval, [])
    if not races:
        return 50
    last_ts = max(r[0] for r in races)
    days = (today_ts - last_ts) / 86400
    if days < 0: return 50
    if days < 5: return 35
    if days < 8: return 55
    if days < 14: return 75
    if days <= 28: return 85
    if days <= 45: return 70
    if days <= 70: return 55
    if days <= 120: return 40
    return 25


def get_elo_trend_score(cheval, elo_hist, current_elo):
    hist = elo_hist.get(cheval, [])
    if len(hist) < 3:
        return 50
    recent = hist[-5:]
    if len(recent) < 2:
        return 50
    delta = recent[-1] - recent[0]
    score = 50 + (delta / 40) * 50
    return max(0, min(100, score))


def get_confrontation_score(cheval, adversaires, horse_races, elo_ratings):
    if not cheval or not adversaires:
        return 50
    my_races = horse_races.get(cheval, [])
    if not my_races:
        return 50
    nb_confrontations = 0
    my_elo = elo_ratings.get(cheval, 1500)
    for _, _, past_adversaires in my_races:
        for adv in adversaires:
            if adv in past_adversaires:
                nb_confrontations += 1
                break
    adv_elos = [elo_ratings.get(a, 1500) for a in adversaires]
    if not adv_elos:
        return 50
    avg_adv_elo = sum(adv_elos) / len(adv_elos)
    elo_diff = my_elo - avg_adv_elo
    exp_score = min(50, nb_confrontations * 8)
    force_score = 50 + max(-50, min(50, elo_diff / 4))
    return (exp_score + force_score) / 2


def score_forme_enrichi(perfs_detail, today=None):
    if not perfs_detail:
        return 50
    if today is None:
        today = datetime.now()
    entries = []
    for course in perfs_detail[:8]:
        try:
            date_ms = course.get("date")
            if not date_ms: continue
            d = datetime.fromtimestamp(date_ms / 1000)
            days_ago = max(1, (today - d).days)
            me = next((p for p in course.get("participants", []) if p.get("itsHim")), None)
            if not me: continue
            place = (me.get("place") or {}).get("place", 0) or 0
            rk_me = me.get("reductionKilometrique") or 0
            rk_winner = course.get("tempsDuPremier") or 0
            allocation = course.get("allocation") or 0
            nb_parts = course.get("nbParticipants") or 10
            entries.append({"days_ago": days_ago, "place": place,
                            "rk_me": rk_me, "rk_winner": rk_winner,
                            "allocation": allocation, "nb_parts": nb_parts})
        except Exception:
            continue
    if not entries:
        return 50
    score = 0
    wt = 0
    for e in entries:
        w = math.exp(-e["days_ago"] / 45)
        place = e["place"]
        nb = max(e["nb_parts"], 4)
        if place == 0: pts = 10
        elif place == 1: pts = 100
        elif place == 2: pts = 80
        elif place == 3: pts = 65
        else: pts = max(5, 65 - (place - 3) * 60 / max(nb - 3, 1))
        bonus_alloc = max(0, math.log10(max(e["allocation"], 1)) - 4) * 5
        bonus_rk = 0
        if e["rk_me"] > 0 and e["rk_winner"] > 0:
            ecart = e["rk_me"] - e["rk_winner"]
            if ecart < 500: bonus_rk = 8
            elif ecart < 1500: bonus_rk = 4
            elif ecart > 5000: bonus_rk = -5
        final_pts = min(100, pts + bonus_alloc + bonus_rk)
        score += final_pts * w
        wt += w
    return score / max(wt, 0.01)


def score_distance(perfs_detail, distance_course):
    if not perfs_detail or not distance_course:
        return 50
    proches = []
    for course in perfs_detail:
        dist = course.get("distance")
        if not dist: continue
        if abs(dist - distance_course) <= 200:
            for p in course.get("participants", []):
                if p.get("itsHim"):
                    place = (p.get("place") or {}).get("place", 0)
                    if place: proches.append(place)
    if not proches: return 50
    pts = 0
    for pl in proches:
        if pl == 1: pts += 100
        elif pl <= 3: pts += 75
        elif pl <= 5: pts += 55
        else: pts += 25
    return pts / len(proches)


# ============================================================
#  ML featurization (v5 = v4)
# ============================================================
def featurize(p, nb_partants):
    s = p["scores"]
    return [
        s.get("marche", 0), s.get("forme", 0), s.get("carriere", 0),
        s.get("gains", 0), s.get("driver", 50), s.get("entraineur", 50),
        s.get("distance", 50), s.get("cheval_stats", 50), s.get("elo", 50),
        s.get("age_sexe", 50), s.get("repos", 50), s.get("elo_trend", 50),
        s.get("confrontation", 50), s.get("pedigree", 50),
        s.get("corde", 50), s.get("equipment", 50), s.get("profile_match", 50),
        nb_partants, 1.0 / max(p.get("cote") or 50, 1),
        p["bonus"].get("team", 0), p["bonus"].get("deferre", 0),
        p.get("age") or 5,
        1 if p.get("sexe") == "FEMELLES" else 0,
    ]


FEATURE_NAMES = ["marche","forme","carriere","gains","driver","entraineur",
                 "distance","cheval_stats","elo","age_sexe","repos",
                 "elo_trend","confrontation","pedigree","corde","equipment",
                 "profile_match","nb_partants","inv_cote",
                 "bonus_team","bonus_deferre","age_raw","is_female"]


def load_ml_model():
    payload = load_pickle(ML_MODEL_FILE, max_age_hours=24*14)
    if not payload:
        return None
    t = payload.get("type")
    if t == "xgb":
        return XGBoostLike.from_dict(payload)
    if t == "mlp":
        return MLPClassifier.from_dict(payload)
    if t == "stacking":
        # Reconstruction de stacking
        bases = []
        for sub in payload.get("base_models", []):
            sub_t = sub.get("type")
            if sub_t == "xgb":
                bases.append(XGBoostLike.from_dict(sub))
            elif sub_t == "mlp":
                bases.append(MLPClassifier.from_dict(sub))
            elif sub_t == "rf":
                bases.append(RandomForest.from_dict(sub))
            else:
                bases.append(GradientBoosting.from_dict(sub))
        stk = StackingEnsemble(bases)
        import numpy as np
        stk.meta_weights = np.array(payload.get("meta_weights", []))
        stk.meta_bias = payload.get("meta_bias", 0)
        return stk
    return load_model_from_dict(payload)


def save_ml_model(model):
    if isinstance(model, StackingEnsemble):
        payload = {
            "type": "stacking",
            "base_models": [m.to_dict() for m in model.base_models],
            "meta_weights": model.meta_weights.tolist() if model.meta_weights is not None else None,
            "meta_bias": float(model.meta_bias),
        }
        save_pickle(ML_MODEL_FILE, payload)
    else:
        save_pickle(ML_MODEL_FILE, model.to_dict())


def load_calibration():
    return load_pickle(CALIBRATION_FILE, max_age_hours=24*7)


def save_calibration(c):
    save_pickle(CALIBRATION_FILE, c)


def _fetch_full(args):
    date_str, r_num, c_num, distance, discipline, hippodrome, type_corde = args
    try:
        return (get_participants(date_str, r_num, c_num),
                get_performances(date_str, r_num, c_num),
                distance, discipline, hippodrome, type_corde)
    except Exception:
        return None


def _collect_training_data(days_back, exclude_recent, ref_date=None,
                           stats_bundle=None):
    """Helper : collecte X, y depuis l'historique.

    ref_date     : date de référence (None => aujourd'hui). En walk-forward, on
                   passe la fin de la fenêtre d'entraînement.
    stats_bundle : tuple de stats déjà calculé (réutilisé par le walk-forward
                   pour ne pas recalculer compute_all_stats à chaque appel).
    """
    X, y = [], []
    today = ref_date if ref_date is not None else datetime.now()
    if stats_bundle is not None:
        team_stats, horse_stats, elo, elo_hist, horse_races, pedigree = stats_bundle
    else:
        team_stats, horse_stats, elo, elo_hist, horse_races, pedigree = compute_all_stats(
            max_days=max(HISTORY_DAYS, days_back + exclude_recent),
            ref_date=ref_date)

    tasks = []
    for delta in range(exclude_recent + 1, exclude_recent + days_back + 1):
        d = today - timedelta(days=delta)
        date_str = fmt_date(d)
        try:
            prog = get_programme(date_str)
        except Exception:
            continue
        for r in prog["programme"]["reunions"]:
            hippo = r["hippodrome"]["libelleCourt"]
            for c in r["courses"]:
                if c.get("arriveeDefinitive"):
                    tasks.append((date_str, r["numOfficiel"], c["numOrdre"],
                                  c.get("distance"), c.get("discipline"), hippo,
                                  c.get("corde", "")))

    with ThreadPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(_fetch_full, tasks))

    for result in results:
        if not result:
            continue
        parts, perfs, distance, discipline, hippodrome, type_corde = result
        analyses = analyser_course_features(parts, perfs, distance, discipline,
                                             hippodrome, type_corde,
                                             team_stats, horse_stats,
                                             elo, elo_hist, horse_races, pedigree)
        nb = len(analyses)
        for a in analyses:
            X.append(featurize(a, nb))
            real = next((p for p in parts["participants"]
                        if p.get("numPmu") == a["numPmu"]), None)
            y.append(1 if real and real.get("ordreArrivee") == 1 else 0)

    return X, y


def train_ml_model(days_back=21, exclude_recent=0, n_trees_gbm=50, n_trees_rf=30,
                   model_type="ensemble", xgb_n_trees=100,
                   mlp_hidden=(32, 16), mlp_epochs=150):
    """Entraîne un modèle (gbm / rf / xgb / mlp / ensemble / stacking)."""
    try:
        import numpy as np
    except ImportError:
        return None

    X, y = _collect_training_data(days_back, exclude_recent)
    if len(X) < 100:
        return None

    print(f"[ML v6] {len(X)} échantillons, {sum(y)} victoires ({sum(y)/len(X)*100:.1f}%)")

    if model_type == "xgb":
        print(f"[ML v6] Entraînement XGBoost-like ({xgb_n_trees} arbres)...")
        model = XGBoostLike(n_trees=xgb_n_trees, max_depth=4,
                            learning_rate=0.1, lambda_reg=1.0, gamma=0.1,
                            subsample=0.5, early_stopping=10)
        model.fit(X, y)
    elif model_type == "gbm":
        print(f"[ML v6] Entraînement GBM ({n_trees_gbm} arbres)...")
        model = GradientBoosting(n_trees=n_trees_gbm, max_depth=3, learning_rate=0.1)
        model.fit(X, y)
    elif model_type == "rf":
        print(f"[ML v6] Entraînement Random Forest ({n_trees_rf} arbres)...")
        model = RandomForest(n_trees=n_trees_rf, max_depth=8, min_samples=15)
        model.fit(X, y)
    elif model_type == "mlp":
        print(f"[ML v6] Entraînement MLP {mlp_hidden} ({mlp_epochs} epochs)...")
        model = MLPClassifier(hidden_sizes=tuple(mlp_hidden), epochs=mlp_epochs,
                              batch_size=64, dropout=0.2, learning_rate=0.001)
        model.fit(X, y)
        print(f"[ML v6] MLP best epoch: {model.best_epoch}, val_loss: {model.best_val_loss:.4f}")
    elif model_type == "stacking":
        # Stacking : entraîne 3 bases sur 80%, méta-modèle sur 20% restants
        print("[ML v6] Stacking : entraînement des modèles de base + méta...")
        import numpy as np
        np.random.seed(42)
        n = len(X)
        idx = np.arange(n)
        np.random.shuffle(idx)
        n_base = int(n * 0.8)
        base_idx = idx[:n_base]
        meta_idx = idx[n_base:]
        X_base = [X[i] for i in base_idx]
        y_base = [y[i] for i in base_idx]
        X_meta = [X[i] for i in meta_idx]
        y_meta = [y[i] for i in meta_idx]

        print("  - Base 1: XGBoost...")
        b1 = XGBoostLike(n_trees=80, max_depth=4, lambda_reg=1.0, subsample=0.5,
                         early_stopping=10)
        b1.fit(X_base, y_base)
        print("  - Base 2: Random Forest...")
        b2 = RandomForest(n_trees=30, max_depth=8, min_samples=15)
        b2.fit(X_base, y_base)
        print("  - Base 3: MLP...")
        b3 = MLPClassifier(hidden_sizes=(24, 12), epochs=100, dropout=0.2)
        b3.fit(X_base, y_base)

        print("  - Méta-modèle (logistic sur 20% out-of-fold)...")
        model = StackingEnsemble(base_models=[b1, b2, b3])
        model.fit_meta(X_meta, y_meta)
        w = model.get_model_weights()
        print(f"  Poids modèles : XGB={w[0]:.2f}, RF={w[1]:.2f}, MLP={w[2]:.2f}")
    else:  # ensemble
        print(f"[ML v6] Entraînement Ensemble GBM + RF...")
        gbm = GradientBoosting(n_trees=n_trees_gbm, max_depth=3, learning_rate=0.1)
        gbm.fit(X, y)
        rf = RandomForest(n_trees=n_trees_rf, max_depth=8, min_samples=15)
        rf.fit(X, y)
        model = Ensemble(gbm=gbm, rf=rf, w_gbm=0.6, w_rf=0.4)

    print(f"[ML v7] Calibration {CALIB_METHOD} sur hold-out ({int(CALIB_HOLDOUT_FRAC*100)}%)...")
    calibrator = fit_calibrator(model, X, y)
    save_calibration(calibrator.to_dict())
    save_ml_model(model)

    # Évaluation finale
    metrics = evaluate_model(model, X, y)

    info = {"n_samples": len(X), "trained_at": datetime.now().isoformat(),
            "model_type": model_type,
            "log_loss": metrics["log_loss"],
            "auc": metrics["auc"],
            "brier": metrics["brier"]}
    return info


def fit_calibrator(model, X, y, method=CALIB_METHOD,
                   holdout_frac=CALIB_HOLDOUT_FRAC):
    """Ajuste un Calibrator sur un HOLD-OUT (anti-optimisme).

    On réserve les `holdout_frac` derniers échantillons (déjà ordonnés du plus
    récent au plus ancien lors de la collecte) pour la calibration, distincts
    du sous-ensemble d'entraînement du modèle. Si trop peu de données, on
    retombe sur une calibration sur tout le train.
    """
    n = len(X)
    if n < 200:  # pas assez pour un hold-out fiable
        preds = [model.predict_one(x) for x in X]
        return Calibrator.fit(preds, y, method=method)
    k = int(n * (1 - holdout_frac))
    X_cal, y_cal = X[k:], y[k:]
    preds = [model.predict_one(x) for x in X_cal]
    if sum(y_cal) < 5:  # hold-out sans assez de positifs
        preds = [model.predict_one(x) for x in X]
        return Calibrator.fit(preds, y, method=method)
    return Calibrator.fit(preds, y_cal, method=method)


def _apply_calib(p, calibration):
    """Applique une calibration, qu'elle soit au nouveau format Calibrator
    (dict {method, model}) ou à l'ancien format liste de paires (isotone v6)."""
    if not calibration:
        return p
    if isinstance(calibration, dict) and "method" in calibration:
        return Calibrator.from_dict(calibration).apply(p)
    if isinstance(calibration, Calibrator):
        return calibration.apply(p)
    # ancien format : table de paires -> apply_calibration historique
    return apply_calibration(p, calibration)


def predict_ml(features, model, calibration=None):
    p = model.predict_one(features)
    if calibration is not None:
        p = _apply_calib(p, calibration)
    return p


# ============================================================
#  ALGORITHME HYBRIDE v5
# ============================================================
def analyser_course_features(participants_data, perfs_data, distance, discipline,
                              hippodrome, type_corde,
                              team_stats, horse_stats, elo,
                              elo_hist=None, horse_races=None, pedigree=None):
    parts = [p for p in participants_data.get("participants", [])
             if p.get("statut") == "PARTANT"]
    if not parts:
        return []

    perfs_by_num = {}
    for pp in (perfs_data or {}).get("participants", []):
        perfs_by_num[pp.get("numPmu")] = pp.get("coursesCourues", [])

    all_horses = [p.get("nom") for p in parts]
    today_ts = datetime.now().timestamp()
    nb_partants = len(parts)
    pedigree = pedigree or {"peres": {}, "meres": {}}

    analyses = []
    cotes = []
    for p in parts:
        rap = p.get("dernierRapportDirect") or p.get("dernierRapportReference")
        cotes.append(float(rap["rapport"]) if rap and rap.get("rapport") else None)

    inv_cotes = [1.0 / c if c and c > 0 else 0 for c in cotes]
    total_inv = sum(inv_cotes) or 1.0
    proba_marche = [x / total_inv * 100 for x in inv_cotes]

    for i, p in enumerate(parts):
        nb_courses = p.get("nombreCourses", 0) or 0
        nb_vict = p.get("nombreVictoires", 0) or 0
        nb_place = p.get("nombrePlaces", 0) or 0
        cheval = p.get("nom")
        driver = p.get("driver")
        entr = p.get("entraineur")
        pere = p.get("nomPere")
        mere = p.get("nomMere")

        perfs_detail = perfs_by_num.get(p.get("numPmu"), [])
        s_forme = score_forme_enrichi(perfs_detail)

        if nb_courses >= 3:
            s_carriere = min(100, (nb_vict / nb_courses) * 250 + (nb_place / nb_courses) * 80)
        elif nb_courses > 0:
            s_carriere = min(100, (nb_vict / nb_courses) * 200 + 20)
        else:
            s_carriere = 25

        gains = p.get("gainsParticipant", {}) or {}
        gains_carriere = gains.get("gainsCarriere", 0) or 0
        if nb_courses > 0:
            gain_moyen = gains_carriere / nb_courses / 100
            s_gains = min(100, 15 * math.log10(max(gain_moyen, 1) + 1))
        else:
            s_gains = 25

        s_driver = get_team_score_multi(driver, "drivers", team_stats, discipline, hippodrome)
        s_entraineur = get_team_score_multi(entr, "entraineurs", team_stats, discipline)
        s_cheval = get_horse_score(cheval, driver, hippodrome, discipline, horse_stats)
        s_elo = get_elo_score(cheval, elo, all_horses)
        s_distance = score_distance(perfs_detail, distance)
        s_age_sexe = get_age_sexe_score(p.get("age"), p.get("sexe"))
        s_repos = get_repos_score(cheval, today_ts, horse_races or {})
        s_elo_trend = get_elo_trend_score(cheval, elo_hist or {}, elo.get(cheval, 1500))
        adversaires = [h for h in all_horses if h and h != cheval]
        s_confrontation = get_confrontation_score(cheval, adversaires, horse_races or {}, elo)
        s_pedigree = get_pedigree_score(pere, mere, pedigree.get("peres", {}), pedigree.get("meres", {}))
        s_corde = get_corde_score(p.get("numPmu"), nb_partants, type_corde, discipline)
        s_equipment = get_equipment_score(p.get("oeilleres"), p.get("deferre"))
        profile = detect_profile(perfs_detail)
        s_profile_match = get_profile_match_score(profile, distance, nb_partants)

        bonus_team = 0
        if driver and entr and driver == entr: bonus_team = 3
        if p.get("driverChange"): bonus_team -= 5
        bonus_deferre = 2 if "DEFERRE" in (p.get("deferre", "") or "") else 0

        analyses.append({
            "numPmu": p.get("numPmu"),
            "nom": cheval, "age": p.get("age"), "sexe": p.get("sexe"),
            "driver": driver or "—", "entraineur": entr or "—",
            "driverChange": p.get("driverChange", False),
            "musique": p.get("musique", ""),
            "nbCourses": nb_courses, "nbVictoires": nb_vict, "nbPlaces": nb_place,
            "cote": cotes[i], "probaMarche": round(proba_marche[i], 2),
            "gainsCarriere": gains_carriere // 100,
            "deferre": p.get("deferre", ""),
            "oeilleres": p.get("oeilleres", ""),
            "pere": pere, "mere": mere,
            "urlCasaque": p.get("urlCasaque"),
            "ordreArrivee": p.get("ordreArrivee"),
            "profile": profile,
            "scores": {
                "marche": round(proba_marche[i], 1),
                "forme": round(s_forme, 1),
                "carriere": round(s_carriere, 1),
                "gains": round(s_gains, 1),
                "driver": round(s_driver, 1),
                "entraineur": round(s_entraineur, 1),
                "distance": round(s_distance, 1),
                "cheval_stats": round(s_cheval, 1),
                "elo": round(s_elo, 1),
                "age_sexe": round(s_age_sexe, 1),
                "repos": round(s_repos, 1),
                "elo_trend": round(s_elo_trend, 1),
                "confrontation": round(s_confrontation, 1),
                "pedigree": round(s_pedigree, 1),
                "corde": round(s_corde, 1),
                "equipment": round(s_equipment, 1),
                "profile_match": round(s_profile_match, 1),
            },
            "bonus": {"team": bonus_team, "deferre": bonus_deferre},
        })

    return analyses


def analyser_course(participants_data, perfs_data=None, distance=None,
                    discipline=None, hippodrome=None, type_corde=None,
                    team_stats=None, horse_stats=None, elo=None,
                    elo_hist=None, horse_races=None, pedigree=None,
                    use_ml=False, capital=100, ml_model=None, calib=None):
    analyses = analyser_course_features(participants_data, perfs_data, distance,
                                         discipline, hippodrome, type_corde,
                                         team_stats, horse_stats, elo,
                                         elo_hist, horse_races, pedigree)
    if not analyses:
        return []

    proba_marche_list = [a["probaMarche"] for a in analyses]
    nb_partants = len(analyses)

    scores_intr = []
    for a in analyses:
        s = (0.15 * a["scores"]["forme"] +
             0.08 * a["scores"]["carriere"] +
             0.07 * a["scores"]["gains"] +
             0.09 * a["scores"]["driver"] +
             0.06 * a["scores"]["entraineur"] +
             0.07 * a["scores"]["distance"] +
             0.09 * a["scores"]["cheval_stats"] +
             0.11 * a["scores"]["elo"] +
             0.04 * a["scores"]["age_sexe"] +
             0.04 * a["scores"]["repos"] +
             0.05 * a["scores"]["elo_trend"] +
             0.03 * a["scores"]["confrontation"] +
             0.06 * a["scores"]["pedigree"] +
             0.03 * a["scores"]["corde"] +
             0.02 * a["scores"]["equipment"] +
             0.01 * a["scores"]["profile_match"] +
             a["bonus"]["team"] + a["bonus"]["deferre"])
        scores_intr.append(max(s, 1))

    total_intr = sum(scores_intr) or 1
    proba_intr = [s / total_intr * 100 for s in scores_intr]

    chances_heur = [0.55 * proba_marche_list[i] + 0.45 * proba_intr[i]
                    for i in range(len(analyses))]
    total = sum(chances_heur) or 1
    chances_heur = [c / total * 100 for c in chances_heur]

    # ml_model/calib peuvent être injectés (walk-forward) ; sinon on charge du disque
    if use_ml and ml_model is None:
        ml_model = load_ml_model()
        if calib is None:
            calib = load_calibration()
    chances_ml = None
    if ml_model:
        nb = len(analyses)
        raw_ml = [predict_ml(featurize(a, nb), ml_model, calib) for a in analyses]
        total_ml = sum(raw_ml) or 1
        chances_ml = [x / total_ml * 100 for x in raw_ml]

    w = ML_BLEND_WEIGHT
    for i, a in enumerate(analyses):
        if chances_ml:
            a["chance"] = round((1 - w) * chances_heur[i] + w * chances_ml[i], 2)
            a["chanceML"] = round(chances_ml[i], 2)
        else:
            a["chance"] = round(chances_heur[i], 2)
        a["chanceHeur"] = round(chances_heur[i], 2)

        if a["cote"] and a["probaMarche"] > 0:
            edge = a["chance"] - a["probaMarche"]
            a["edge"] = round(edge, 2)
            a["valueBet"] = edge > 4 and a["cote"] >= 4
            p = a["chance"] / 100
            a["kellyMise"] = kelly_amount(p, a["cote"], capital, kelly_mult=0.25)
            a["kellyFraction"] = round(kelly_fraction(p, a["cote"], 0.25) * 100, 2)
            a["expectedROI"] = round(expected_roi(p, a["cote"]), 2)
        else:
            a["edge"] = 0
            a["valueBet"] = False
            a["kellyMise"] = 0
            a["kellyFraction"] = 0
            a["expectedROI"] = 0

    total = sum(a["chance"] for a in analyses) or 1
    for a in analyses:
        a["chance"] = round(a["chance"] / total * 100, 2)

    analyses.sort(key=lambda x: -x["chance"])
    for rank, a in enumerate(analyses, 1):
        a["rang"] = rank

    # NEW v5 : proba placé (top 3)
    chances_list = [a["chance"] for a in analyses]
    places_3 = proba_place_simple(chances_list, n_places=3, nb_partants=nb_partants)
    places_2 = proba_place_simple(chances_list, n_places=2, nb_partants=nb_partants)
    for i, a in enumerate(analyses):
        a["chancePlace3"] = round(places_3[i], 2)
        a["chancePlace2"] = round(places_2[i], 2)

    return analyses


# ============================================================
#  Backtest v5
# ============================================================
def backtest(days_back=7, use_ml=False):
    team_stats, horse_stats, elo, elo_hist, horse_races, pedigree = compute_all_stats(
        max_days=HISTORY_DAYS)
    today = datetime.now()
    results = {
        "total_courses": 0, "top1_winner": 0, "top1_top3": 0, "top3_winner": 0,
        "value_bets": [], "mise_totale": 0.0, "gain_total": 0.0,
        "kelly_mise_totale": 0.0, "kelly_gain_total": 0.0,
    }

    tasks = []
    metas = []
    for delta in range(1, days_back + 1):
        d = today - timedelta(days=delta)
        date_str = fmt_date(d)
        try:
            prog = get_programme(date_str)
        except Exception:
            continue
        for r in prog["programme"]["reunions"]:
            hippo = r["hippodrome"]["libelleCourt"]
            for c in r["courses"]:
                if c.get("arriveeDefinitive"):
                    tasks.append((date_str, r["numOfficiel"], c["numOrdre"],
                                  c.get("distance"), c.get("discipline"), hippo,
                                  c.get("corde", "")))
                    metas.append({"date": d.strftime("%d/%m"),
                                  "course": f"R{r['numOfficiel']}C{c['numOrdre']}"})

    with ThreadPoolExecutor(max_workers=20) as ex:
        fetched = list(ex.map(_fetch_full, tasks))

    for result, meta in zip(fetched, metas):
        if not result:
            continue
        parts, perfs, distance, discipline, hippodrome, type_corde = result
        analyses = analyser_course(parts, perfs, distance, discipline, hippodrome,
                                    type_corde,
                                    team_stats, horse_stats, elo, elo_hist,
                                    horse_races, pedigree, use_ml=use_ml,
                                    capital=100)
        if not analyses:
            continue

        results["total_courses"] += 1
        vainqueur = next((a for a in analyses if a["ordreArrivee"] == 1), None)
        if not vainqueur:
            continue

        top1 = analyses[0]
        if top1["ordreArrivee"] == 1: results["top1_winner"] += 1
        if top1["ordreArrivee"] and 1 <= top1["ordreArrivee"] <= 3: results["top1_top3"] += 1
        if any(a["ordreArrivee"] == 1 for a in analyses[:3]): results["top3_winner"] += 1

        results["mise_totale"] += 1
        if top1["ordreArrivee"] == 1 and top1["cote"]:
            results["gain_total"] += top1["cote"]

        for a in analyses:
            if a.get("valueBet"):
                results["value_bets"].append({
                    "course": meta["course"], "date": meta["date"],
                    "cheval": a["nom"], "cote": a["cote"], "edge": a["edge"],
                    "gagne": a["ordreArrivee"] == 1,
                    "kellyMise": a.get("kellyMise", 0),
                })
                km = a.get("kellyMise", 0)
                if km > 0:
                    results["kelly_mise_totale"] += km
                    if a["ordreArrivee"] == 1 and a["cote"]:
                        results["kelly_gain_total"] += km * a["cote"]

    n = results["total_courses"] or 1
    results["taux_top1"] = round(results["top1_winner"] / n * 100, 2)
    results["taux_top1_place"] = round(results["top1_top3"] / n * 100, 2)
    results["taux_top3"] = round(results["top3_winner"] / n * 100, 2)
    results["roi"] = round((results["gain_total"] - results["mise_totale"]) /
                           max(results["mise_totale"], 1) * 100, 2)
    results["mise_totale"] = round(results["mise_totale"], 2)
    results["gain_total"] = round(results["gain_total"], 2)

    km_tot = results["kelly_mise_totale"]
    kg_tot = results["kelly_gain_total"]
    results["kelly_roi"] = round((kg_tot - km_tot) / max(km_tot, 1) * 100, 2) if km_tot else 0
    results["kelly_profit"] = round(kg_tot - km_tot, 2)
    results["kelly_mise_totale"] = round(km_tot, 2)
    results["kelly_gain_total"] = round(kg_tot, 2)

    vb = results["value_bets"]
    if vb:
        gains_vb = sum((b["cote"] if b["gagne"] else 0) for b in vb)
        results["vb_nb"] = len(vb)
        results["vb_winrate"] = round(sum(1 for b in vb if b["gagne"]) / len(vb) * 100, 2)
        results["vb_roi"] = round((gains_vb - len(vb)) / len(vb) * 100, 2)
    else:
        results["vb_nb"] = 0; results["vb_winrate"] = 0; results["vb_roi"] = 0

    results["value_bets"] = results["value_bets"][-30:]
    return results


# ============================================================
#  WALK-FORWARD BACKTEST (validation temporelle rigoureuse)
# ============================================================
def _make_model(model_type):
    """Fabrique un modèle frais selon le type (mêmes hyperparams que train_ml_model
    mais en version 'rapide' pour les multiples folds du walk-forward)."""
    if model_type == "xgb":
        return XGBoostLike(n_trees=60, max_depth=4, learning_rate=0.1,
                           lambda_reg=1.0, gamma=0.1, subsample=0.5,
                           early_stopping=10)
    if model_type == "gbm":
        return GradientBoosting(n_trees=40, max_depth=3, learning_rate=0.1)
    if model_type == "rf":
        return RandomForest(n_trees=25, max_depth=8, min_samples=15)
    if model_type == "mlp":
        return MLPClassifier(hidden_sizes=(24, 12), epochs=100, batch_size=64,
                             dropout=0.2, learning_rate=0.001)
    if model_type == "ensemble":
        gbm = GradientBoosting(n_trees=40, max_depth=3, learning_rate=0.1)
        rf = RandomForest(n_trees=25, max_depth=8, min_samples=15)
        return ("ensemble", gbm, rf)
    # défaut : xgb
    return XGBoostLike(n_trees=60, max_depth=4, subsample=0.5, early_stopping=10)


def _fit_fold_model(model_type, X, y):
    """Entraîne le modèle du fold + calibration isotone OUT-OF-SAMPLE interne."""
    spec = _make_model(model_type)
    if isinstance(spec, tuple) and spec[0] == "ensemble":
        _, gbm, rf = spec
        gbm.fit(X, y)
        rf.fit(X, y)
        model = Ensemble(gbm=gbm, rf=rf, w_gbm=0.6, w_rf=0.4)
    else:
        model = spec
        model.fit(X, y)
    # Calibration hold-out (v7.1) — cohérent avec train_ml_model, sans optimisme
    calibrator = fit_calibrator(model, X, y)
    return model, calibrator


def walk_forward_backtest(n_folds=4, train_window=30, test_window=7, gap=1,
                          model_type="xgb", mode="rolling", use_ml=True,
                          stats_window=None):
    """
    Backtest temporel sans look-ahead.

    Pour chaque fold :
      1. stats calculées UNIQUEMENT jusqu'à train_end (ref_date=train_end+1j)
      2. modèle ML entraîné sur les courses [train_start, train_end]
      3. évaluation sur les courses [test_start, test_end] avec ce modèle-là
    Aucune information postérieure à la course testée n'est utilisée.
    """
    ref_date = datetime.now()
    if stats_window is None:
        stats_window = HISTORY_DAYS

    windows = generate_windows(ref_date, n_folds=n_folds,
                               train_window=train_window,
                               test_window=test_window, gap=gap, mode=mode)

    fold_results = []
    for f in windows:
        train_start, train_end = f["train_start"], f["train_end"]
        test_start, test_end = f["test_start"], f["test_end"]

        # --- Stats "telles qu'on les connaissait" juste après train_end ---
        # ref_date = lendemain de train_end => n'inclut que train_end et avant.
        stats_ref = train_end + timedelta(days=1)
        stats_bundle = compute_all_stats(max_days=stats_window, ref_date=stats_ref)

        fold = {"fold": f["fold"], **fmt_window(f),
                "n_test_courses": 0, "n_test_samples": 0,
                "top1_correct": 0, "top3_correct": 0,
                "log_loss": 0.0, "auc": 0.0, "brier": 0.0,
                "kelly_mise": 0.0, "kelly_gain": 0.0,
                "value_bets": 0, "value_bets_won": 0, "top1_rate": 0.0}

        # --- Entraînement (si ML) sur la fenêtre de train, stats=stats_bundle ---
        ml_model, calib = None, None
        if use_ml:
            tw = (train_end - train_start).days + 1
            X_tr, y_tr = _collect_training_data(
                days_back=tw, exclude_recent=0,
                ref_date=train_end + timedelta(days=1),
                stats_bundle=stats_bundle)
            if len(X_tr) >= 80:
                try:
                    ml_model, calib = _fit_fold_model(model_type, X_tr, y_tr)
                except Exception as e:
                    print(f"[WF] fold {f['fold']} train échoué: {e}")
                    ml_model, calib = None, None

        # --- Évaluation sur la fenêtre de test, course par course ---
        team_stats, horse_stats, elo, elo_hist, horse_races, pedigree = stats_bundle
        y_true_all, y_pred_all = [], []

        tasks = []
        for d in daterange_dt(test_start, test_end):
            date_str = fmt_date(d)
            try:
                prog = get_programme(date_str)
            except Exception:
                continue
            for r in prog["programme"]["reunions"]:
                hippo = r["hippodrome"]["libelleCourt"]
                for c in r["courses"]:
                    if c.get("arriveeDefinitive"):
                        tasks.append((date_str, r["numOfficiel"], c["numOrdre"],
                                      c.get("distance"), c.get("discipline"),
                                      hippo, c.get("corde", "")))

        with ThreadPoolExecutor(max_workers=20) as ex:
            fetched = list(ex.map(_fetch_full, tasks))

        for result in fetched:
            if not result:
                continue
            parts, perfs, distance, discipline, hippodrome, type_corde = result
            analyses = analyser_course(
                parts, perfs, distance, discipline, hippodrome, type_corde,
                team_stats, horse_stats, elo, elo_hist, horse_races, pedigree,
                use_ml=bool(ml_model), capital=100,
                ml_model=ml_model, calib=calib)
            if not analyses:
                continue
            vainqueur = next((a for a in analyses if a["ordreArrivee"] == 1), None)
            if not vainqueur:
                continue

            fold["n_test_courses"] += 1
            nb = len(analyses)
            # Probas (chance normalisée -> proba d'être 1er) pour métriques de classif
            for a in analyses:
                y_true_all.append(1 if a["ordreArrivee"] == 1 else 0)
                y_pred_all.append((a["chance"] or 0) / 100.0)
                fold["n_test_samples"] += 1

            top1 = analyses[0]
            if top1["ordreArrivee"] == 1:
                fold["top1_correct"] += 1
            if any(a["ordreArrivee"] == 1 for a in analyses[:3]):
                fold["top3_correct"] += 1

            for a in analyses:
                if a.get("valueBet"):
                    fold["value_bets"] += 1
                    won = a["ordreArrivee"] == 1
                    if won:
                        fold["value_bets_won"] += 1
                    km = a.get("kellyMise", 0) or 0
                    if km > 0:
                        fold["kelly_mise"] += km
                        if won and a["cote"]:
                            fold["kelly_gain"] += km * a["cote"]

        # Métriques de classification du fold (sans fuite)
        if y_true_all and any(y_true_all):
            fold["log_loss"] = round(log_loss(y_true_all, y_pred_all), 4)
            fold["auc"] = round(roc_auc(y_true_all, y_pred_all), 4)
            fold["brier"] = round(brier_score(y_true_all, y_pred_all), 4)
        nc = fold["n_test_courses"] or 1
        fold["top1_rate"] = round(fold["top1_correct"] / nc * 100, 2)
        fold["top3_rate"] = round(fold["top3_correct"] / nc * 100, 2)
        fold_results.append(fold)

    summary = aggregate_fold_metrics(fold_results)
    summary["params"] = {"n_folds": n_folds, "train_window": train_window,
                         "test_window": test_window, "gap": gap,
                         "model_type": model_type, "mode": mode,
                         "use_ml": use_ml}
    return summary


def daterange_dt(start, end):
    """Itère sur chaque jour de start à end (inclus) — version datetime locale."""
    d = start
    while d <= end:
        yield d
        d = d + timedelta(days=1)


# ============================================================
#  ROUTES
# ============================================================
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/backtest")
def backtest_page():
    return render_template("backtest.html")


@app.route("/paris")
def paris_page():
    return render_template("paris.html")


@app.route("/dashboard")
def dashboard_page():
    return render_template("dashboard.html")


@app.route("/api/reunions")
def api_reunions():
    date_str = request.args.get("date") or fmt_date(datetime.now())
    try:
        prog = get_programme(date_str)
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    out = []
    for r in prog["programme"]["reunions"]:
        out.append({
            "numReunion": r["numOfficiel"],
            "hippodrome": r["hippodrome"]["libelleCourt"],
            "courses": [{
                "numCourse": c["numOrdre"],
                "libelle": c.get("libelle") or c.get("libelleCourt"),
                "discipline": c.get("discipline"),
                "distance": c.get("distance"),
                "heure": datetime.fromtimestamp(c["heureDepart"] / 1000).strftime("%H:%M") if c.get("heureDepart") else "",
                "nbPartants": c.get("nombreDeclaresPartants"),
                "arriveeDefinitive": c.get("arriveeDefinitive", False),
            } for c in r["courses"]],
        })
    return jsonify({"date": date_str, "reunions": out})


@app.route("/api/course/<int:r_num>/<int:c_num>")
def api_course(r_num, c_num):
    date_str = request.args.get("date") or fmt_date(datetime.now())
    use_ml = request.args.get("ml") == "1"
    live = request.args.get("live") == "1"
    capital = float(request.args.get("capital", 100))
    try:
        prog = get_programme(date_str)
        parts = get_participants_live(date_str, r_num, c_num) if live else get_participants(date_str, r_num, c_num)
        perfs = get_performances(date_str, r_num, c_num)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    course_info = None
    reunion_info = None
    discipline = None
    hippodrome = None
    type_corde = None
    for r in prog["programme"]["reunions"]:
        if r["numOfficiel"] == r_num:
            hippodrome = r["hippodrome"]["libelleCourt"]
            reunion_info = {"hippodrome": hippodrome}
            for c in r["courses"]:
                if c["numOrdre"] == c_num:
                    discipline = c.get("discipline")
                    type_corde = c.get("corde", "")
                    course_info = {
                        "libelle": c.get("libelle"),
                        "discipline": discipline,
                        "distance": c.get("distance"),
                        "specialite": c.get("specialite"),
                        "corde": type_corde,
                        "heure": datetime.fromtimestamp(c["heureDepart"] / 1000).strftime("%H:%M") if c.get("heureDepart") else "",
                        "montantPrix": c.get("montantPrix"),
                        "nbPartants": c.get("nombreDeclaresPartants"),
                        "arriveeDefinitive": c.get("arriveeDefinitive", False),
                        "ordreArrivee": c.get("ordreArrivee"),
                    }

    team_stats, horse_stats, elo, elo_hist, horse_races, pedigree = compute_all_stats(
        max_days=HISTORY_DAYS)
    analyses = analyser_course(parts, perfs,
                                course_info.get("distance") if course_info else None,
                                discipline, hippodrome, type_corde,
                                team_stats, horse_stats, elo, elo_hist,
                                horse_races, pedigree, use_ml=use_ml,
                                capital=capital)

    # NEW v5 : combinaisons multi-paris
    combinations_data = best_combinations(analyses, n_top=5) if analyses else {}

    return jsonify({
        "date": date_str, "reunion": reunion_info, "course": course_info,
        "analyses": analyses,
        "combinations": combinations_data,
        "ml_active": use_ml and load_ml_model() is not None,
        "live": live,
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/backtest")
def api_backtest():
    days = int(request.args.get("days", 7))
    use_ml = request.args.get("ml") == "1"
    days = min(days, 30)
    try:
        return jsonify(backtest(days_back=days, use_ml=use_ml))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest/walkforward", methods=["POST"])
def api_walkforward():
    """Backtest temporel rigoureux (walk-forward, sans look-ahead)."""
    data = request.get_json(silent=True) or {}
    try:
        n_folds = min(max(int(data.get("n_folds", 4)), 1), 8)
        train_window = min(max(int(data.get("train_window", 30)), 5), 120)
        test_window = min(max(int(data.get("test_window", 7)), 1), 30)
        gap = min(max(int(data.get("gap", 1)), 0), 7)
        model_type = data.get("model_type", "xgb")
        if model_type not in ("xgb", "gbm", "rf", "mlp", "ensemble"):
            model_type = "xgb"
        mode = data.get("mode", "rolling")
        if mode not in ("rolling", "expanding"):
            mode = "rolling"
        use_ml = bool(data.get("use_ml", True))

        result = walk_forward_backtest(
            n_folds=n_folds, train_window=train_window,
            test_window=test_window, gap=gap, model_type=model_type,
            mode=mode, use_ml=use_ml)
        return jsonify(result)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
def api_train():
    days = int(request.args.get("days", 21))
    days = min(days, 30)
    n_trees_gbm = int(request.args.get("trees_gbm", 50))
    n_trees_rf = int(request.args.get("trees_rf", 30))
    xgb_n_trees = int(request.args.get("trees_xgb", 100))
    model_type = request.args.get("type", "ensemble")
    try:
        info = train_ml_model(days_back=days, n_trees_gbm=n_trees_gbm,
                              n_trees_rf=n_trees_rf, model_type=model_type,
                              xgb_n_trees=xgb_n_trees)
        if info is None:
            return jsonify({"error": "Pas assez de données"}), 400
        return jsonify({"ok": True, **info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/team-stats")
def api_team_stats():
    team_stats, _, _, _, _, _ = compute_all_stats(max_days=HISTORY_DAYS)
    drivers = sorted(team_stats["drivers"].items(),
                    key=lambda x: -(x[1]["v"] if x[1]["c"] >= 10 else 0))[:30]
    entr = sorted(team_stats["entraineurs"].items(),
                 key=lambda x: -(x[1]["v"] if x[1]["c"] >= 10 else 0))[:30]
    return jsonify({
        "drivers": [{"nom": k, "courses": v["c"], "victoires": v["v"], "places": v["p"],
                    "taux_victoire": round(v["v"]/v["c"]*100, 1) if v["c"] else 0}
                   for k, v in drivers],
        "entraineurs": [{"nom": k, "courses": v["c"], "victoires": v["v"], "places": v["p"],
                        "taux_victoire": round(v["v"]/v["c"]*100, 1) if v["c"] else 0}
                       for k, v in entr],
    })


# ============================================================
#  v5 - Paris (SQLite)
# ============================================================
@app.route("/api/bets", methods=["GET"])
def api_bets_list():
    statut = request.args.get("statut")
    bets = db.list_bets(statut=statut, limit=500)
    stats = db.compute_stats()
    return jsonify({"bets": bets, "stats": stats})


@app.route("/api/bets", methods=["POST"])
def api_bets_add():
    data = request.get_json() or {}
    if not all(k in data for k in ["cheval", "cote", "mise"]):
        return jsonify({"error": "Missing fields"}), 400
    bet = db.add_bet(data)
    return jsonify(bet)


@app.route("/api/bets/<int:bet_id>", methods=["PUT"])
def api_bets_update(bet_id):
    data = request.get_json() or {}
    gagne = bool(data.get("gagne"))
    place = data.get("place")
    db.update_bet_result(bet_id, gagne, place)
    return jsonify({"ok": True})


@app.route("/api/bets/<int:bet_id>", methods=["DELETE"])
def api_bets_delete(bet_id):
    db.delete_bet(bet_id)
    return jsonify({"ok": True})


# ============================================================
#  v5 - Dashboard
# ============================================================
@app.route("/api/dashboard")
def api_dashboard():
    days = int(request.args.get("days", 30))
    return jsonify({
        "stats_global": db.compute_stats(),
        "stats_30j": db.compute_stats(days=30),
        "stats_7j": db.compute_stats(days=7),
        "stats_par_hippodrome": db.stats_by_dimension("hippodrome"),
        "stats_par_discipline": db.stats_by_dimension("discipline"),
        "stats_par_type": db.stats_by_dimension("type_pari"),
        "evolution_profit": db.cumulative_profit(),
    })


# ============================================================
#  v5 - Watchlist
# ============================================================
@app.route("/api/watchlist", methods=["GET"])
def api_watchlist():
    return jsonify(db.get_watchlist())


@app.route("/api/watchlist", methods=["POST"])
def api_watchlist_add():
    data = request.get_json() or {}
    cheval = data.get("cheval")
    notes = data.get("notes", "")
    if cheval:
        db.add_to_watchlist(cheval, notes)
    return jsonify({"ok": True})


@app.route("/api/watchlist/<cheval>", methods=["DELETE"])
def api_watchlist_remove(cheval):
    db.remove_from_watchlist(cheval)
    return jsonify({"ok": True})


# ============================================================
#  v5 - Alertes config
# ============================================================
@app.route("/api/alerts-config", methods=["GET"])
def api_alerts_get():
    return jsonify(db.get_alerts_config())


@app.route("/api/alerts-config", methods=["POST"])
def api_alerts_set():
    data = request.get_json() or {}
    db.update_alerts_config(
        min_edge=float(data.get("min_edge", 5.0)),
        min_cote=float(data.get("min_cote", 4.0)),
        max_cote=float(data.get("max_cote", 50.0)),
        enabled=bool(data.get("enabled", True))
    )
    return jsonify({"ok": True})


# ============================================================
#  v5 - Alertes : scan toutes les courses du jour pour value bets
# ============================================================
@app.route("/api/scan-alerts")
def api_scan_alerts():
    """Scanne toutes les courses du jour et retourne les value bets matchant les critères."""
    date_str = request.args.get("date") or fmt_date(datetime.now())
    use_ml = request.args.get("ml") == "1"

    config = db.get_alerts_config()
    if not config.get("enabled"):
        return jsonify({"alerts": [], "config": config})

    try:
        prog = get_programme(date_str)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    team_stats, horse_stats, elo, elo_hist, horse_races, pedigree = compute_all_stats(
        max_days=HISTORY_DAYS)

    alerts = []
    for r in prog["programme"]["reunions"]:
        hippo = r["hippodrome"]["libelleCourt"]
        for c in r["courses"]:
            # Skip courses passées
            if c.get("arriveeDefinitive"):
                continue
            try:
                parts = get_participants_live(date_str, r["numOfficiel"], c["numOrdre"])
                perfs = get_performances(date_str, r["numOfficiel"], c["numOrdre"])
            except Exception:
                continue

            analyses = analyser_course(parts, perfs, c.get("distance"),
                                        c.get("discipline"), hippo, c.get("corde", ""),
                                        team_stats, horse_stats, elo, elo_hist,
                                        horse_races, pedigree, use_ml=use_ml)
            for a in analyses:
                if (a.get("edge", 0) >= config["min_edge"]
                    and a.get("cote", 0) >= config["min_cote"]
                    and a.get("cote", 999) <= config["max_cote"]):
                    alerts.append({
                        "course": f"R{r['numOfficiel']}C{c['numOrdre']}",
                        "hippodrome": hippo,
                        "heure": datetime.fromtimestamp(c["heureDepart"] / 1000).strftime("%H:%M") if c.get("heureDepart") else "",
                        "cheval": a["nom"],
                        "numPmu": a["numPmu"],
                        "cote": a["cote"],
                        "chance": a["chance"],
                        "edge": a["edge"],
                        "kellyMise": a.get("kellyMise", 0),
                    })

    # Tri par edge décroissant
    alerts.sort(key=lambda x: -x["edge"])
    return jsonify({"alerts": alerts[:50], "config": config})


# ============================================================
#  v5 - Geny data (terrain, météo)
# ============================================================
@app.route("/api/geny/<int:r_num>/<int:c_num>")
def api_geny(r_num, c_num):
    date_str = request.args.get("date") or fmt_date(datetime.now())
    try:
        prog = get_programme(date_str)
        course = None
        for r in prog["programme"]["reunions"]:
            if r["numOfficiel"] == r_num:
                for c in r["courses"]:
                    if c["numOrdre"] == c_num:
                        course = c
                        break
        if not course:
            return jsonify({"error": "Course not found"}), 404

        data = geny_scraper.get_geny_data(date_str, r_num, c_num, course.get("libelle", ""))
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e), "terrain": None, "meteo": None,
                       "pronostics_presse": {}}), 200


# ============================================================
#  v6 - Comparaison de modèles + AutoML
# ============================================================
@app.route("/models")
def models_page():
    return render_template("models.html")


@app.route("/api/models/compare", methods=["POST"])
def api_models_compare():
    """Entraîne plusieurs modèles et les compare via CV."""
    days = int(request.args.get("days", 15))
    days = min(days, 30)
    try:
        X, y = _collect_training_data(days, 0)
        if len(X) < 200:
            return jsonify({"error": "Pas assez de données (min 200)"}), 400

        print(f"[Compare] Entraînement de 4 modèles sur {len(X)} échantillons...")
        results = []

        # GBM
        print("  - GBM...")
        m = GradientBoosting(n_trees=50, max_depth=3, learning_rate=0.1)
        m.fit(X, y)
        ev = evaluate_model(m, X, y)
        results.append({"name": "Gradient Boosting", "type": "gbm", **ev})

        # Random Forest
        print("  - Random Forest...")
        m = RandomForest(n_trees=30, max_depth=8, min_samples=15)
        m.fit(X, y)
        ev = evaluate_model(m, X, y)
        results.append({"name": "Random Forest", "type": "rf", **ev})

        # XGBoost-like
        print("  - XGBoost-like...")
        m = XGBoostLike(n_trees=80, max_depth=4, lambda_reg=1.0,
                        subsample=0.5, early_stopping=10)
        m.fit(X, y)
        ev = evaluate_model(m, X, y)
        results.append({"name": "XGBoost-like", "type": "xgb", **ev,
                       "n_trees_used": m.best_n_trees})

        # MLP
        print("  - MLP...")
        m = MLPClassifier(hidden_sizes=(32, 16), epochs=100, dropout=0.2)
        m.fit(X, y)
        ev = evaluate_model(m, X, y)
        results.append({"name": "Neural Network (MLP)", "type": "mlp", **ev,
                       "best_epoch": m.best_epoch})

        # Pour éviter renvoi de la calibration entière (verbeux)
        for r in results:
            r["calibration_n_bins"] = len(r.pop("calibration", []))

        # Trie par log_loss croissant
        results.sort(key=lambda r: r["log_loss"])

        return jsonify({
            "n_samples": len(X),
            "n_wins": sum(y),
            "win_rate": round(sum(y) / len(y) * 100, 2),
            "models": results,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/models/automl", methods=["POST"])
def api_models_automl():
    """Recherche aléatoire des meilleurs hyperparamètres pour MLP."""
    days = int(request.args.get("days", 15))
    days = min(days, 30)
    try:
        X, y = _collect_training_data(days, 0)
        if len(X) < 200:
            return jsonify({"error": "Pas assez de données"}), 400

        print("[AutoML] Random search sur MLP (12 combinaisons, 2-fold CV)...")
        param_grid = {
            "hidden_sizes": [(16, 8), (32, 16), (32, 16, 8), (64, 32)],
            "dropout": [0.0, 0.2, 0.3],
            "learning_rate": [0.001, 0.005],
            "epochs": [80],  # fixe pour la vitesse
        }
        result = random_search(MLPClassifier, param_grid, X, y,
                                n_iter=8, n_folds=2)
        # Convertir hidden_sizes en str pour JSON
        for r in result["all_results"]:
            if "params" in r and "hidden_sizes" in r["params"]:
                r["params"]["hidden_sizes"] = str(r["params"]["hidden_sizes"])
        if result["best_params"]:
            result["best_params"]["hidden_sizes"] = str(result["best_params"]["hidden_sizes"])
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/explain/<int:r_num>/<int:c_num>/<int:num_pmu>")
def api_explain(r_num, c_num, num_pmu):
    """
    Explique pourquoi le modèle prédit X% pour ce cheval.
    Calcule l'importance de chaque feature par perturbation.
    """
    date_str = request.args.get("date") or fmt_date(datetime.now())
    try:
        prog = get_programme(date_str)
        parts = get_participants(date_str, r_num, c_num)
        perfs = get_performances(date_str, r_num, c_num)

        discipline = None
        hippodrome = None
        type_corde = None
        distance = None
        for r in prog["programme"]["reunions"]:
            if r["numOfficiel"] == r_num:
                hippodrome = r["hippodrome"]["libelleCourt"]
                for c in r["courses"]:
                    if c["numOrdre"] == c_num:
                        discipline = c.get("discipline")
                        type_corde = c.get("corde", "")
                        distance = c.get("distance")

        team_stats, horse_stats, elo, elo_hist, horse_races, pedigree = compute_all_stats()
        analyses = analyser_course_features(parts, perfs, distance, discipline,
                                             hippodrome, type_corde,
                                             team_stats, horse_stats, elo,
                                             elo_hist, horse_races, pedigree)
        target = next((a for a in analyses if a["numPmu"] == num_pmu), None)
        if not target:
            return jsonify({"error": "Cheval introuvable"}), 404

        ml_model = load_ml_model()
        if not ml_model:
            return jsonify({"error": "Aucun modèle ML entraîné"}), 400

        nb = len(analyses)
        features = featurize(target, nb)
        base_pred = ml_model.predict_one(features)
        importances = feature_importance_perturbation(ml_model, features,
                                                       FEATURE_NAMES, n_perturb=1)

        return jsonify({
            "cheval": target["nom"],
            "numPmu": num_pmu,
            "prediction_ml": round(base_pred * 100, 2),
            "feature_importances": importances,
            "explanation": "Impact = combien la prédiction diminuerait si cette feature passait à 50 (valeur neutre)",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # Désactive le chargement auto de .env (bug avec OneDrive sur Windows)
    import os
    os.environ["FLASK_SKIP_DOTENV"] = "1"
    app.run(host="0.0.0.0", port=5000, debug=False, load_dotenv=False)
