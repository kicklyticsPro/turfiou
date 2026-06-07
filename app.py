"""
Turf Analyzer v6 - Analyse hippique PMU professionnelle

v6 nouveautés :
  Features (27 → 41) :
    - Taux driver/hippodrome, régularité, équipement, style, tendance gains...
    + 2 interactions : régularité×forme, driver_hippo×terrain

  Modèle DUAL Win + Top 4 :
    - Modèle WIN  : y=1 si gagnant    (~8% positifs, classement)
    - Modèle TOP4 : y=1 si ≤4e place  (~40% positifs, placement)
    - Mêmes 41 features, labels différents
    - Top 4 = beaucoup plus stable et calibré (4× plus de signal)
    - Backtest enrichi : accuracy Top 4 par bucket de confiance

v5 :
  Ensemble Stacking (LGBM+CatBoost+HGB+RF+LR), Calibration Platt/Isotone

v4 :
  Ensemble GBM+RF, Pedigree, Corde, Équipements, Profils, Kelly, Live
"""

from flask import Flask, jsonify, render_template, request, session, redirect, url_for
from functools import wraps
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
from lib.kelly import kelly_amount, kelly_fraction, expected_value, expected_roi
from lib.features_v4 import (build_pedigree_stats, get_pedigree_score,
                              get_corde_score, get_equipment_score,
                              detect_profile, get_profile_match_score,
                              compute_gains_trend, compute_terrain_perf,
                              detect_equipment_change, compute_days_since_last,
                              compute_nb_courses_recent, compute_corde_avantage)
from lib import bets_tracker

# NEW v5 - modèle avancé
try:
    from lib.ml_advanced import train_advanced, load_advanced
    HAS_ADVANCED = True
except:
    HAS_ADVANCED = False

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "turf-analyzer-secret-2026")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")


def admin_required(f):
    """Protège les routes admin avec un mot de passe session."""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin"):
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated

PMU_BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/61/programme"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TurfAnalyzer/4.0)"}

CACHE_DIR = os.environ.get("CACHE_DIR", "/tmp/turf_cache")
try:
    os.makedirs(CACHE_DIR, exist_ok=True)
except Exception:
    CACHE_DIR = "/tmp/turf_cache"
    os.makedirs(CACHE_DIR, exist_ok=True)

# Caches v4
STATS_CACHE_FILE = os.path.join(CACHE_DIR, "stats_team_v4.pkl")
HORSE_STATS_FILE = os.path.join(CACHE_DIR, "horse_stats_v4.pkl")
ELO_CACHE_FILE = os.path.join(CACHE_DIR, "elo_v4.pkl")
ELO_HIST_FILE = os.path.join(CACHE_DIR, "elo_hist_v4.pkl")
HORSE_RACES_FILE = os.path.join(CACHE_DIR, "horse_races_v4.pkl")
PEDIGREE_FILE = os.path.join(CACHE_DIR, "pedigree_v4.pkl")           # NEW v4
ML_MODEL_FILE = os.path.join(CACHE_DIR, "ml_ensemble_v4.pkl")        # NEW v4 : ensemble
ML_MODEL_FILE_V5 = os.path.join(CACHE_DIR, "ml_advanced_v5.pkl")     # NEW v5
# NEW v6 — Modèle Top 4 (placement binaire)
ML_MODEL_TOP4_FILE = os.path.join(CACHE_DIR, "ml_top4_ensemble_v6.pkl")
ML_MODEL_TOP4_V5_FILE = os.path.join(CACHE_DIR, "ml_top4_advanced_v6.pkl")
CALIBRATION_FILE = os.path.join(CACHE_DIR, "calibration_v4.pkl")
BETS_FILE = os.path.join(CACHE_DIR, "bets_v4.json")                   # NEW v4
ML_STATUS_FILE = os.path.join(CACHE_DIR, "ml_status.json")            # Auto-train status

WINDOW_SHORT = 30
HISTORY_DAYS = 180


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


# Pour cotes live : pas de cache (refresh)
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
#  Construction massive des stats (avec pedigree v4)
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


def compute_all_stats(max_days=HISTORY_DAYS):
    """Construit tous les caches en parallèle (v4)."""
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
    pedigree_data = []  # liste (cheval, pere, mere, place)

    elo_K = 16

    tasks = []
    today = datetime.now()
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

            # NEW v4 : pedigree
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

    # Compute pedigree stats
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

    save_pickle(STATS_CACHE_FILE, team_out)
    save_pickle(HORSE_STATS_FILE, horse_out)
    save_pickle(ELO_CACHE_FILE, elo_out)
    save_pickle(ELO_HIST_FILE, elo_hist_out)
    save_pickle(HORSE_RACES_FILE, horse_races_out)
    save_pickle(PEDIGREE_FILE, pedigree)

    return team_out, horse_out, elo_out, elo_hist_out, horse_races_out, pedigree


# ============================================================
#  Scoring helpers (v3 inchangés)
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
        db = team_stats["drivers_disc"].get(name, {}).get(discipline) if discipline else None
        hb = team_stats["drivers_hippo"].get(name, {}).get(hippodrome) if hippodrome else None
    else:
        gb = team_stats["entraineurs"].get(name)
        sb = team_stats["entraineurs_short"].get(name)
        db = team_stats["entraineurs_disc"].get(name, {}).get(discipline) if discipline else None
        hb = None
    s_g = get_bucket_score(gb) or 50
    s_s = get_bucket_score(sb, min_courses=3)
    s_d = get_bucket_score(db, min_courses=3)
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
#  ML featurization v4 — 23 features (vs 17 en v3)
# ============================================================
def featurize(p, nb_partants):
    s = p["scores"]
    # v5 : ajout d'interactions pour capturer non-linéarités
    forme = s.get("forme", 0)
    elo = s.get("elo", 50)
    driver = s.get("driver", 50)
    # marché neutralisé → on ne se fie plus aux cotes
    marche = 50

    return [
        marche,
        forme,
        s.get("carriere", 0),
        s.get("gains", 0),
        driver,
        s.get("entraineur", 50),
        s.get("distance", 50),
        s.get("cheval_stats", 50),
        elo,
        s.get("age_sexe", 50),
        s.get("repos", 50),
        s.get("elo_trend", 50),
        s.get("confrontation", 50),
        s.get("pedigree", 50),
        s.get("corde", 50),
        s.get("equipment", 50),
        s.get("profile_match", 50),
        nb_partants,
        1.0 / max(p.get("cote") or 50, 1),
        p["bonus"].get("team", 0),
        p["bonus"].get("deferre", 0),
        p.get("age") or 5,
        1 if p.get("sexe") == "FEMELLES" else 0,
        # v5 interactions
        forme * elo / 100,  # forme × elo
        driver * s.get("entraineur", 50) / 100,  # team synergy
        marche * s.get("cheval_stats", 50) / 100,  # marché vs stats (neutralisé)
        abs(forme - 50),  # écart à la moyenne (forme extrême)
        # NEW v6 — 12 features avancées
        s.get("driver_hippo", 50),       # 🔴 Taux driver/hippodrome
        s.get("regularite", 50),         # 🔴 Indice de régularité
        s.get("equip_change", 50),       # 🟡 Changement d'équipement
        s.get("style_attaquant", 50),    # 🟡 Style attaquant
        s.get("style_finisseur", 50),    # 🟡 Style finisseur
        s.get("style_fragile", 50),      # 🟡 Penal fragile
        s.get("gains_trend", 50),        # 🟡 Tendance des gains
        s.get("jours_derniere", 50),     # 🟢 Jours depuis dernière course
        s.get("nb_courses_mois", 50),    # 🟢 Nb courses ce mois
        s.get("perf_terrain", 50),       # 🟡 Performance terrain
        s.get("corde_avantage", 50),     # 🟢 Avantage corde historique
        s.get("chimie_driver", 50),      # 🟡 Chimie cheval/driver
        # NEW v6 — interactions
        s.get("regularite", 50) * forme / 100,  # régularité × forme = fiabilité
        s.get("driver_hippo", 50) * s.get("perf_terrain", 50) / 100,  # expertise locale × terrain
    ]


FEATURE_NAMES = ["marche","forme","carriere","gains","driver","entraineur",
                 "distance","cheval_stats","elo","age_sexe","repos",
                 "elo_trend","confrontation","pedigree","corde","equipment",
                 "profile_match","nb_partants","inv_cote",
                 "bonus_team","bonus_deferre","age_raw","is_female",
                 "forme_x_elo","team_synergy","marche_x_stats","forme_extreme",
                 # v6
                 "driver_hippo","regularite","equip_change",
                 "style_attaquant","style_finisseur","style_fragile",
                 "gains_trend","jours_derniere","nb_courses_mois",
                 "perf_terrain","corde_avantage","chimie_driver",
                 "regularite_x_forme","driver_hippo_x_terrain"]


def load_ml_model():
    # Priorité au modèle v5 avancé
    if HAS_ADVANCED:
        adv = load_advanced(ML_MODEL_FILE_V5)
        if adv:
            return adv
    payload = load_pickle(ML_MODEL_FILE, max_age_hours=24*14)
    return load_model_from_dict(payload) if payload else None


def save_ml_model(model):
    save_pickle(ML_MODEL_FILE, model.to_dict())


def load_ml_model_top4():
    """Charge le modèle Top 4 (placement binaire)."""
    if HAS_ADVANCED:
        adv = load_advanced(ML_MODEL_TOP4_V5_FILE)
        if adv:
            return adv
    payload = load_pickle(ML_MODEL_TOP4_FILE, max_age_hours=24*14)
    return load_model_from_dict(payload) if payload else None


def save_ml_model_top4(model):
    save_pickle(ML_MODEL_TOP4_FILE, model.to_dict())


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


def train_ml_model(days_back=21, exclude_recent=0, n_trees_gbm=50, n_trees_rf=30,
                   model_type="ensemble"):
    """Entraîne 2 modèles : WIN (y=1 si gagnant) + TOP4 (y=1 si ≤4e).
    Les deux partagent les mêmes features X mais ont des labels différents."""
    try:
        import numpy as np
    except ImportError:
        return None

    X, y_win, y_top4 = [], [], []
    today = datetime.now()
    team_stats, horse_stats, elo, elo_hist, horse_races, pedigree = compute_all_stats(
        max_days=max(HISTORY_DAYS, days_back + exclude_recent))

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
                                             elo, elo_hist, horse_races, pedigree,
                                             terrain=None)
        nb = len(analyses)
        for a in analyses:
            X.append(featurize(a, nb))
            real = next((p for p in parts["participants"]
                        if p.get("numPmu") == a["numPmu"]), None)
            place = (real.get("ordreArrivee") or 0) if real else 0
            # === DUAL LABELS ===
            y_win.append(1 if place == 1 else 0)
            y_top4.append(1 if 1 <= place <= 4 else 0)

    if len(X) < 100:
        return None

    n_top4 = sum(y_top4)
    n_win = sum(y_win)
    print(f"[ML v6] {len(X)} échantillons")
    print(f"  WIN  : {n_win} victoires ({n_win/len(X)*100:.1f}%)")
    print(f"  TOP4 : {n_top4} placés ({n_top4/len(X)*100:.1f}%)")

    info = {"n_samples": len(X), "trained_at": datetime.now().isoformat(),
            "model_type": model_type, "win_positives": n_win, "top4_positives": n_top4}

    # ===========================================================
    #  TRAINING — modèle avancé (Stacking sklearn)
    # ===========================================================
    if model_type == "advanced" and HAS_ADVANCED:
        # --- Modèle WIN ---
        print("[ML v6] 🔵 Entraînement WIN (stacking avancé)...")
        train_advanced(X, y_win, ML_MODEL_FILE_V5)

        # --- Modèle TOP4 ---
        print("[ML v6] 🟢 Entraînement TOP4 (stacking avancé)...")
        train_advanced(X, y_top4, ML_MODEL_TOP4_V5_FILE)

        info["win_model"] = "LGBM+CatBoost+HGB+RF+LR (TimeSeriesSplit)"
        info["top4_model"] = "LGBM+CatBoost+HGB+RF+LR (TimeSeriesSplit)"
        info["models_trained"] = ["win", "top4"]
        return info

    # ===========================================================
    #  TRAINING — modèle ensemble (NumPy pur)
    # ===========================================================
    # --- Modèle WIN ---
    gbm_win = None
    rf_win = None
    if model_type in ("ensemble", "gbm"):
        print(f"[ML v6] 🔵 Entraînement WIN GBM ({n_trees_gbm} arbres)...")
        gbm_win = GradientBoosting(n_trees=n_trees_gbm, max_depth=3, learning_rate=0.1)
        gbm_win.fit(X, y_win)
    if model_type in ("ensemble", "rf"):
        print(f"[ML v6] 🔵 Entraînement WIN RF ({n_trees_rf} arbres)...")
        rf_win = RandomForest(n_trees=n_trees_rf, max_depth=8, min_samples=15)
        rf_win.fit(X, y_win)

    if model_type == "ensemble":
        model_win = Ensemble(gbm=gbm_win, rf=rf_win, w_gbm=0.6, w_rf=0.4)
    elif model_type == "gbm":
        model_win = gbm_win
    else:
        model_win = rf_win

    print("[ML v6] 🔵 Calibration WIN isotone...")
    preds_win = [model_win.predict_one(x) for x in X]
    calib = fit_isotonic(preds_win, y_win, n_bins=20)
    save_calibration(calib)
    save_ml_model(model_win)
    info["n_trees_gbm"] = n_trees_gbm if gbm_win else 0
    info["n_trees_rf"] = n_trees_rf if rf_win else 0

    # --- Modèle TOP4 ---
    gbm_t4 = None
    rf_t4 = None
    if model_type in ("ensemble", "gbm"):
        print(f"[ML v6] 🟢 Entraînement TOP4 GBM ({n_trees_gbm} arbres)...")
        gbm_t4 = GradientBoosting(n_trees=n_trees_gbm, max_depth=3, learning_rate=0.1)
        gbm_t4.fit(X, y_top4)
    if model_type in ("ensemble", "rf"):
        print(f"[ML v6] 🟢 Entraînement TOP4 RF ({n_trees_rf} arbres)...")
        rf_t4 = RandomForest(n_trees=n_trees_rf, max_depth=8, min_samples=15)
        rf_t4.fit(X, y_top4)

    if model_type == "ensemble":
        model_t4 = Ensemble(gbm=gbm_t4, rf=rf_t4, w_gbm=0.6, w_rf=0.4)
    elif model_type == "gbm":
        model_t4 = gbm_t4
    else:
        model_t4 = rf_t4

    print("[ML v6] 🟢 Sauvegarde modèle TOP4...")
    save_ml_model_top4(model_t4)
    info["models_trained"] = ["win", "top4"]

    return info


def predict_ml(features, model, calibration=None):
    p = model.predict_one(features)
    if calibration:
        p = apply_calibration(p, calibration)
    return p


# ============================================================
#  Score Top 4 — logique indépendante du gagnant
#  Probabilité INDIVIDUELLE ABSOLUE (pas de normalisation)
# ============================================================
def compute_top4_rate(perfs_detail):
    """Taux historique de courses terminées dans les 4 premiers.
    Retourne toujours un float (0.0 si pas de données)."""
    if not perfs_detail:
        return 0.0
    top4 = 0
    total = 0
    for course in perfs_detail[:12]:
        for p in course.get("participants", []):
            if p.get("itsHim"):
                place = (p.get("place") or {}).get("place", 0) or 0
                if place > 0:
                    total += 1
                    if place <= 4:
                        top4 += 1
    if total == 0:
        return 0.0
    return (top4 / total) * 100


def compute_avg_place(perfs_detail):
    """Place moyenne sur les 10 dernières courses (arrivée connue).
    Retourne toujours un float (0.0 si pas de données)."""
    if not perfs_detail:
        return 0.0
    places = []
    for course in perfs_detail[:10]:
        for p in course.get("participants", []):
            if p.get("itsHim"):
                place = (p.get("place") or {}).get("place", 0) or 0
                if place > 0:
                    places.append(place)
                break
    if not places:
        return 0.0
    return sum(places) / len(places)


def compute_regularity(perfs_detail):
    """Score de régularité : inverse de la variance des places récentes."""
    if not perfs_detail:
        return 50.0
    places = []
    for course in perfs_detail[:10]:
        for p in course.get("participants", []):
            if p.get("itsHim"):
                place = (p.get("place") or {}).get("place", 0) or 0
                if place > 0:
                    places.append(place)
    if len(places) < 3:
        return 50.0
    mean_p = sum(places) / len(places)
    variance = sum((p - mean_p) ** 2 for p in places) / len(places)
    return max(0, min(100, 100 - variance * 2.5))


def compute_placement_probability(a, nb_partants):
    """
    Probabilité ABSOLUE d'être dans les 4 premiers.
    Pas de normalisation entre chevaux.
    
    Logique simple et directe :
    1. Base = taux top4 historique réel (déjà une probabilité !)
    2. Petits ajustements pour les conditions du jour (±5-10%)
    3. Pénalité forte si manque de données
    """
    top4_rate = a.get("top4", {}).get("rate", 0.0) or 0.0
    nb_courses = a.get("nbCourses", 0) or 0
    proba = None
    
    # === ÉTAPE 1 : Base = taux historique réel ===
    if top4_rate > 0 and nb_courses >= 8:
        # Assez de données → on se fie au taux réel
        proba = top4_rate
    elif top4_rate > 0 and nb_courses >= 3:
        # Quelques données → confiance modérée, on tire vers la moyenne théorique
        theorique = (4 / max(nb_partants, 8)) * 100
        confiance = nb_courses / 8  # 0.375 à 1.0
        proba = top4_rate * confiance + theorique * (1 - confiance)
    else:
        # Pas ou peu de données → moyenne théorique avec forte incertitude
        proba = (4 / max(nb_partants, 8)) * 100
        # Pénalité inconnu
        if nb_courses == 0:
            proba -= 10  # inconnu = risqué
    
    # === ÉTAPE 2 : Ajustement par place moyenne ===
    # Un cheval qui moyenne 2e a +8%, un qui moyenne 9e a -12%
    perfs = a.get("_perfs_detail", [])
    avg_place = compute_avg_place(perfs)
    if avg_place is not None:
        if avg_place <= 2.5:
            proba += 8
        elif avg_place <= 3.5:
            proba += 5
        elif avg_place <= 4.5:
            proba += 2
        elif avg_place <= 6.0:
            proba -= 3
        elif avg_place <= 8.0:
            proba -= 8
        else:
            proba -= 15
    
    # === ÉTAPE 3 : Ajustement par régularité ===
    # Un cheval constant (toujours entre 3e et 5e) est +fiable pour un top4
    regularity = a.get("top4", {}).get("regularity", 50)
    if regularity >= 75:
        proba += 4  # très régulier
    elif regularity >= 60:
        proba += 2  # assez régulier
    elif regularity <= 25:
        proba -= 5  # très irrégulier
    
    # === ÉTAPE 4 : Ajustement par repos ===
    repos_score = a["scores"]["repos"]
    if repos_score >= 80:
        proba += 3  # frais et disposé
    elif repos_score <= 25:
        proba -= 5  # pas frais ou trop reposé
    elif repos_score <= 40:
        proba -= 2  # un peu juste
    
    # === ÉTAPE 5 : Ajustement par driver ===
    driver_score = a["scores"]["driver"]
    if driver_score >= 75:
        proba += 2  # top driver
    elif driver_score <= 30:
        proba -= 3  # driver faible
    
    # === ÉTAPE 6 : Expérience ===
    if nb_courses == 0:
        proba -= 5  # inconnu = risque
    elif nb_courses >= 40:
        proba += 3  # vétéran fiable
    
    # Bornes réalistes
    proba = max(2, min(80, proba))
    
    return round(proba, 1)


# ============================================================
#  ALGORITHME HYBRIDE v4
# ============================================================
def analyser_course_features(participants_data, perfs_data, distance, discipline,
                              hippodrome, type_corde,
                              team_stats, horse_stats, elo,
                              elo_hist=None, horse_races=None, pedigree=None,
                              terrain=None):
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

        try:
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

            # NEW v4 : pedigree
            s_pedigree = get_pedigree_score(pere, mere, pedigree.get("peres", {}), pedigree.get("meres", {}))
            # NEW v4 : corde
            s_corde = get_corde_score(p.get("numPmu"), nb_partants, type_corde, discipline)
            # NEW v4 : equipment
            s_equipment = get_equipment_score(p.get("oeilleres"), p.get("deferre"))
            # NEW v4 : profile match
            profile = detect_profile(perfs_detail)
            s_profile_match = get_profile_match_score(profile, distance, nb_partants)

            # ===========================================================
            #  NEW v6 — Features avancées pour le ML
            # ===========================================================

            # 🔴 Taux driver/hippodrome spécifique
            driver_hippo_data = team_stats.get("drivers_hippo", {}).get(driver, {}).get(hippodrome) if driver and hippodrome else None
            s_driver_hippo = get_bucket_score(driver_hippo_data, min_courses=3) or 50

            # 🟡 Changement d'équipement (signal d'intention)
            s_equip_change = detect_equipment_change(perfs_detail, p.get("oeilleres"), p.get("deferre"))

            # 🟡 Style de course — scores bruts du profil
            s_style_attaquant = profile.get("attaquant", 50)
            s_style_finisseur = profile.get("finisseur", 50)
            s_style_fragile = profile.get("fragile", 50)

            # 🟡 Tendance des gains sur 5 dernières courses
            s_gains_trend = compute_gains_trend(perfs_detail)

            # 🟢 Jours depuis dernière course (score normalisé)
            s_jours_derniere = compute_days_since_last(perfs_detail)

            # 🟢 Nombre de courses ce mois (fatigue cumulative)
            s_nb_courses_mois = compute_nb_courses_recent(perfs_detail)

            # 🟡 Performance terrain (affinité historique)
            s_perf_terrain = compute_terrain_perf(perfs_detail)

            # 🟢 Avantage corde spécifique (historique par position de départ)
            s_corde_avantage = compute_corde_avantage(perfs_detail)

            # 🟡 Chimie cheval/driver actuel (taux de réussite spécifique)
            chimie_data = horse_stats.get("with_driver", {}).get(cheval, {}).get(driver) if cheval and driver else None
            s_chimie_driver = get_bucket_score(chimie_data, min_courses=2) or 50

            # === SCORE TOP 4 (logique indépendante du gagnant) ===
            s_top4_rate = compute_top4_rate(perfs_detail) or 0.0
            s_regularity = compute_regularity(perfs_detail)
            # Un cheval qui finit souvent placé + régulier = bon top 4
            # On pondère : taux top4 (60%) + régularité (40%)
            s_top4_raw = 0.60 * s_top4_rate + 0.40 * s_regularity

        except Exception as e:
            # 🛡️ Un cheval ne doit jamais crasher toute la course
            print(f"[WARN] Erreur scoring {cheval or '?'}: {e}")
            s_forme = 50; s_carriere = 25; s_gains = 25
            s_driver = 50; s_entraineur = 50; s_cheval = 50
            s_elo = 50; s_distance = 50; s_age_sexe = 50; s_repos = 50
            s_elo_trend = 50; s_confrontation = 50
            s_pedigree = 50; s_corde = 50; s_equipment = 50; s_profile_match = 50
            s_driver_hippo = 50; s_regularity = 50.0; s_equip_change = 50.0
            s_style_attaquant = 50; s_style_finisseur = 50; s_style_fragile = 50
            s_gains_trend = 50.0; s_jours_derniere = 50; s_nb_courses_mois = 50
            s_perf_terrain = 50.0; s_corde_avantage = 50; s_chimie_driver = 50
            s_top4_rate = 0.0; s_top4_raw = 20.0
            profile = {"attaquant": 50, "finisseur": 50, "fragile": 50, "regulier": 50}
            gains = p.get("gainsParticipant", {}) or {}
            gains_carriere = gains.get("gainsCarriere", 0) or 0

        bonus_team = 0
        if driver and entr and driver == entr: bonus_team = 3
        if p.get("driverChange"): bonus_team -= 5
        bonus_deferre = 2 if "DEFERRE" in (p.get("deferre", "") or "") else 0

        # Stocker perfs_detail pour le calcul Top 4 (logique indépendante)
        a_entry = {
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
                "forme": round(float(s_forme or 50), 1),
                "carriere": round(float(s_carriere or 25), 1),
                "gains": round(float(s_gains or 25), 1),
                "driver": round(float(s_driver or 50), 1),
                "entraineur": round(float(s_entraineur or 50), 1),
                "distance": round(float(s_distance or 50), 1),
                "cheval_stats": round(float(s_cheval or 50), 1),
                "elo": round(float(s_elo or 50), 1),
                "age_sexe": round(float(s_age_sexe or 50), 1),
                "repos": round(float(s_repos or 50), 1),
                "elo_trend": round(float(s_elo_trend or 50), 1),
                "confrontation": round(float(s_confrontation or 50), 1),
                "pedigree": round(float(s_pedigree or 50), 1),
                "corde": round(float(s_corde or 50), 1),
                "equipment": round(float(s_equipment or 50), 1),
                "profile_match": round(float(s_profile_match or 50), 1),
                # NEW v6 features
                "driver_hippo": round(float(s_driver_hippo or 50), 1),
                "regularite": round(float(s_regularity or 50), 1),
                "equip_change": round(float(s_equip_change or 50), 1),
                "style_attaquant": round(float(s_style_attaquant or 50), 1),
                "style_finisseur": round(float(s_style_finisseur or 50), 1),
                "style_fragile": round(float(s_style_fragile or 50), 1),
                "gains_trend": round(float(s_gains_trend or 50), 1),
                "jours_derniere": round(float(s_jours_derniere or 50), 1),
                "nb_courses_mois": round(float(s_nb_courses_mois or 50), 1),
                "perf_terrain": round(float(s_perf_terrain or 50), 1),
                "corde_avantage": round(float(s_corde_avantage or 50), 1),
                "chimie_driver": round(float(s_chimie_driver or 50), 1),
            },
            "top4": {
                "rate": round(float(s_top4_rate or 0), 1),
                "regularity": round(float(s_regularity or 50), 1),
                "raw": round(float(s_top4_raw or 20), 1),
            },
            "bonus": {"team": bonus_team, "deferre": bonus_deferre},
            "_perfs_detail": perfs_detail,  # pour calcul Top 4
        }
        analyses.append(a_entry)

    return analyses


def analyser_course(participants_data, perfs_data=None, distance=None,
                    discipline=None, hippodrome=None, type_corde=None,
                    team_stats=None, horse_stats=None, elo=None,
                    elo_hist=None, horse_races=None, pedigree=None,
                    use_ml=False, capital=100, terrain=None):
    analyses = analyser_course_features(participants_data, perfs_data, distance,
                                         discipline, hippodrome, type_corde,
                                         team_stats, horse_stats, elo,
                                         elo_hist, horse_races, pedigree,
                                         terrain=terrain)
    if not analyses:
        return []

    # Score intrinsèque v6 (29 composantes) — None-safe
    scores_intr = []
    for a in analyses:
        s = a["scores"]
        # Helper: float-safe avec fallback 50
        def _f(key, default=50.0):
            return float(s.get(key) or default)
        s_val = (
            0.12 * _f("forme") +
            0.06 * _f("carriere", 25) +
            0.05 * _f("gains", 25) +
            0.07 * _f("driver") +
            0.04 * _f("entraineur") +
            0.05 * _f("distance") +
            0.07 * _f("cheval_stats") +
            0.08 * _f("elo") +
            0.03 * _f("age_sexe") +
            0.03 * _f("repos") +
            0.04 * _f("elo_trend") +
            0.02 * _f("confrontation") +
            0.04 * _f("pedigree") +
            0.02 * _f("corde") +
            0.01 * _f("equipment") +
            0.01 * _f("profile_match") +
            0.05 * _f("driver_hippo") +
            0.06 * _f("regularite") +
            0.02 * _f("equip_change") +
            0.02 * _f("style_attaquant") +
            0.02 * _f("style_finisseur") +
            0.02 * _f("gains_trend") +
            0.01 * _f("jours_derniere") +
            0.01 * _f("nb_courses_mois") +
            0.02 * _f("perf_terrain") +
            0.01 * _f("corde_avantage") +
            0.02 * _f("chimie_driver") +
            a["bonus"]["team"] + a["bonus"]["deferre"]
        )
        scores_intr.append(max(s_val, 1))

    total_intr = sum(scores_intr) or 1
    proba_intr = [s / total_intr * 100 for s in scores_intr]

    # Classement basé à 100% sur le score intrinsèque (SANS cotes du marché)
    chances_heur = list(proba_intr)

    # === PROBABILITÉ TOP 4 (logique INDEPENDANTE du gagnant) ===
    # Chaque cheval a sa proba ABSOLUE (pas de normalisation entre chevaux)
    # Critères spécifiques au placement : fiabilité, régularité, expérience
    nb_partants = len(analyses)
    for a in analyses:
        a["chanceTop4"] = compute_placement_probability(a, nb_partants)

    ml_model = load_ml_model() if use_ml else None
    calib = load_calibration() if use_ml else None
    chances_ml = None
    if ml_model:
        nb = len(analyses)
        raw_ml = [predict_ml(featurize(a, nb), ml_model, calib) for a in analyses]
        total_ml = sum(raw_ml) or 1
        chances_ml = [x / total_ml * 100 for x in raw_ml]

    # NEW v6 — Top 4 ML model (modèle binaire placement)
    ml_top4_model = load_ml_model_top4() if use_ml else None
    raw_top4_ml = None
    if ml_top4_model:
        nb = len(analyses)
        raw_top4_ml = [predict_ml(featurize(a, nb), ml_top4_model) for a in analyses]

    for i, a in enumerate(analyses):
        if chances_ml:
            # Avec ML : 20% heuristique + 80% ML
            a["chance"] = round(0.2 * chances_heur[i] + 0.8 * chances_ml[i], 2)
            a["chanceML"] = round(chances_ml[i], 2)
        else:
            a["chance"] = round(chances_heur[i], 2)
        a["chanceHeur"] = round(chances_heur[i], 2)

        # NEW v6 — Top 4 : ML remplace heuristique si disponible
        if raw_top4_ml:
            ml_top4_prob = raw_top4_ml[i] * 100
            # Sauvegarder l'heuristique avant blend
            a["chanceTop4Heur"] = round(a["chanceTop4"], 2)
            a["chanceTop4ML"] = round(ml_top4_prob, 2)
            # Blend : 20% heuristique + 80% ML Top4
            a["chanceTop4"] = round(0.2 * a["chanceTop4Heur"] + 0.8 * ml_top4_prob, 2)

        if a["cote"] and a["probaMarche"] > 0:
            edge = a["chance"] - a["probaMarche"]
            a["edge"] = round(edge, 2)
            a["valueBet"] = edge > 4 and a["cote"] >= 4
            # NEW v4 : Kelly + EV
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

    # Supprimer les données internes non JSON-sérialisables
    for a in analyses:
        a.pop("_perfs_detail", None)

    return analyses


# ============================================================
#  Backtest v4
# ============================================================
def backtest(days_back=7, use_ml=False):
    team_stats, horse_stats, elo, elo_hist, horse_races, pedigree = compute_all_stats(
        max_days=HISTORY_DAYS)
    today = datetime.now()
    results = {
        "total_courses": 0, "top1_winner": 0, "top1_top3": 0, "top3_winner": 0,
        "value_bets": [], "mise_totale": 0.0, "gain_total": 0.0,
        # NEW v4 : Kelly tracking
        "kelly_mise_totale": 0.0, "kelly_gain_total": 0.0,
        # NEW v6 : Top 4 tracking
        "top4_ml_hit": 0, "top4_ml_total": 0,
        "top1_top4_hit": 0,  # le #1 algo est-il dans le top 4 ?
        "top4_by_confidence": {"high": {"hit": 0, "total": 0}, "medium": {"hit": 0, "total": 0}, "low": {"hit": 0, "total": 0}},
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

        # NEW v6 — Top 4 tracking
        top1_place = top1.get("ordreArrivee", 0) or 0
        if 1 <= top1_place <= 4:
            results["top1_top4_hit"] += 1

        # Top 4 ML confidence tracking : chevaux avec chanceTop4ML > 60%
        for a in analyses:
            ml_top4_prob = a.get("chanceTop4ML", 0)
            actual_place = a.get("ordreArrivee", 0) or 0
            if ml_top4_prob > 0:
                actual_top4 = 1 <= actual_place <= 4
                results["top4_ml_total"] += 1
                if actual_top4:
                    results["top4_ml_hit"] += 1
                # Par niveau de confiance
                if ml_top4_prob >= 0.60:
                    bucket = "high"
                elif ml_top4_prob >= 0.35:
                    bucket = "medium"
                else:
                    bucket = "low"
                results["top4_by_confidence"][bucket]["total"] += 1
                if actual_top4:
                    results["top4_by_confidence"][bucket]["hit"] += 1

        for a in analyses:
            if a.get("valueBet"):
                results["value_bets"].append({
                    "course": meta["course"], "date": meta["date"],
                    "cheval": a["nom"], "cote": a["cote"], "edge": a["edge"],
                    "gagne": a["ordreArrivee"] == 1,
                    "kellyMise": a.get("kellyMise", 0),
                })
                # Kelly tracking
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

    # Kelly stats
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

    # NEW v6 — Top 4 ML stats
    n_top4 = results["top4_ml_total"]
    if n_top4 > 0:
        results["top4_ml_accuracy"] = round(results["top4_ml_hit"] / n_top4 * 100, 2)
        results["top4_ml_brier"] = "N/A"  # TODO: compute Brier score
    else:
        results["top4_ml_accuracy"] = None
    n = results["total_courses"] or 1
    results["top1_top4_rate"] = round(results["top1_top4_hit"] / n * 100, 2)

    # Par bucket de confiance
    for bucket in ["high", "medium", "low"]:
        b = results["top4_by_confidence"][bucket]
        b["accuracy"] = round(b["hit"] / b["total"] * 100, 2) if b["total"] > 0 else None

    results["value_bets"] = results["value_bets"][-30:]
    return results


# ============================================================
#  Bilan quotidien
# ============================================================
def bilan(days_back=7, use_ml=False):
    """Stats par jour : où finit le #1 de l'algo (1er, 2e, 3e, hors podium)."""
    team_stats, horse_stats, elo, elo_hist, horse_races, pedigree = compute_all_stats(
        max_days=HISTORY_DAYS)
    today = datetime.now()
    daily_results = []

    for delta in range(1, days_back + 1):
        d = today - timedelta(days=delta)
        date_str = fmt_date(d)
        day = {
            "date": d.strftime("%d/%m/%Y"),
            "date_short": d.strftime("%d/%m"),
            "jour": ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"][d.weekday()],
            "total": 0, "top1": 0, "top2": 0, "top3": 0, "hors": 0,
            "top3_total": 0,
            "courses": [],
        }

        try:
            prog = get_programme(date_str)
        except Exception:
            continue

        tasks = []
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
            analyses = analyser_course(parts, perfs, distance, discipline,
                                        hippodrome, type_corde,
                                        team_stats, horse_stats, elo, elo_hist,
                                        horse_races, pedigree, use_ml=use_ml,
                                        capital=100)
            if not analyses:
                continue

            day["total"] += 1
            top1 = analyses[0]
            place = top1.get("ordreArrivee", 0) or 0

            course_detail = {
                "nom": top1["nom"],
                "place": place,
                "cote": top1.get("cote"),
            }

            if place == 1:
                day["top1"] += 1
                course_detail["resultat"] = "🥇"
            elif place == 2:
                day["top2"] += 1
                course_detail["resultat"] = "🥈"
            elif place == 3:
                day["top3"] += 1
                course_detail["resultat"] = "🥉"
            elif place > 0:
                day["hors"] += 1
                course_detail["resultat"] = f"#{place}"
            else:
                day["hors"] += 1
                course_detail["resultat"] = "—"

            if 1 <= place <= 3:
                day["top3_total"] += 1

            day["courses"].append(course_detail)

        if day["total"] > 0:
            day["taux_top1"] = round(day["top1"] / day["total"] * 100, 1)
            day["taux_top3"] = round(day["top3_total"] / day["total"] * 100, 1)
            daily_results.append(day)

    return daily_results


# ============================================================
#  Crack horses (public API)
# ============================================================
def get_crack_horses(date_str):
    """Retourne les chevaux avec le badge ⚡ Crack (score ELO normalisé ≥ 85)
    en excluant ceux qui ont 0 courses. Même logique que l'analyse admin."""
    team_stats, horse_stats, elo, elo_hist, horse_races, pedigree = compute_all_stats()

    try:
        programme = get_programme(date_str)
    except Exception:
        return []

    cracks = []
    CRACK_ELO_SCORE_MIN = 85  # Même seuil que le badge ⚡ Crack dans index.html

    for r in programme["programme"]["reunions"]:
        hippo = r["hippodrome"]["libelleCourt"]
        for c in r["courses"]:
            r_num = r["numOfficiel"]
            c_num = c["numOrdre"]
            distance = c.get("distance")
            discipline = c.get("discipline", "")
            type_corde = c.get("corde", "")
            try:
                parts = get_participants(date_str, r_num, c_num)
                perfs = get_performances(date_str, r_num, c_num)
            except Exception:
                continue

            # Utiliser le moteur complet avec ML
            analyses = analyser_course(
                parts, perfs, distance, discipline, hippo, type_corde,
                team_stats, horse_stats, elo, elo_hist, horse_races, pedigree,
                use_ml=True, capital=100)

            partants = [p for p in parts.get("participants", [])
                        if p.get("statut") == "PARTANT"]
            nb_partants = len(partants)

            for a in analyses:
                nb_courses = a.get("nbCourses", 0) or 0
                elo_score = a["scores"]["elo"]

                # Exclure les chevaux avec 0 courses
                if nb_courses == 0:
                    continue

                # Même condition que le badge ⚡ Crack dans l'admin
                if elo_score < CRACK_ELO_SCORE_MIN:
                    continue

                rap = None
                for p in partants:
                    if p.get("numPmu") == a["numPmu"]:
                        rap = p.get("dernierRapportDirect") or p.get("dernierRapportReference")
                        break
                cote = float(rap["rapport"]) if rap and rap.get("rapport") else None

                cracks.append({
                    "numPmu": a["numPmu"],
                    "cheval": a["nom"],
                    "elo_score": round(elo_score, 1),
                    "driver": a["driver"],
                    "entraineur": a["entraineur"],
                    "age": a.get("age"),
                    "sexe": a.get("sexe"),
                    "cote": cote,
                    "hippodrome": hippo,
                    "num_reunion": r_num,
                    "course": f"R{r_num}C{c_num}",
                    "num_course": c_num,
                    "heure": datetime.fromtimestamp(
                        c["heureDepart"] / 1000
                    ).strftime("%H:%M") if c.get("heureDepart") else "",
                    "discipline": discipline,
                    "distance": distance,
                    "nb_partants": nb_partants,
                    "nb_courses": nb_courses,
                    "nb_victoires": a.get("nbVictoires", 0),
                    "nb_places": a.get("nbPlaces", 0),
                    "gains_carriere": a.get("gainsCarriere", 0),
                })

    # Trier : par num_reunion, puis num_course, puis score ELO décroissant
    cracks.sort(key=lambda x: (x["num_reunion"], x["num_course"], -x["elo_score"]))
    return cracks


# ============================================================
#  Super Base (public API) — #1 de chaque course
# ============================================================
def get_super_base(date_str):
    """Retourne le #1 (meilleure chance heuristique) de chaque course,
    en excluant les chevaux avec 0 courses."""
    team_stats, horse_stats, elo, elo_hist, horse_races, pedigree = compute_all_stats()

    try:
        programme = get_programme(date_str)
    except Exception:
        return []

    bases = []

    for r in programme["programme"]["reunions"]:
        hippo = r["hippodrome"]["libelleCourt"]
        for c in r["courses"]:
            r_num = r["numOfficiel"]
            c_num = c["numOrdre"]
            distance = c.get("distance")
            discipline = c.get("discipline", "")
            type_corde = c.get("corde", "")
            try:
                parts = get_participants(date_str, r_num, c_num)
                perfs = get_performances(date_str, r_num, c_num)
            except Exception:
                continue

            # Analyse complète avec ML pour avoir le classement
            analyses = analyser_course(
                parts, perfs, distance, discipline, hippo, type_corde,
                team_stats, horse_stats, elo, elo_hist, horse_races, pedigree,
                use_ml=True, capital=100)

            if not analyses:
                continue

            top1 = analyses[0]
            nb_courses = top1.get("nbCourses", 0) or 0

            # Exclure si 0 courses
            if nb_courses == 0:
                continue

            partants = [p for p in parts.get("participants", [])
                        if p.get("statut") == "PARTANT"]

            rap = None
            for p in partants:
                if p.get("numPmu") == top1["numPmu"]:
                    rap = p.get("dernierRapportDirect") or p.get("dernierRapportReference")
                    break
            cote = float(rap["rapport"]) if rap and rap.get("rapport") else None

            bases.append({
                "numPmu": top1["numPmu"],
                "cheval": top1["nom"],
                "chance": top1.get("chance", 0),
                "elo_score": top1["scores"]["elo"],
                "forme": top1["scores"]["forme"],
                "driver": top1["driver"],
                "entraineur": top1["entraineur"],
                "age": top1.get("age"),
                "sexe": top1.get("sexe"),
                "cote": cote,
                "musique": top1.get("musique", ""),
                "hippodrome": hippo,
                "num_reunion": r_num,
                "course": f"R{r_num}C{c_num}",
                "num_course": c_num,
                "heure": datetime.fromtimestamp(
                    c["heureDepart"] / 1000
                ).strftime("%H:%M") if c.get("heureDepart") else "",
                "discipline": discipline,
                "distance": distance,
                "nb_partants": len(partants),
                "nb_courses": nb_courses,
                "nb_victoires": top1.get("nbVictoires", 0),
                "nb_places": top1.get("nbPlaces", 0),
                "gains_carriere": top1.get("gainsCarriere", 0),
                "edge": top1.get("edge", 0),
                "value_bet": top1.get("valueBet", False),
            })

    # Trier par num_reunion puis num_course
    bases.sort(key=lambda x: (x["num_reunion"], x["num_course"]))
    return bases


# ============================================================
#  Jeux du Jour (public API)
# ============================================================
def _selection_entry(a, is_crack=False):
    return {
        "numPmu": a["numPmu"],
        "cheval": a["nom"],
        "chance": a.get("chance", 0),
        "rang": a.get("rang", 0),
        "elo_score": a["scores"]["elo"],
        "forme": a["scores"]["forme"],
        "driver": a["driver"],
        "entraineur": a["entraineur"],
        "age": a.get("age"),
        "sexe": a.get("sexe"),
        "cote": a.get("cote"),
        "musique": a.get("musique", ""),
        "nb_courses": a.get("nbCourses", 0) or 0,
        "nb_victoires": a.get("nbVictoires", 0),
        "nb_places": a.get("nbPlaces", 0),
        "gains_carriere": a.get("gainsCarriere", 0),
        "edge": a.get("edge", 0),
        "value_bet": a.get("valueBet", False),
        "is_crack": is_crack,
    }


def get_selection_course(date_str, r_num, c_num):
    """Retourne une sélection de 7 chevaux pour une course donnée :
    d'abord les cracks (ELO ≥ 85), puis le classement algo."""
    team_stats, horse_stats, elo, elo_hist, horse_races, pedigree = compute_all_stats()

    try:
        prog = get_programme(date_str)
    except Exception:
        return None

    # Trouver les infos de la course
    hippodrome = None
    discipline = None
    distance = None
    type_corde = None
    heure = ""
    libelle = ""

    for r in prog["programme"]["reunions"]:
        if r["numOfficiel"] == r_num:
            hippodrome = r["hippodrome"]["libelleCourt"]
            for c in r["courses"]:
                if c["numOrdre"] == c_num:
                    discipline = c.get("discipline", "")
                    distance = c.get("distance")
                    type_corde = c.get("corde", "")
                    heure = (datetime.fromtimestamp(c["heureDepart"] / 1000)
                             .strftime("%H:%M") if c.get("heureDepart") else "")
                    libelle = c.get("libelle") or c.get("libelleCourt") or ""
                    break

    if hippodrome is None:
        return None

    try:
        parts = get_participants(date_str, r_num, c_num)
        perfs = get_performances(date_str, r_num, c_num)
    except Exception:
        return None

    analyses = analyser_course(
        parts, perfs, distance, discipline, hippodrome, type_corde,
        team_stats, horse_stats, elo, elo_hist, horse_races, pedigree,
        use_ml=True, capital=100)

    if not analyses:
        return None

    nb_partants = len([p for p in parts.get("participants", [])
                        if p.get("statut") == "PARTANT"])

    # Prendre les 7 premiers du classement
    SELECTION_SIZE = 7

    selection = []
    for a in analyses:
        if (a.get("nbCourses", 0) or 0) > 0:
            selection.append(_selection_entry(a, is_crack=a["scores"]["elo"] >= 85))
            if len(selection) >= SELECTION_SIZE:
                break

    return {
        "course": {
            "r_num": r_num, "c_num": c_num,
            "hippodrome": hippodrome,
            "discipline": discipline,
            "distance": distance,
            "heure": heure,
            "libelle": libelle,
            "nb_partants": nb_partants,
        },
        "selection": selection,
    }


# ============================================================
#  ROUTES PUBLIQUES (sans auth)
# ============================================================
@app.route("/")
def public_home():
    return render_template("public.html")


@app.route("/api/crack-horses")
def api_crack_horses():
    date_str = request.args.get("date") or fmt_date(datetime.now())
    try:
        cracks = get_crack_horses(date_str)
        return jsonify({"date": date_str, "cracks": cracks})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/programme-public")
def api_programme_public():
    """Liste des courses du jour (public, sans auth)."""
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
                "libelle": c.get("libelle") or c.get("libelleCourt") or "",
                "discipline": c.get("discipline"),
                "distance": c.get("distance"),
                "heure": datetime.fromtimestamp(
                    c["heureDepart"] / 1000
                ).strftime("%H:%M") if c.get("heureDepart") else "",
                "nbPartants": c.get("nombreDeclaresPartants"),
            } for c in r["courses"]],
        })
    return jsonify({"date": date_str, "reunions": out})


@app.route("/api/super-base")
def api_super_base():
    date_str = request.args.get("date") or fmt_date(datetime.now())
    try:
        bases = get_super_base(date_str)
        return jsonify({"date": date_str, "bases": bases})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/course-du-jour")
def api_course_du_jour():
    date_str = request.args.get("date") or fmt_date(datetime.now())
    r_num = int(request.args.get("r", 0))
    c_num = int(request.args.get("c", 0))
    if not r_num or not c_num:
        return jsonify({"error": "Paramètres r et c requis"}), 400
    try:
        result = get_selection_course(date_str, r_num, c_num)
        if result is None:
            return jsonify({"error": "Course introuvable"}), 404
        return jsonify({"date": date_str, "course": result["course"],
                        "selection": result["selection"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ============================================================
#  AUTH ADMIN
# ============================================================
@app.route("/login", methods=["GET", "POST"])
def login_page():
    error = None
    if request.method == "POST":
        password = request.form.get("password", "")
        if password == ADMIN_PASSWORD:
            session["admin"] = True
            return redirect(url_for("admin_home"))
        error = "Mot de passe incorrect"
    return render_template("admin_login.html", error=error)


@app.route("/logout")
def logout():
    session.pop("admin", None)
    return redirect(url_for("public_home"))


# ============================================================
#  ROUTES ADMIN (protégées)
# ============================================================
@app.route("/admin")
@admin_required
def admin_home():
    return render_template("index.html")


@app.route("/backtest")
@admin_required
def backtest_page():
    return render_template("backtest.html")


@app.route("/paris")
@admin_required
def paris_page():
    return render_template("paris.html")


@app.route("/bilan")
@admin_required
def bilan_page():
    return render_template("bilan.html")


@app.route("/api/reunions")
@admin_required
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
@admin_required
def api_course(r_num, c_num):
    date_str = request.args.get("date") or fmt_date(datetime.now())
    use_ml = request.args.get("ml") == "1"
    live = request.args.get("live") == "1"  # NEW v4
    capital = float(request.args.get("capital", 100))
    try:
        prog = get_programme(date_str)
        # Si live, on bypass le cache LRU pour avoir les cotes à jour
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
                    terrain = c.get("natureTerrain") or c.get("terrain") or None
                    course_info = {
                        "libelle": c.get("libelle"),
                        "discipline": discipline,
                        "distance": c.get("distance"),
                        "specialite": c.get("specialite"),
                        "corde": type_corde,
                        "terrain": terrain,
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
                                capital=capital,
                                terrain=terrain)

    return jsonify({
        "date": date_str, "reunion": reunion_info, "course": course_info,
        "analyses": analyses,
        "ml_active": use_ml and load_ml_model() is not None,
        "ml_top4_active": use_ml and load_ml_model_top4() is not None,
        "live": live,
        "timestamp": datetime.now().isoformat(),
    })


@app.route("/api/bilan")
@admin_required
def api_bilan():
    days = int(request.args.get("days", 7))
    use_ml = request.args.get("ml") == "1"
    days = min(days, 30)
    try:
        data = bilan(days_back=days, use_ml=use_ml)
        # Totaux globaux
        totals = {"total": 0, "top1": 0, "top2": 0, "top3": 0, "hors": 0,
                  "top3_total": 0}
        for d in data:
            for k in totals:
                totals[k] += d.get(k, 0)
        n = totals["total"] or 1
        totals["taux_top1"] = round(totals["top1"] / n * 100, 1)
        totals["taux_top3"] = round(totals["top3_total"] / n * 100, 1)
        return jsonify({"daily": data, "totals": totals})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/backtest")
@admin_required
def api_backtest():
    days = int(request.args.get("days", 7))
    use_ml = request.args.get("ml") == "1"
    days = min(days, 30)
    try:
        return jsonify(backtest(days_back=days, use_ml=use_ml))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/train", methods=["POST"])
@admin_required
def api_train():
    days = int(request.args.get("days", 21))
    days = min(days, 30)
    n_trees_gbm = int(request.args.get("trees_gbm", 50))
    n_trees_rf = int(request.args.get("trees_rf", 30))
    model_type = request.args.get("type", "advanced")  # v5 par défaut
    try:
        info = train_ml_model(days_back=days, n_trees_gbm=n_trees_gbm,
                              n_trees_rf=n_trees_rf, model_type=model_type)
        if info is None:
            return jsonify({"error": "Pas assez de données"}), 400
        return jsonify({"ok": True, **info})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/team-stats")
@admin_required
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
#  NEW v4 - Tracking des paris
# ============================================================
@app.route("/api/bets", methods=["GET"])
@admin_required
def api_bets_list():
    bets = bets_tracker.load_bets(BETS_FILE)
    stats = bets_tracker.compute_stats(bets)
    # Trier par date desc
    bets.sort(key=lambda b: b.get("created_at", ""), reverse=True)
    return jsonify({"bets": bets, "stats": stats})


@app.route("/api/bets", methods=["POST"])
@admin_required
def api_bets_add():
    data = request.get_json() or {}
    required = ["cheval", "cote", "mise"]
    if not all(k in data for k in required):
        return jsonify({"error": "Missing fields"}), 400
    bet = {
        "date": data.get("date", datetime.now().strftime("%d/%m/%Y")),
        "course": data.get("course", ""),
        "cheval": data["cheval"],
        "cote": float(data["cote"]),
        "mise": float(data["mise"]),
        "type": data.get("type", "simple_gagnant"),
        "edge": float(data.get("edge", 0)),
    }
    return jsonify(bets_tracker.add_bet(BETS_FILE, bet))


@app.route("/api/bets/<int:bet_id>", methods=["PUT"])
@admin_required
def api_bets_update(bet_id):
    data = request.get_json() or {}
    gagne = bool(data.get("gagne"))
    place = data.get("place")
    bets_tracker.update_bet_result(BETS_FILE, bet_id, gagne, place)
    return jsonify({"ok": True})


@app.route("/api/bets/<int:bet_id>", methods=["DELETE"])
@admin_required
def api_bets_delete(bet_id):
    bets_tracker.delete_bet(BETS_FILE, bet_id)
    return jsonify({"ok": True})


# ============================================================
#  Auto-train ML + Status (public)
# ============================================================
import threading
import json as _json

ML_AUTO_TRAIN_HOUR = 5
ML_AUTO_TRAIN_MIN = 0
ML_AUTO_PARAMS = {
    "model_type": "advanced",
    "days_back": 30,
    "n_trees_gbm": 100,
    "n_trees_rf": 50,
}


def _save_ml_status(status_dict):
    try:
        with open(ML_STATUS_FILE, "w") as f:
            _json.dump(status_dict, f, ensure_ascii=False)
    except Exception:
        pass


def _load_ml_status():
    try:
        if os.path.exists(ML_STATUS_FILE):
            with open(ML_STATUS_FILE, "r") as f:
                return _json.load(f)
    except Exception:
        pass
    return None


def _run_auto_train():
    """Exécute l'entraînement ML en arrière-plan."""
    now_str = datetime.now().strftime("%d/%m/%Y %H:%M")
    print(f"[Auto-Train] Démarrage — {now_str}")
    _save_ml_status({
        "status": "training",
        "started_at": datetime.now().isoformat(),
        "params": ML_AUTO_PARAMS,
    })
    try:
        info = train_ml_model(
            days_back=ML_AUTO_PARAMS["days_back"],
            n_trees_gbm=ML_AUTO_PARAMS["n_trees_gbm"],
            n_trees_rf=ML_AUTO_PARAMS["n_trees_rf"],
            model_type=ML_AUTO_PARAMS["model_type"],
        )
        if info:
            _save_ml_status({
                "status": "ok",
                "finished_at": datetime.now().isoformat(),
                "params": ML_AUTO_PARAMS,
                "result": {k: str(v) for k, v in info.items()},
            })
            print(f"[Auto-Train] ✅ Terminé — {info}")
        else:
            _save_ml_status({
                "status": "error",
                "finished_at": datetime.now().isoformat(),
                "error": "Pas assez de données",
            })
            print("[Auto-Train] ❌ Pas assez de données")
    except Exception as e:
        _save_ml_status({
            "status": "error",
            "finished_at": datetime.now().isoformat(),
            "error": str(e),
        })
        print(f"[Auto-Train] ❌ Erreur : {e}")


def _scheduler_loop():
    """Boucle en arrière-plan qui déclenche l'entraînement à 5h00."""
    trained_today = None
    while True:
        now = datetime.now()
        today_key = now.strftime("%Y-%m-%d")
        if (now.hour == ML_AUTO_TRAIN_HOUR and now.minute == ML_AUTO_TRAIN_MIN
                and trained_today != today_key):
            trained_today = today_key
            _run_auto_train()
        # Réinitialiser le flag à minuit
        if now.hour == 0 and now.minute == 0:
            trained_today = None
        import time
        time.sleep(30)


def start_auto_train_scheduler():
    """Lance le scheduler en arrière-plan."""
    t = threading.Thread(target=_scheduler_loop, daemon=True)
    t.start()
    print(f"[Auto-Train] Planificateur actif — entraînement quotidien à {ML_AUTO_TRAIN_HOUR:02d}:{ML_AUTO_TRAIN_MIN:02d}")


@app.route("/api/ml-status")
def api_ml_status():
    """Statut public de la dernière mise à jour ML."""
    status = _load_ml_status()
    if status is None:
        # Vérifier si un modèle existe déjà (entraîné manuellement)
        ml = load_ml_model()
        ml_top4 = load_ml_model_top4()
        if ml:
            return jsonify({"status": "ok", "source": "manual",
                            "models": {"win": True, "top4": ml_top4 is not None},
                            "message": "Modèle(s) chargé(s)"})
        return jsonify({"status": "none", "source": "none",
                        "message": "Aucun modèle entraîné"})
    # Enrichir le status avec la présence des modèles
    status["models_loaded"] = {
        "win": load_ml_model() is not None,
        "top4": load_ml_model_top4() is not None,
    }
    return jsonify(status)


if __name__ == "__main__":
    start_auto_train_scheduler()
    app.run(host="0.0.0.0", port=5000, debug=False)
