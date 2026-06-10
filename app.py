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

# NEW v7 - pipeline XGBoost + LightGBM + TabNet
try:
    from lib.ml_v7 import train_v7, load_v7
    HAS_V7 = True
except:
    HAS_V7 = False

# NEW v8 - pipeline V8 (Optuna + Purged TS-CV + Feature Eng + Poids dynamiques)
try:
    from lib.ml_v8 import train_v8, load_v8, engineer_interactions
    from lib.ml_v8 import train_ranker_v8, load_ranker_v8
    HAS_V8 = True
except:
    HAS_V8 = False

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "turf-analyzer-secret-2026")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")


# ── Helper : nettoyer les NaN pour la sérialisation JSON ──
# Python float('nan') → JSON "NaN" (invalide) → on convertit en None → JSON "null"
def _clean_nan(obj):
    """Remplace récursivement tous les NaN/Inf par None (null en JSON)."""
    import math
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _clean_nan(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_nan(v) for v in obj]
    return obj


# Surcharger jsonify pour qu'il nettoie les NaN automatiquement
_original_jsonify = jsonify
def jsonify(*args, **kwargs):
    """jsonify qui remplace NaN par null."""
    # On nettoie les données avant de les passer à la vraie jsonify
    if args and isinstance(args[0], dict):
        args = (_clean_nan(args[0]),) + args[1:]
    if kwargs:
        kwargs = {k: _clean_nan(v) for k, v in kwargs.items()}
    return _original_jsonify(*args, **kwargs)


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
# NEW v6.1 — Modèle Top 3 (placement binaire, ~25% positifs = idéal pour sklearn)
ML_MODEL_TOP3_FILE = os.path.join(CACHE_DIR, "ml_top3_ensemble_v6.pkl")
ML_MODEL_TOP3_V5_FILE = os.path.join(CACHE_DIR, "ml_top3_advanced_v6.pkl")
# NEW v7 — XGBoost + LightGBM + TabNet
ML_MODEL_WIN_V7_FILE = os.path.join(CACHE_DIR, "ml_win_v7.pkl")
ML_MODEL_TOP3_V7_FILE = os.path.join(CACHE_DIR, "ml_top3_v7.pkl")
ML_MODEL_TOP4_V7_FILE = os.path.join(CACHE_DIR, "ml_top4_v7.pkl")
# NEW v8 — Optuna + Purged TS-CV + Feature Eng + TabNet V8
ML_MODEL_WIN_V8_FILE = os.path.join(CACHE_DIR, "ml_win_v8.pkl")
ML_MODEL_TOP3_V8_FILE = os.path.join(CACHE_DIR, "ml_top3_v8.pkl")
ML_MODEL_TOP4_V8_FILE = os.path.join(CACHE_DIR, "ml_top4_v8.pkl")
# NEW v8.1 — Modèles par discipline
ML_MODEL_WIN_V8_TROT_FILE = os.path.join(CACHE_DIR, "ml_win_v8_trot.pkl")
ML_MODEL_TOP3_V8_TROT_FILE = os.path.join(CACHE_DIR, "ml_top3_v8_trot.pkl")
ML_MODEL_WIN_V8_GALOP_FILE = os.path.join(CACHE_DIR, "ml_win_v8_galop.pkl")
ML_MODEL_TOP3_V8_GALOP_FILE = os.path.join(CACHE_DIR, "ml_top3_v8_galop.pkl")
# NEW v8.1 — Ranker séquentiel
ML_MODEL_RANKER_V8_FILE = os.path.join(CACHE_DIR, "ml_ranker_v8.pkl")
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


def compute_difficulty(cotes):
    """Calcule un indice de difficulté 1-10 basé sur la concentration des cotes.
    
    1 = course facile (1-2 favoris écrasants, cotes 2-4)
    10 = course impossible (tous outsiders, cotes similaires 8-15)
    
    Formule : on regarde la part des 3 plus petites cotes (les favoris).
    - Si top3 cotes concentrent 70%+ de la proba → facile (1-3)
    - Si top3 cotes ne concentrent que 30% → difficile (8-10)
    """
    if not cotes or len(cotes) < 3:
        return None
    # Filtrer les cotes valides (> 1.0)
    cotes = [c for c in cotes if c and c > 1.0]
    if len(cotes) < 3:
        return None
    # Probabilités implicites : 1/cote
    probs = [1.0 / c for c in cotes]
    total = sum(probs)
    if total == 0:
        return None
    # Normaliser
    probs_norm = [p / total for p in probs]
    # Prendre les 3 plus grandes (les favoris)
    probs_sorted = sorted(probs_norm, reverse=True)
    top3_share = sum(probs_sorted[:3])
    # Mapper : top3_share élevé → facile (1), top3_share bas → difficile (10)
    # top3_share typique : 0.25 (très ouvert) à 0.80 (très concentré)
    # Échelle : 0.80 → 1, 0.25 → 10
    difficulty = max(1, min(10, round(10 - (top3_share - 0.20) * 17.5)))
    return difficulty


@app.route("/api/course-difficulty")
@admin_required
def api_course_difficulty():
    """Calcule l'indice de difficulté pour toutes les courses d'un jour.
    Utilise uniquement les cotes → rapide (pas d'analyse complète).
    """
    date_str = request.args.get("date") or fmt_date(datetime.now())
    try:
        prog = get_programme(date_str)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    tasks = []
    keys = []
    for r in prog["programme"]["reunions"]:
        r_num = r["numOfficiel"]
        for c in r["courses"]:
            c_num = c["numOrdre"]
            tasks.append((date_str, r_num, c_num))
            keys.append(f"{r_num}_{c_num}")

    # Fetcher les participants en parallèle (juste pour les cotes)
    def _fetch_cotes(task):
        try:
            parts = get_participants(*task)
            cotes = []
            for p in parts.get("participants", []):
                cote = p.get("dernierRapportDirect") or p.get("cote")
                if cote and float(cote) > 1.0:
                    cotes.append(float(cote))
            return cotes
        except:
            return []

    with ThreadPoolExecutor(max_workers=20) as ex:
        results = list(ex.map(_fetch_cotes, tasks))

    out = {}
    for key, cotes in zip(keys, results):
        d = compute_difficulty(cotes)
        if d is not None:
            out[key] = d

    return jsonify({"date": date_str, "difficulties": out})


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
#  Raw data extraction — données brutes pour le ML
#  Pas de scoring synthétique, le ML apprend les relations
# ============================================================

def _extract_last_places(perfs_detail, n=5):
    """Extrait les N dernières places brutes. NaN = pas de donnée."""
    places = []
    for course in (perfs_detail or [])[:n]:
        found = False
        for p in course.get("participants", []):
            if p.get("itsHim"):
                place = (p.get("place") or {}).get("place", 0) or 0
                places.append(place if place > 0 else float('nan'))
                found = True
                break
        if not found:
            places.append(float('nan'))
    while len(places) < n:
        places.append(float('nan'))
    return places


def _extract_distance_raw(perfs_detail, distance_course):
    """Stats brutes sur courses à distance similaire (±200m). NaN si pas de données."""
    if not perfs_detail or not distance_course:
        return 0, float('nan'), float('nan')  # count, avg_place, win_rate
    places = []
    wins = 0
    for course in perfs_detail:
        dist = course.get("distance")
        if not dist: continue
        if abs(dist - distance_course) <= 200:
            for p in course.get("participants", []):
                if p.get("itsHim"):
                    place = (p.get("place") or {}).get("place", 0) or 0
                    if place > 0:
                        places.append(place)
                        if place == 1:
                            wins += 1
                    break
    if not places:
        return 0, float('nan'), float('nan')
    return len(places), sum(places) / len(places), wins / len(places)


def _extract_team_raw(name, kind, team_stats, discipline=None, hippodrome=None):
    """Extrait les stats brutes d'un driver/entraineur.
    Retourne NaN pour les taux quand pas de données (le ML les ignore).
    """
    _nan = float('nan')
    if not team_stats or not name:
        return 0, _nan, _nan, 0, _nan, 0, _nan

    if kind == "drivers":
        gb = team_stats.get("drivers", {}).get(name)
        db = team_stats.get("drivers_disc", {}).get(name, {}).get(discipline) if discipline else None
        hb = team_stats.get("drivers_hippo", {}).get(name, {}).get(hippodrome) if hippodrome else None
    else:
        gb = team_stats.get("entraineurs", {}).get(name)
        db = team_stats.get("entraineurs_disc", {}).get(name, {}).get(discipline) if discipline else None
        hb = None

    c = (gb or {}).get("c", 0)
    v = (gb or {}).get("v", 0)
    p = (gb or {}).get("p", 0)
    dc = (db or {}).get("c", 0)
    dv = (db or {}).get("v", 0)
    hc = (hb or {}).get("c", 0) if hb else 0
    hv = (hb or {}).get("v", 0) if hb else 0

    return c, v, p, dc, dv, hc, hv


def _extract_chimie_raw(cheval, driver, horse_stats):
    """Taux de victoire brut du duo cheval/driver. NaN si pas de données."""
    _nan = float('nan')
    if not cheval or not driver or not horse_stats:
        return 0, _nan
    data = horse_stats.get("with_driver", {}).get(cheval, {}).get(driver)
    if not data:
        return 0, _nan
    return data.get("c", 0), data.get("v", 0)


def _is_nan(v):
    """Teste si une valeur est NaN (int/float safe)."""
    try:
        return v != v  # NaN est le seul != lui-même
    except Exception:
        return False


# ── Nouvelles features v8.1 : momentum, streak, compétitivité ──

def _compute_momentum(perfs_detail):
    """Progression du cheval : avg_place_3récentes - avg_place_3précédentes.
    Négatif = en progression (places plus basses = mieux).
    NaN si moins de 4 courses."""
    _nan = float('nan')
    if not perfs_detail:
        return _nan
    places = []
    for course in perfs_detail[:6]:
        for p in course.get("participants", []):
            if p.get("itsHim"):
                place = (p.get("place") or {}).get("place", 0) or 0
                if place > 0:
                    places.append(place)
                break
    if len(places) < 4:
        return _nan
    recent = sum(places[:3]) / 3
    older_count = min(3, len(places) - 3)
    if older_count == 0:
        return _nan
    older = sum(places[3:3 + older_count]) / older_count
    return recent - older


def _compute_top3_streak(perfs_detail):
    """Nombre de top3 consécutifs en partant de la course la plus récente.
    Un cheval sur une série de 3+ top3 a une forte confiance."""
    if not perfs_detail:
        return 0
    streak = 0
    for course in perfs_detail[:10]:
        found = False
        for p in course.get("participants", []):
            if p.get("itsHim"):
                place = (p.get("place") or {}).get("place", 0) or 0
                if 0 < place <= 3:
                    streak += 1
                else:
                    return streak
                found = True
                break
        if not found:
            return streak
    return streak


def _compute_pct_battus(perfs_detail):
    """Pourcentage moyen de concurrents battus sur les 6 dernières courses.
    Finir 3e dans un plateau de 18 est beaucoup plus fort que 3e sur 6.
    NaN si pas assez de données."""
    _nan = float('nan')
    if not perfs_detail:
        return _nan
    pcts = []
    for course in perfs_detail[:6]:
        parts = course.get("participants", [])
        nb = len(parts)
        if nb < 2:
            continue
        for p in parts:
            if p.get("itsHim"):
                place = (p.get("place") or {}).get("place", 0) or 0
                if place > 0:
                    pcts.append((nb - place) / (nb - 1) * 100)
                break
    return sum(pcts) / len(pcts) if len(pcts) >= 2 else _nan


def _compute_worst_recent(perfs_detail, n=5):
    """Pire place parmi les N dernières courses (indicateur de fragilité).
    NaN si pas de données."""
    _nan = float('nan')
    if not perfs_detail:
        return _nan
    places = []
    for course in perfs_detail[:n]:
        for p in course.get("participants", []):
            if p.get("itsHim"):
                place = (p.get("place") or {}).get("place", 0) or 0
                if place > 0:
                    places.append(place)
                break
    return max(places) if places else _nan


def build_raw_features(p, perfs_detail, distance_course, driver, entraineur,
                       cheval, team_stats, horse_stats, elo_ratings, elo_hist,
                       discipline, hippodrome, nb_partants):
    """Construit le dict de features brutes pour le ML.
    NaN pour les données manquantes — XGBoost/LightGBM les ignorent nativement.
    """
    import math as _math
    _nan = float('nan')

    # --- Carrière ---
    nb_courses = p.get("nombreCourses", 0) or 0
    nb_vict = p.get("nombreVictoires", 0) or 0
    nb_place = p.get("nombrePlaces", 0) or 0
    gains = p.get("gainsParticipant", {}) or {}
    gains_carriere = gains.get("gainsCarriere", 0) or 0

    # --- Dernières places ---
    last5 = _extract_last_places(perfs_detail, 5)
    last_place = last5[0]

    # --- Agrégats perf ---
    top3_rate = compute_top3_rate(perfs_detail)
    top4_rate = compute_top4_rate(perfs_detail)
    avg_place_3 = compute_avg_place_recent(perfs_detail, 3)
    all_places = [pl for pl in last5 if not _is_nan(pl) and pl > 0]
    avg_place_5 = sum(all_places) / len(all_places) if all_places else _nan
    variance_places = 0.0
    if len(all_places) >= 3:
        mean_p = sum(all_places) / len(all_places)
        variance_places = sum((x - mean_p) ** 2 for x in all_places) / len(all_places)
    elif not all_places:
        variance_places = _nan

    # --- Jours depuis dernière course ---
    days_since = _nan
    if perfs_detail:
        for course in perfs_detail[:1]:
            date_ms = course.get("date")
            if date_ms:
                d = datetime.fromtimestamp(date_ms / 1000)
                days_since = max(0, (datetime.now() - d).days)

    nb_courses_mois = 0
    if perfs_detail:
        from datetime import timedelta
        cutoff = datetime.now() - timedelta(days=30)
        for course in perfs_detail:
            date_ms = course.get("date")
            if not date_ms: continue
            d = datetime.fromtimestamp(date_ms / 1000)
            if d >= cutoff:
                nb_courses_mois += 1

    # --- Distance ---
    dist_count, dist_avg_place, dist_win_rate = _extract_distance_raw(perfs_detail, distance_course)

    # --- Driver ---
    dr_c, dr_v, dr_p, dr_dc, dr_dv, dr_hc, dr_hv = _extract_team_raw(
        driver, "drivers", team_stats, discipline, hippodrome)

    # --- Entraineur ---
    en_c, en_v, en_p, en_dc, en_dv, en_hc, en_hv = _extract_team_raw(
        entraineur, "entraineurs", team_stats, discipline)

    # --- Chimie cheval/driver ---
    chimie_c, chimie_v = _extract_chimie_raw(cheval, driver, horse_stats)

    # --- Elo ---
    elo_val = elo_ratings.get(cheval, 1500) if elo_ratings else 1500
    elo_trend_raw = _nan
    if elo_hist and cheval in elo_hist:
        hist = elo_hist[cheval]
        if len(hist) >= 2:
            recent = hist[-1] if hist else elo_val
            older = hist[-2] if len(hist) >= 2 else elo_val
            elo_trend_raw = recent - older

    # --- Course ---
    rap = p.get("dernierRapportDirect") or p.get("dernierRapportReference")
    cote = float(rap["rapport"]) if rap and rap.get("rapport") else 0
    age = p.get("age") or 0

    return {
        # Carrière (6) — données réelles du cheval, pas de NaN ici
        "age": age,
        "nb_courses": nb_courses,
        "nb_victoires": nb_vict,
        "nb_places": nb_place,
        "win_rate": nb_vict / nb_courses * 100 if nb_courses > 0 else 0,
        "place_rate": nb_place / nb_courses * 100 if nb_courses > 0 else 0,
        # Gains (2)
        "gains_carriere_log": _math.log10(max(gains_carriere, 1)),
        "gains_par_course_log": _math.log10(max(gains_carriere / max(nb_courses, 1), 1)),
        # 5 dernières places (5) — NaN si pas de course
        "place_1": last5[0], "place_2": last5[1], "place_3": last5[2],
        "place_4": last5[3], "place_5": last5[4],
        # Agrégats perf (6) — NaN si pas de perfs
        "avg_place_3": avg_place_3 if avg_place_3 > 0 else _nan,
        "avg_place_5": avg_place_5,
        "top3_rate": top3_rate if top3_rate > 0 else _nan,
        "top4_rate": top4_rate if top4_rate > 0 else _nan,
        "variance_places": variance_places,
        "nb_raced_recent": len(all_places),
        # Dernière course (2)
        "last_place": last_place,
        "days_since_last": days_since,
        # Activité (1)
        "nb_courses_mois": nb_courses_mois,
        # Distance (3) — NaN si pas de course à cette distance
        "dist_similar_count": dist_count,
        "dist_avg_place": dist_avg_place,
        "dist_win_rate": dist_win_rate * 100 if not _is_nan(dist_win_rate) else _nan,
        # Driver global (3) — NaN si driver inconnu
        "driver_courses": dr_c,
        "driver_win_rate": dr_v / dr_c * 100 if dr_c > 0 else _nan,
        "driver_place_rate": dr_p / dr_c * 100 if dr_c > 0 else _nan,
        # Driver discipline (2) — NaN si jamais couru dans cette discipline
        "driver_disc_courses": dr_dc,
        "driver_disc_win_rate": dr_dv / dr_dc * 100 if dr_dc > 0 else _nan,
        # Driver hippodrome (2) — NaN si jamais couru sur cet hippo
        "driver_hippo_courses": dr_hc,
        "driver_hippo_win_rate": dr_hv / dr_hc * 100 if dr_hc > 0 else _nan,
        # Entraineur (3) — NaN si entraineur inconnu
        "entraineur_courses": en_c,
        "entraineur_win_rate": en_v / en_c * 100 if en_c > 0 else _nan,
        "entraineur_disc_win_rate": en_dv / en_dc * 100 if en_dc > 0 else _nan,
        # Chimie cheval/driver (2) — NaN si jamais associé
        "chimie_courses": chimie_c,
        "chimie_win_rate": chimie_v / chimie_c * 100 if chimie_c > 0 else _nan,
        # Elo (2)
        "elo": elo_val,
        "elo_trend_raw": elo_trend_raw,
        # Course (6)
        "nb_partants": nb_partants,
        "cote": cote,
        "inv_cote": 1.0 / max(cote, 1),
        "is_female": 1 if p.get("sexe") == "FEMELLES" else 0,
        "has_oeilleres": 1 if p.get("oeilleres") and p.get("oeilleres") != "SANS_OEILLERES" else 0,
        "is_deferre": 1 if "DEFERRE" in (p.get("deferre", "") or "") else 0,
        "driver_changed": 1 if p.get("driverChange") else 0,
        # Bonus contextuels (2)
        "bonus_team": 1 if driver and entraineur and driver == entraineur else 0,
        "bonus_deferre": 1 if "DEFERRE" in (p.get("deferre", "") or "") else 0,
        # ═══ Momentum & forme avancée (6) — v8.1 ═══
        # Progression : négatif = en amélioration (places plus basses)
        "momentum_3": _compute_momentum(perfs_detail),
        # Série de top3 consécutifs (0 = pas de série)
        "nb_top3_consecutif": _compute_top3_streak(perfs_detail),
        # % de concurrents battus (finir 3e/18 > 3e/6)
        "pct_battus_recent": _compute_pct_battus(perfs_detail),
        # Discipline : 0=TROT_ATTELE, 1=TROT_MONTE, 2=GALOP, 3=AUTRE
        "discipline_code": {"TROT_ATTELE": 0, "TROT_MONTE": 1, "GALOP": 2}.get(discipline, 3),
        # Position de départ (numPmu = numéro de corde, avantage au trot)
        "corde_numero": p.get("numPmu", 0) or 0,
        # Pire place récente (indicateur fragilité)
        "worst_recent_place": _compute_worst_recent(perfs_detail),
    }


# ============================================================
#  ML featurization v7 — 44 features brutes (0% synthétique)
# ============================================================
def featurize(p, nb_partants):
    """ML featurization v8.1 — 54 features brutes (0% synthétique).
    Utilise p["raw"] si disponible (v7+), sinon fallback sur p["scores"] (v6).
    """
    raw = p.get("raw")
    if raw:
        # === v7 : toutes les features brutes ===
        return [
            # Carrière (6)
            raw["age"],
            raw["nb_courses"],
            raw["nb_victoires"],
            raw["nb_places"],
            raw["win_rate"],
            raw["place_rate"],
            # Gains (2)
            raw["gains_carriere_log"],
            raw["gains_par_course_log"],
            # 5 dernières places (5)
            raw["place_1"], raw["place_2"], raw["place_3"],
            raw["place_4"], raw["place_5"],
            # Agrégats perf (6)
            raw["avg_place_3"],
            raw["avg_place_5"],
            raw["top3_rate"],
            raw["top4_rate"],
            raw["variance_places"],
            raw["nb_raced_recent"],
            # Dernière course (2)
            raw["last_place"],
            raw["days_since_last"],
            # Activité (1)
            raw["nb_courses_mois"],
            # Distance (3)
            raw["dist_similar_count"],
            raw["dist_avg_place"],
            raw["dist_win_rate"],
            # Driver global (3)
            raw["driver_courses"],
            raw["driver_win_rate"],
            raw["driver_place_rate"],
            # Driver discipline (2)
            raw["driver_disc_courses"],
            raw["driver_disc_win_rate"],
            # Driver hippodrome (2)
            raw["driver_hippo_courses"],
            raw["driver_hippo_win_rate"],
            # Entraineur (3)
            raw["entraineur_courses"],
            raw["entraineur_win_rate"],
            raw["entraineur_disc_win_rate"],
            # Chimie cheval/driver (2)
            raw["chimie_courses"],
            raw["chimie_win_rate"],
            # Elo (2)
            raw["elo"],
            raw["elo_trend_raw"],
            # Course (6)
            raw["nb_partants"],
            raw["cote"],
            raw["inv_cote"],
            raw["is_female"],
            raw["has_oeilleres"],
            raw["is_deferre"],
            # Contexte (3)
            raw["driver_changed"],
            raw["bonus_team"],
            raw["bonus_deferre"],
            # Momentum & forme avancée (6) — v8.1
            raw["momentum_3"],
            raw["nb_top3_consecutif"],
            raw["pct_battus_recent"],
            raw["discipline_code"],
            raw["corde_numero"],
            raw["worst_recent_place"],
        ]

    # === Fallback v6 : scores synthétiques (compatibilité anciens modèles) ===
    s = p["scores"]
    forme = s.get("forme", 0)
    elo = s.get("elo", 50)
    driver = s.get("driver", 50)
    return [
        forme, s.get("carriere", 0), s.get("gains", 0), driver,
        s.get("entraineur", 50), s.get("distance", 50), s.get("cheval_stats", 50),
        elo, s.get("age_sexe", 50), s.get("repos", 50), s.get("elo_trend", 50),
        s.get("confrontation", 50), s.get("pedigree", 50), s.get("corde", 50),
        s.get("equipment", 50), s.get("profile_match", 50), nb_partants,
        1.0 / max(p.get("cote") or 50, 1), p["bonus"].get("team", 0),
        p["bonus"].get("deferre", 0), 1 if p.get("sexe") == "FEMELLES" else 0,
        forme * elo / 100, driver * s.get("entraineur", 50) / 100, abs(forme - 50),
        s.get("driver_hippo", 50), s.get("regularite", 50), s.get("equip_change", 50),
        s.get("style_attaquant", 50), s.get("style_finisseur", 50),
        s.get("gains_trend", 50), s.get("jours_derniere", 50), s.get("nb_courses_mois", 50),
        s.get("perf_terrain", 50), s.get("corde_avantage", 50), s.get("chimie_driver", 50),
        s.get("regularite", 50) * forme / 100,
        s.get("driver_hippo", 50) * s.get("perf_terrain", 50) / 100,
        s.get("top3_rate", 0), s.get("last_place", 0), s.get("avg_place_3", 0),
        s.get("win_rate", 0), s.get("place_rate", 0),
    ]


FEATURE_NAMES = [
    # Carrière (6)
    "age","nb_courses","nb_victoires","nb_places","win_rate","place_rate",
    # Gains (2)
    "gains_carriere_log","gains_par_course_log",
    # 5 dernières places (5)
    "place_1","place_2","place_3","place_4","place_5",
    # Agrégats perf (6)
    "avg_place_3","avg_place_5","top3_rate","top4_rate","variance_places","nb_raced_recent",
    # Dernière course (2)
    "last_place","days_since_last",
    # Activité (1)
    "nb_courses_mois",
    # Distance (3)
    "dist_similar_count","dist_avg_place","dist_win_rate",
    # Driver global (3)
    "driver_courses","driver_win_rate","driver_place_rate",
    # Driver discipline (2)
    "driver_disc_courses","driver_disc_win_rate",
    # Driver hippodrome (2)
    "driver_hippo_courses","driver_hippo_win_rate",
    # Entraineur (3)
    "entraineur_courses","entraineur_win_rate","entraineur_disc_win_rate",
    # Chimie cheval/driver (2)
    "chimie_courses","chimie_win_rate",
    # Elo (2)
    "elo","elo_trend_raw",
    # Course (6)
    "nb_partants","cote","inv_cote","is_female","has_oeilleres","is_deferre",
    # Contexte (3)
    "driver_changed","bonus_team","bonus_deferre",
    # Momentum & forme avancée (6) — v8.1
    "momentum_3","nb_top3_consecutif","pct_battus_recent",
    "discipline_code","corde_numero","worst_recent_place",
]


def load_ml_model():
    # Priorité : v8 > v7 > v5 > v4
    if HAS_V8:
        v8 = load_v8(ML_MODEL_WIN_V8_FILE)
        if v8:
            return v8
    if HAS_V7:
        v7 = load_v7(ML_MODEL_WIN_V7_FILE)
        if v7:
            return v7
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
    if HAS_V8:
        v8 = load_v8(ML_MODEL_TOP4_V8_FILE)
        if v8:
            return v8
    if HAS_V7:
        v7 = load_v7(ML_MODEL_TOP4_V7_FILE)
        if v7:
            return v7
    if HAS_ADVANCED:
        adv = load_advanced(ML_MODEL_TOP4_V5_FILE)
        if adv:
            return adv
    payload = load_pickle(ML_MODEL_TOP4_FILE, max_age_hours=24*14)
    return load_model_from_dict(payload) if payload else None


def save_ml_model_top4(model):
    save_pickle(ML_MODEL_TOP4_FILE, model.to_dict())

def load_ml_model_top3():
    """Charge le modèle TOP3 (v8 > v7 > advanced > ensemble numpy)."""
    if HAS_V8:
        v8 = load_v8(ML_MODEL_TOP3_V8_FILE)
        if v8:
            return v8
    if HAS_V7:
        v7 = load_v7(ML_MODEL_TOP3_V7_FILE)
        if v7:
            return v7
    if HAS_ADVANCED:
        adv = load_advanced(ML_MODEL_TOP3_V5_FILE)
        if adv:
            return adv
    payload = load_pickle(ML_MODEL_TOP3_FILE, max_age_hours=24*14)
    if payload:
        return load_model_from_dict(payload)
    return None

def save_ml_model_top3(model):
    save_pickle(ML_MODEL_TOP3_FILE, model.to_dict())


def load_calibration():
    return load_pickle(CALIBRATION_FILE, max_age_hours=24*7)


def save_calibration(c):
    save_pickle(CALIBRATION_FILE, c)


def load_ml_model_discipline(discipline):
    """Charge le modèle WIN spécifique à la discipline (TROT ou GALOP).
    Fallback sur le modèle générique si pas disponible.
    """
    if not HAS_V8:
        return load_ml_model()
    code = {"TROT_ATTELE": 0, "TROT_MONTE": 1, "GALOP": 2}.get(discipline, 3)
    if code in (0, 1):  # TROT
        m = load_v8(ML_MODEL_WIN_V8_TROT_FILE)
        if m:
            return m
    elif code == 2:  # GALOP
        m = load_v8(ML_MODEL_WIN_V8_GALOP_FILE)
        if m:
            return m
    # Fallback : modèle générique
    return load_ml_model()


def load_ml_model_top3_discipline(discipline):
    """Charge le modèle TOP3 spécifique à la discipline."""
    if not HAS_V8:
        return load_ml_model_top3()
    code = {"TROT_ATTELE": 0, "TROT_MONTE": 1, "GALOP": 2}.get(discipline, 3)
    if code in (0, 1):
        m = load_v8(ML_MODEL_TOP3_V8_TROT_FILE)
        if m:
            return m
    elif code == 2:
        m = load_v8(ML_MODEL_TOP3_V8_GALOP_FILE)
        if m:
            return m
    return load_ml_model_top3()


def load_ml_ranker():
    """Charge le modèle Ranker séquentiel (XGBRanker)."""
    if not HAS_V8:
        return None
    return load_ranker_v8(ML_MODEL_RANKER_V8_FILE)


def _fetch_full(args):
    date_str, r_num, c_num, distance, discipline, hippodrome, type_corde = args
    try:
        return (get_participants(date_str, r_num, c_num),
                get_performances(date_str, r_num, c_num),
                distance, discipline, hippodrome, type_corde)
    except Exception:
        return None


def train_ml_model(days_back=45, exclude_recent=0, n_trees_gbm=50, n_trees_rf=30,
                   model_type="ensemble"):
    """Entraîne 2 modèles : WIN (y=1 si gagnant) + TOP4 (y=1 si ≤4e).
    Les deux partagent les mêmes features X mais ont des labels différents."""
    try:
        import numpy as np
    except ImportError:
        return None

    X, y_win, y_top4, y_top3, course_ids = [], [], [], [], []
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

    # Course-level accumulator for intra-course features
    _course_buffers = {}  # course_key -> list of feature dicts (for v8 ranking)
    _course_idx = 0  # simple counter for unique keys

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
        _course_key = f"c{_course_idx}"
        _course_idx += 1
        _course_buf = []
        for a in analyses:
            feat_vec = featurize(a, nb)
            real = next((p for p in parts["participants"]
                        if p.get("numPmu") == a["numPmu"]), None)
            place = (real.get("ordreArrivee") or 0) if real else 0
            _course_buf.append({
                "feat": feat_vec,
                "y_win": 1 if place == 1 else 0,
                "y_top3": 1 if 1 <= place <= 3 else 0,
                "y_top4": 1 if 1 <= place <= 4 else 0,
                "y_place": place,  # place réelle pour le ranker
            })

        # Toujours stocker le buffer
        if len(_course_buf) >= 2:
            _course_buffers[_course_key] = _course_buf

    # ===========================================================
    #  Assembler les features avec ranking intra-course (v8)
    #  ou sans ranking (autres modes)
    # ===========================================================
    X, y_win, y_top3, y_top4, y_places, course_ids = [], [], [], [], [], []

    is_v8_mode = model_type in ("advanced_v8", "advanced_v8_no_optuna") and HAS_V8

    for ck, buf in _course_buffers.items():
        course_feats = [b["feat"] for b in buf]

        # Calculer les features de ranking intra-course (14 features) seulement en v8
        if is_v8_mode:
            try:
                from lib.ml_v8 import compute_course_ranking_features
                ranking_feats = compute_course_ranking_features(course_feats)
            except Exception:
                ranking_feats = [np.zeros(14) for _ in buf]
        else:
            ranking_feats = None

        for i, b in enumerate(buf):
            if ranking_feats is not None:
                X.append(list(b["feat"]) + list(ranking_feats[i]))
            else:
                X.append(list(b["feat"]))
            y_win.append(b["y_win"])
            y_top3.append(b["y_top3"])
            y_top4.append(b["y_top4"])
            y_places.append(b["y_place"])
            course_ids.append(ck)

    if len(X) < 100:
        return None

    n_top4 = sum(y_top4)
    n_top3 = sum(y_top3)
    n_win = sum(y_win)
    print(f"[ML v6] {len(X)} échantillons")
    print(f"  WIN  : {n_win} victoires ({n_win/len(X)*100:.1f}%)")
    print(f"  TOP3 : {n_top3} placés ({n_top3/len(X)*100:.1f}%)")
    print(f"  TOP4 : {n_top4} placés ({n_top4/len(X)*100:.1f}%)")

    info = {"n_samples": len(X), "trained_at": datetime.now().isoformat(),
            "model_type": model_type, "win_positives": n_win,
            "top3_positives": n_top3, "top4_positives": n_top4}

    # ===========================================================
    #  TRAINING — modèle v8 (Optuna + TS-CV + Feature Eng V8)
    # ===========================================================
    if model_type in ("advanced_v8", "advanced_v8_no_optuna") and HAS_V8:
        # X contient déjà 54+14=68 features (base + ranking intra-course)
        # Ajouter les 16 interactions → 84 features total
        from lib.ml_v8 import augment_course_level
        X_v8 = []
        for sample in X:
            X_v8.append(_engineer_v8_from_vector(sample[:54]) + list(sample[54:]))
        X = X_v8

        # Data augmentation au niveau course
        X_aug, y_win_aug = augment_course_level(X, y_win, course_ids)
        _, y_top3_aug = augment_course_level(X, y_top3, course_ids)
        _, y_top4_aug = augment_course_level(X, y_top4, course_ids)

        # Vérifier cohérence (augment garde les mêmes tailles)
        if len(X_aug) != len(y_win_aug):
            X_aug, y_win_aug = X, y_win
            _, _ = y_top3, y_top4
            y_top3_aug, y_top4_aug = y_top3, y_top4

        print(f"[ML v8] {len(X_aug)} échantillons (dont {len(X_aug)-len(X)} augmentés)")
        print(f"[ML v8] {len(X_aug[0])} features (54 base + 14 ranking + 16 interactions)")
        X = X_aug
        y_win = y_win_aug
        y_top3 = y_top3_aug
        y_top4 = y_top4_aug

        use_optuna = model_type == "advanced_v8"

        # --- Modèle WIN ---
        print("[ML v8] 🔵 Entraînement WIN (Optuna + XGBoost + LightGBM + TabNet)...")
        train_v8(X, y_win, ML_MODEL_WIN_V8_FILE, target="win", use_optuna=use_optuna)

        # --- Modèle TOP3 ---
        print("[ML v8] 🟡 Entraînement TOP3...")
        train_v8(X, y_top3, ML_MODEL_TOP3_V8_FILE, target="top3", use_optuna=use_optuna)

        # --- Modèle TOP4 ---
        print("[ML v8] 🟢 Entraînement TOP4...")
        train_v8(X, y_top4, ML_MODEL_TOP4_V8_FILE, target="top4", use_optuna=use_optuna)

        # --- Ranker séquentiel (XGBRanker) ---
        print("[ML v8] 🏆 Entraînement Ranker (XGBRanker — ordre d'arrivée)...")
        ranker_groups = course_ids  # utiliser les course_ids comme groupes
        if len(X) >= 200:
            train_ranker_v8(X, y_places, ranker_groups, ML_MODEL_RANKER_V8_FILE)
        else:
            print("  ⚠️ Pas assez de données pour le ranker")

        # --- Modèles par discipline (TROT vs GALOP) ---
        _train_discipline_models(X, y_win, y_top3, y_places, course_ids, use_optuna)

        info["win_model"] = f"V8 stack ({len(X[0])}feat, {len(X)}samples)"
        info["top3_model"] = info["win_model"]
        info["top4_model"] = info["win_model"]
        info["models_trained"] = ["win", "top3", "top4", "ranker"]
        info["n_features"] = len(X[0])
        info["n_samples_original"] = len(set(course_ids))
        info["n_samples_augmented"] = len(X)
        return info

    # ===========================================================
    #  TRAINING — modèle v7 (XGBoost + LightGBM + TabNet)
    # ===========================================================
    if model_type == "advanced_v7" and HAS_V7:
        # --- Modèle WIN ---
        print("[ML v7] 🔵 Entraînement WIN (XGBoost + LightGBM + TabNet)...")
        train_v7(X, y_win, ML_MODEL_WIN_V7_FILE, target="win")

        # --- Modèle TOP3 ---
        print("[ML v7] 🟡 Entraînement TOP3 (XGBoost + LightGBM + TabNet)...")
        train_v7(X, y_top3, ML_MODEL_TOP3_V7_FILE, target="top3")

        # --- Modèle TOP4 ---
        print("[ML v7] 🟢 Entraînement TOP4 (XGBoost + LightGBM + TabNet)...")
        train_v7(X, y_top4, ML_MODEL_TOP4_V7_FILE, target="top4")

        info["win_model"] = "XGBoost + LightGBM + HGB + TabNet (Stacking V7)"
        info["top3_model"] = "XGBoost + LightGBM + HGB + TabNet (Stacking V7)"
        info["top4_model"] = "XGBoost + LightGBM + HGB + TabNet (Stacking V7)"
        info["models_trained"] = ["win", "top3", "top4"]
        return info

    # ===========================================================
    #  TRAINING — modèle avancé (Stacking sklearn v5)
    # ===========================================================
    if model_type == "advanced" and HAS_ADVANCED:
        # --- Modèle WIN ---
        print("[ML v6] 🔵 Entraînement WIN (stacking avancé)...")
        train_advanced(X, y_win, ML_MODEL_FILE_V5)

        # --- Modèle TOP3 ---
        print("[ML v6] 🟡 Entraînement TOP3 (stacking avancé)...")
        train_advanced(X, y_top3, ML_MODEL_TOP3_V5_FILE)

        # --- Modèle TOP4 ---
        print("[ML v6] 🟢 Entraînement TOP4 (stacking avancé)...")
        train_advanced(X, y_top4, ML_MODEL_TOP4_V5_FILE)

        info["win_model"] = "LGBM+CatBoost+HGB+RF+LR (Stacking)"
        info["top3_model"] = "LGBM+CatBoost+HGB+RF+LR (Stacking)"
        info["top4_model"] = "LGBM+CatBoost+HGB+RF+LR (Stacking)"
        info["models_trained"] = ["win", "top3", "top4"]
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

    # --- Modèle TOP3 (ensemble numpy) ---
    gbm_t3 = None
    rf_t3 = None
    if model_type in ("ensemble", "gbm"):
        print(f"[ML v6] 🟡 Entraînement TOP3 GBM ({n_trees_gbm} arbres)...")
        gbm_t3 = GradientBoosting(n_trees=n_trees_gbm, max_depth=3, learning_rate=0.1)
        gbm_t3.fit(X, y_top3)
    if model_type in ("ensemble", "rf"):
        print(f"[ML v6] 🟡 Entraînement TOP3 RF ({n_trees_rf} arbres)...")
        rf_t3 = RandomForest(n_trees=n_trees_rf, max_depth=8, min_samples=15)
        rf_t3.fit(X, y_top3)

    if model_type == "ensemble":
        model_t3 = Ensemble(gbm=gbm_t3, rf=rf_t3, w_gbm=0.6, w_rf=0.4)
    elif model_type == "gbm":
        model_t3 = gbm_t3
    else:
        model_t3 = rf_t3

    print("[ML v6] 🟡 Sauvegarde modèle TOP3...")
    save_ml_model_top3(model_t3)
    info["models_trained"] = ["win", "top3", "top4"]

    return info


def _engineer_v8_from_vector(v):
    """
    Prend le vecteur de 54 features (depuis featurize()) et retourne
    54 + 16 = 70 features avec les 16 interactions v8.1.
    
    Index mapping du vecteur v (54 features v8.1) :
      0-5   : carrière (age, nb_courses, nb_victoires, nb_places, win_rate, place_rate)
      6-7   : gains (gains_carriere_log, gains_par_course_log)
      8-12  : 5 dernières places (place_1..5)
      13-18 : agrégats (avg_place_3, avg_place_5, top3_rate, top4_rate, variance, nb_raced)
      19-20 : dernière course (last_place, days_since_last)
      21    : activité (nb_courses_mois)
      22-24 : distance (dist_count, dist_avg, dist_wr)
      25-27 : driver global (dr_courses, dr_wr, dr_pr)
      28-29 : driver discipline (dr_disc_courses, dr_disc_wr)
      30-31 : driver hippo (dr_hippo_courses, dr_hippo_wr)
      32-34 : entraineur (en_courses, en_wr, en_disc_wr)
      35-36 : chimie (chimie_courses, chimie_wr)
      37-38 : elo (elo, elo_trend)
      39-44 : course (nb_partants, cote, inv_cote, is_female, has_oeilleres, is_deferre)
      45-47 : contexte (driver_changed, bonus_team, bonus_deferre)
      48-53 : momentum & forme avancée (momentum_3, nb_top3_consecutif, pct_battus,
                                        discipline_code, corde_numero, worst_recent_place)
    """
    # Sécuriser
    def safe(idx, default=0.0):
        val = v[idx] if idx < len(v) else default
        return float(val) if val is not None else default

    age = safe(0)
    nb_courses = max(safe(1), 1.0)
    win_rate = safe(4)
    place_rate = safe(5)
    gains_pc_log = safe(7)
    avg_place_3 = safe(13)
    avg_place_5 = safe(14)
    top3_rate = safe(15)
    top4_rate = safe(16)
    variance = safe(17)
    place_1 = safe(8)
    last_place = safe(19)
    days_since = safe(20)
    nb_courses_mois = safe(21)
    dist_count = safe(22)
    dist_avg = safe(23)
    dist_wr = safe(24)
    dr_wr = safe(26)
    dr_pr = safe(27)
    dr_disc_wr = safe(29)
    dr_hippo_wr = safe(31)
    en_wr = safe(33)
    en_disc_wr = safe(34)
    chimie_courses = safe(35)
    chimie_wr = safe(36)
    elo = safe(37)
    elo_trend = safe(38)
    nb_partants = max(safe(39), 1.0)
    cote = max(safe(40), 1.0)
    inv_cote = safe(41)
    driver_courses = safe(25)

    # v8.1 : nouvelles features momentum & forme
    momentum_3 = safe(48)
    streak = safe(49)
    pct_battus = safe(50)
    discipline_code = safe(51)
    corde_num = safe(52)
    worst_recent = safe(53)

    interactions = [
        # 1. forme × cote (value bet signal)
        avg_place_3 * cote,
        # 2. momentum (amélioration récente)
        avg_place_5 - place_1 if (avg_place_5 > 0 and place_1 > 0) else 0.0,
        # 3. driver × entraineur
        dr_wr * en_wr / 100.0 if dr_wr > 0 and en_wr > 0 else 0.0,
        # 4. elo × marché
        elo * inv_cote * 100,
        # 5. régularité (négatif = mieux)
        -variance,
        # 6. exp distance ratio
        dist_count / nb_courses,
        # 7. spécialisation driver
        dr_disc_wr / max(dr_wr, 1.0),
        # 8. proba marché
        inv_cote * 100,
        # 9. chimie relative
        chimie_wr / max(dr_wr, 1.0) if chimie_courses >= 2 else 0.0,
        # 10. tendance pondérée
        elo_trend * nb_courses_mois,
        # 11. dernier top3
        1.0 if 0 < last_place <= 3 else 0.0,
        # 12. inactivité score
        min(days_since, 120) / 30.0,
        # 13. progression gains
        gains_pc_log,
        # 14. compétitivité
        1.0 / nb_partants * 100,
        # ═══ v8.1 : 2 nouvelles interactions haute valeur ═══
        # 15. momentum × marché : cheval en progression + sous-estimé par le marché
        momentum_3 * inv_cote * 100,
        # 16. puissance_série : streak × pct_battus (confiance + compétitivité)
        streak * pct_battus / 100.0,
    ]
    return list(v) + interactions


def predict_ml(features, model, calibration=None):
    """Prédiction ML — enrichit en v8 si le modèle l'attend."""
    # Si le modèle v8 attend 62+ features et qu'on en a 48, enrichir
    if HAS_V8 and hasattr(model, 'stacking') and hasattr(model.stacking, 'n_features'):
        if model.stacking.n_features and len(features) < model.stacking.n_features:
            features = _engineer_v8_from_vector(features)
    elif HAS_V8 and hasattr(model, 'model') and isinstance(model.model, dict):
        nf = model.n_features
        if nf and len(features) < nf:
            features = _engineer_v8_from_vector(features)
    p = model.predict_one(features)
    if calibration:
        p = apply_calibration(p, calibration)
    return p


def _train_discipline_models(X, y_win, y_top3, y_places, course_ids, use_optuna=False):
    """Entraîne des modèles séparés pour TROT et GALOP.
    Les courses de trot et de galop ont des dynamiques fondamentalement différentes.
    Chaque discipline a son propre ensemble WIN + TOP3 + Ranker.
    """
    import numpy as _np
    
    X = _np.asarray(X)
    y_win = _np.asarray(y_win)
    y_top3 = _np.asarray(y_top3)
    y_places = _np.asarray(y_places, dtype=float)
    course_ids = list(course_ids)
    
    # Feature 51 = discipline_code (0=TROT_ATTELE, 1=TROT_MONTE, 2=GALOP)
    disc_col = 51
    if X.shape[1] <= disc_col:
        print("  [Discipline] ⚠️ discipline_code pas dans les features, skip")
        return
    
    disc_vals = X[:, disc_col]
    
    # Regrouper : TROT (0+1) vs GALOP (2)
    trot_mask = _np.isin(disc_vals, [0, 1])
    galop_mask = disc_vals == 2
    
    for label, mask, win_file, top3_file in [
        ("TROT", trot_mask, ML_MODEL_WIN_V8_TROT_FILE, ML_MODEL_TOP3_V8_TROT_FILE),
        ("GALOP", galop_mask, ML_MODEL_WIN_V8_GALOP_FILE, ML_MODEL_TOP3_V8_GALOP_FILE),
    ]:
        n_samples = mask.sum()
        n_pos = y_win[mask].sum()
        if n_samples < 200 or n_pos < 10:
            print(f"  [Discipline] ⚠️ {label}: {n_samples} samples ({int(n_pos)}+) — pas assez, skip")
            continue
        
        X_disc = X[mask]
        y_win_disc = y_win[mask]
        y_top3_disc = y_top3[mask]
        
        # Reconstruire les course_ids pour cette discipline
        disc_indices = _np.where(mask)[0]
        disc_course_ids = [course_ids[i] for i in disc_indices]
        # Re-numéroter pour éviter les conflits
        unique_courses = list(set(disc_course_ids))
        course_map = {c: f"{label[0]}_{i}" for i, c in enumerate(unique_courses)}
        disc_course_ids = [course_map[c] for c in disc_course_ids]
        
        print(f"\n  [Discipline] 🏇 {label}: {n_samples} samples, {int(n_pos)}+ ({n_pos/n_samples*100:.1f}%)")
        
        try:
            print(f"  [Discipline] {label} — WIN...")
            train_v8(X_disc, y_win_disc, win_file, target=f"win_{label.lower()}", use_optuna=use_optuna)
            print(f"  [Discipline] {label} — TOP3...")
            train_v8(X_disc, y_top3_disc, top3_file, target=f"top3_{label.lower()}", use_optuna=use_optuna)
        except Exception as e:
            print(f"  [Discipline] ⚠️ {label} erreur: {e}")


def _enrich_features_v8(analyses):
    """
    Pour une course complète, enrichit les features de chaque cheval
    avec le ranking intra-course + interactions v8.
    Retourne une liste de feature vectors (84 features).
    """
    nb = len(analyses)
    base_feats = [featurize(a, nb) for a in analyses]

    # Features ranking intra-course (14 features)
    try:
        from lib.ml_v8 import compute_course_ranking_features
        ranking_feats = compute_course_ranking_features(base_feats)
    except Exception:
        ranking_feats = [np.zeros(14) for _ in base_feats]

    # Combiner : 54 base + 14 ranking + 16 interactions = 84
    enriched = []
    for i in range(nb):
        vec54 = base_feats[i]
        interactions = _engineer_v8_from_vector(vec54)[54:]  # les 16 interactions
        full = list(vec54) + list(ranking_feats[i]) + interactions
        enriched.append(full)
    return enriched


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


def compute_top3_rate(perfs_detail):
    """Taux de top3 sur les 10 dernières courses."""
    if not perfs_detail:
        return 0.0
    top3 = 0
    total = 0
    for course in perfs_detail[:10]:
        for p in course.get("participants", []):
            if p.get("itsHim"):
                place = (p.get("place") or {}).get("place", 0) or 0
                if place > 0:
                    total += 1
                    if place <= 3:
                        top3 += 1
    if total == 0:
        return 0.0
    return (top3 / total) * 100


def compute_last_place(perfs_detail):
    """Place de la dernière course (0 si inconnu)."""
    if not perfs_detail:
        return 0
    for course in perfs_detail[:1]:
        for p in course.get("participants", []):
            if p.get("itsHim"):
                return (p.get("place") or {}).get("place", 0) or 0
    return 0


def compute_avg_place_recent(perfs_detail, n=3):
    """Place moyenne sur les N dernières courses. 0 si pas de données."""
    if not perfs_detail:
        return 0.0
    places = []
    for course in perfs_detail[:n]:
        for p in course.get("participants", []):
            if p.get("itsHim"):
                place = (p.get("place") or {}).get("place", 0) or 0
                if place > 0:
                    places.append(place)
                break
    if not places:
        return 0.0
    return sum(places) / len(places)


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
            s_top4_raw = 0.60 * s_top4_rate + 0.40 * s_regularity

            # === NEW v6.2 — Features haute valeur pour Top3 ML ===
            s_top3_rate = compute_top3_rate(perfs_detail)       # Taux top3 réel
            s_last_place = compute_last_place(perfs_detail)     # Place dernière course
            s_avg_place_3 = compute_avg_place_recent(perfs_detail, 3)  # Moyenne 3 dernières
            s_win_rate = (nb_vict / nb_courses * 100) if nb_courses > 0 else 0   # Taux victoire réel
            s_place_rate = (nb_place / nb_courses * 100) if nb_courses > 0 else 0  # Taux placement réel

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
            s_top3_rate = 0.0; s_last_place = 0; s_avg_place_3 = 0.0
            s_win_rate = 0.0; s_place_rate = 0.0
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
                # NEW v6.2 — Features haute valeur Top3
                "top3_rate": round(float(s_top3_rate or 0), 1),
                "last_place": round(float(s_last_place or 0), 1),
                "avg_place_3": round(float(s_avg_place_3 or 0), 1),
                "win_rate": round(float(s_win_rate or 0), 1),
                "place_rate": round(float(s_place_rate or 0), 1),
            },
            "top4": {
                "rate": round(float(s_top4_rate or 0), 1),
                "regularity": round(float(s_regularity or 50), 1),
                "raw": round(float(s_top4_raw or 20), 1),
            },
            "bonus": {"team": bonus_team, "deferre": bonus_deferre},
            "_perfs_detail": perfs_detail,  # pour calcul Top 4
            # NEW v7 — Raw features pour ML (données brutes, pas de scoring)
            "raw": build_raw_features(
                p, perfs_detail, distance, driver, entr,
                cheval, team_stats, horse_stats, elo, elo_hist,
                discipline, hippodrome, nb_partants),
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

    ml_model = load_ml_model_discipline(discipline) if use_ml else None
    calib = load_calibration() if use_ml else None
    chances_ml = None

    # V8 enrichissement intra-course (ranking + interactions)
    v8_feats = None
    if ml_model and HAS_V8:
        try:
            v8_feats = _enrich_features_v8(analyses)
        except Exception:
            v8_feats = None

    if ml_model:
        nb = len(analyses)
        if v8_feats and len(v8_feats[0]) >= 62:
            raw_ml = [predict_ml(v8_feats[i], ml_model, calib) for i in range(nb)]
        else:
            raw_ml = [predict_ml(featurize(a, nb), ml_model, calib) for a in analyses]
        total_ml = sum(raw_ml) or 1
        chances_ml = [x / total_ml * 100 for x in raw_ml]

    # NEW v8.1 — Ranker séquentiel (XGBRanker)
    ml_ranker = load_ml_ranker() if use_ml else None
    raw_ranker_scores = None
    if ml_ranker:
        nb = len(analyses)
        if v8_feats and len(v8_feats[0]) >= 62:
            raw_ranker_scores = [predict_ml(v8_feats[i], ml_ranker) for i in range(nb)]
        else:
            raw_ranker_scores = [predict_ml(featurize(a, nb), ml_ranker) for a in analyses]
        # Normaliser les scores du ranker en probabilités relatives
        total_ranker = sum(raw_ranker_scores) or 1
        raw_ranker_scores = [s / total_ranker * 100 for s in raw_ranker_scores]

    # NEW v6 — Top 4 ML model (modèle binaire placement)
    ml_top4_model = load_ml_model_top4() if use_ml else None
    raw_top4_ml = None
    if ml_top4_model:
        nb = len(analyses)
        if v8_feats and len(v8_feats[0]) >= 62:
            raw_top4_ml = [predict_ml(v8_feats[i], ml_top4_model) for i in range(nb)]
        else:
            raw_top4_ml = [predict_ml(featurize(a, nb), ml_top4_model) for a in analyses]
        # Normaliser : dans une course, exactement 4 chevaux sont dans le top 4
        total_top4 = sum(raw_top4_ml) or 1
        raw_top4_ml = [p / total_top4 * 4 for p in raw_top4_ml]

    # NEW v6.1 — Top 3 ML model (modèle dédié placement top3, ~25% positifs)
    ml_top3_model = load_ml_model_top3_discipline(discipline) if use_ml else None
    raw_top3_ml = None
    if ml_top3_model:
        nb = len(analyses)
        if v8_feats and len(v8_feats[0]) >= 62:
            raw_top3_ml = [predict_ml(v8_feats[i], ml_top3_model) for i in range(nb)]
        else:
            raw_top3_ml = [predict_ml(featurize(a, nb), ml_top3_model) for a in analyses]
        # Normaliser : dans une course, exactement 3 chevaux sont dans le top 3
        total_top3 = sum(raw_top3_ml) or 1
        raw_top3_ml = [p / total_top3 * 3 for p in raw_top3_ml]
        # Log pour debug
        print(f"[TOP3 ML] {nb} chevaux, raw min={min(raw_top3_ml)*100:.1f}% max={max(raw_top3_ml)*100:.1f}%")

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
            ml_top4_prob = min(ml_top4_prob, 95.0)
            # Sauvegarder l'heuristique avant blend
            a["chanceTop4Heur"] = round(a["chanceTop4"], 2)
            a["chanceTop4ML"] = round(ml_top4_prob, 2)
            # Blend : 20% heuristique + 80% ML Top4
            a["chanceTop4"] = round(0.2 * a["chanceTop4Heur"] + 0.8 * ml_top4_prob, 2)

        # NEW v6.1 — Top 3 : modèle dédié (~25% positifs)
        if raw_top3_ml:
            ml_top3_prob = raw_top3_ml[i] * 100
            # Sécurité : clamp à 95% max (impossible > 95% d'être top3)
            ml_top3_prob = min(ml_top3_prob, 95.0)
            a["chanceTop3ML"] = round(ml_top3_prob, 2)
        else:
            a["chanceTop3ML"] = None

        # NEW v8.1 — Ranker séquentiel (XGBRanker)
        if raw_ranker_scores:
            a["chanceRanker"] = round(raw_ranker_scores[i], 2)
        else:
            a["chanceRanker"] = None

        if a["cote"] and a["probaMarche"] > 0:
            edge = a["chance"] - a["probaMarche"]
            a["edge"] = round(edge, 2)

            # ═══ VALUE BET DETECTION ═══
            # Edge = proba_modèle - proba_marché
            # On compare NOTRE estimation (ML) vs le marché (cote)
            # Le marché a raison 95% du temps → il faut un edge SIGNIFICATIF
            p_model = a["chance"] / 100  # proba modèle (normalisée)
            p_market = a["probaMarche"] / 100  # proba marché (inverse cote)
            cote = a["cote"]

            # EV = espérance de gain (en unités)
            ev = p_model * (cote - 1) - (1 - p_model)
            a["expectedROI"] = round(ev * 100, 2)

            # Value bet : EV positif ET edge suffisant
            # - cote >= 2.5 : le marché sous-estime
            # - edge >= 3% : marge de sécurité
            # - proba_model > proba_market : notre modèle est plus optimiste
            # - Top 3 ML si dispo : confirmation
            top3_confirm = True
            if a.get("chanceTop3ML") and a["chanceTop3ML"] > 0:
                # Le modèle top3 doit aussi être optimiste
                top3_confirm = a["chanceTop3ML"] > p_market * 100 * 2  # au moins 2x le marché
            is_value = (
                ev > 0.05 and           # EV > +5%
                cote >= 2.5 and         # Pas les ultra-favoris (cote trop basse)
                cote <= 25.0 and        # Pas les outsiders improbables
                p_model > p_market and  # Notre modèle plus optimiste
                top3_confirm and        # Confirmation top3 si dispo
                edge >= 3               # Edge >= 3 points
            )
            a["valueBet"] = is_value

            # Kelly only sur les value bets confirmés
            if is_value:
                a["kellyMise"] = kelly_amount(p_model, cote, capital, kelly_mult=0.15)
                a["kellyFraction"] = round(kelly_fraction(p_model, cote, 0.15) * 100, 2)
            else:
                a["kellyMise"] = 0
                a["kellyFraction"] = 0
        else:
            a["edge"] = 0
            a["valueBet"] = False
            a["kellyMise"] = 0
            a["kellyFraction"] = 0
            a["expectedROI"] = 0

    total = sum(a["chance"] for a in analyses) or 1
    for a in analyses:
        a["chance"] = round(a["chance"] / total * 100, 2)

    # ═══ SCORE COMPOSITE POUR CLASSEMENT ═══
    # Au lieu de trier juste par "chance" (WIN ML seul),
    # on utilise les 3 modèles pour un classement plus robuste.
    #
    # La clé : TOP3 ML est le modèle le plus fiable (25% positifs)
    # Donc il doit dominer le classement.
    for a in analyses:
        win_ml = a.get("chanceML") or a.get("chance") or 0
        top3_ml = a.get("chanceTop3ML") or 0
        top4_ml = a.get("chanceTop4ML") or 0
        heur = a.get("chanceHeur") or a.get("chance") or 0
        cote = a.get("cote") or 0
        ranker_score = a.get("chanceRanker") or 0

        # Score composite v8.1 : TOP3 + WIN + Ranker + TOP4 + heur + marché
        inv_cote_score = 100 / max(cote, 1) if cote > 0 else 0

        if ranker_score > 0:
            raw_composite = (
                top3_ml * 0.30 +       # TOP3 = fiable
                win_ml * 0.15 +        # WIN = discriminant
                ranker_score * 0.25 +  # RANKER = signal séquentiel fort
                top4_ml * 0.10 +       # TOP4 = confirmation
                heur * 0.05 +          # Heuristique = filet
                inv_cote_score * 0.15   # Marché
            )
        else:
            raw_composite = (
                top3_ml * 0.40 +
                win_ml * 0.25 +
                top4_ml * 0.15 +
                heur * 0.10 +
                inv_cote_score * 0.10
            )
        a["scoreComposite"] = round(raw_composite, 2)

    # Trier par score composite (pas juste chance)
    analyses.sort(key=lambda x: -(x.get("scoreComposite") or x.get("chance") or 0))
    for rank, a in enumerate(analyses, 1):
        a["rang"] = rank

    # ═══ SCORE DE CONFIANCE ═══
    # Mesure à quel point on est sûr du #1.
    # Un score haut = #1 clairement au-dessus = plus de chance de gagner.
    if len(analyses) >= 2:
        p1 = analyses[0].get("scoreComposite") or analyses[0].get("chance") or 0
        p2 = analyses[1].get("scoreComposite") or analyses[1].get("chance") or 0

        # 1) Gap #1 vs #2
        gap_12 = p1 - p2

        # 2) Accord des modèles : est-ce que WIN, TOP3, TOP4 ont le même #1 ?
        win_rank = sorted(range(len(analyses)),
                          key=lambda i: -(analyses[i].get("chanceML") or analyses[i].get("chance") or 0))
        top3_rank = sorted(range(len(analyses)),
                           key=lambda i: -(analyses[i].get("chanceTop3ML") or 0))
        top4_rank = sorted(range(len(analyses)),
                           key=lambda i: -(analyses[i].get("chanceTop4ML") or 0))

        # Le #1 du classement est-il aussi #1 dans chaque modèle ?
        idx_top_composite = 0  # analyses[0] après tri
        win_top1 = win_rank[0] == idx_top_composite if win_rank else True
        top3_top1 = top3_rank[0] == idx_top_composite if top3_rank else True
        top4_top1 = top4_rank[0] == idx_top_composite if top4_rank else True

        models_agree = sum([win_top1, top3_top1, top4_top1])
        # 3/3 = accord total, 2/3 = bon, 1/3 = faible

        # 3) Score composite
        confiance = 0
        confiance += min(gap_12, 30) / 30 * 30   # gap : max 30 points
        confiance += models_agree / 3 * 40         # accord modèles : max 40 points
        confiance += min(p1, 50) / 50 * 20         # force du #1 : max 20 points
        # Marché d'accord ? Le favori de la cote est-il notre #1 ?
        fav_marche = max(analyses, key=lambda a: a.get("probaMarche") or 0)
        if fav_marche.get("nom") == analyses[0].get("nom"):
            confiance += 10  # marché confirme
        confiance = min(round(confiance), 100)

        analyses[0]["confiance"] = confiance
        analyses[0]["gap12"] = round(gap_12, 1)
        analyses[0]["modelsAgree"] = f"{models_agree}/3"
        analyses[0]["winTop1"] = win_top1
        analyses[0]["top3Top1"] = top3_top1
        analyses[0]["top4Top1"] = top4_top1

        # Niveau de confiance pour l'UI
        if confiance >= 70:
            analyses[0]["confianceNiveau"] = "🔥 HAUTE"
        elif confiance >= 50:
            analyses[0]["confianceNiveau"] = "✅ BONNE"
        elif confiance >= 35:
            analyses[0]["confianceNiveau"] = "⚠️ MOYENNE"
        else:
            analyses[0]["confianceNiveau"] = "❌ FAIBLE"

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
        "top4_brier_sum": 0.0,  # Somme des (p - y)^2 pour Brier score
        # NEW v6.1 : Top 3 ML tracking
        "top3_ml_hit": 0, "top3_ml_total": 0,
        "top3_brier_sum": 0.0,
        "super_base_hit": 0, "super_base_total": 0,  # Super Base = meilleur top3 ML dans top 5
        # NEW v8 : Confiance tracking
        "confiance_high": {"win": 0, "total": 0, "gain": 0.0},
        "confiance_bonne": {"win": 0, "total": 0, "gain": 0.0},
        "confiance_moyenne": {"win": 0, "total": 0, "gain": 0.0},
        "confiance_faible": {"win": 0, "total": 0, "gain": 0.0},
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

        # NEW v8 — Confiance tracking
        confiance_niveau = (top1.get("confianceNiveau") or "").lower()
        confiance_key = None
        if "haute" in confiance_niveau: confiance_key = "confiance_high"
        elif "bonne" in confiance_niveau: confiance_key = "confiance_bonne"
        elif "moyenne" in confiance_niveau: confiance_key = "confiance_moyenne"
        elif "faible" in confiance_niveau: confiance_key = "confiance_faible"
        if confiance_key:
            results[confiance_key]["total"] += 1
            if top1["ordreArrivee"] == 1:
                results[confiance_key]["win"] += 1
                if top1["cote"]:
                    results[confiance_key]["gain"] += top1["cote"]

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
                # Brier: (p - y)^2
                results["top4_brier_sum"] += (ml_top4_prob - (1.0 if actual_top4 else 0.0)) ** 2
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

        # NEW v6.1 — Top 3 ML tracking
        for a in analyses:
            ml_top3_prob = a.get("chanceTop3ML")
            actual_place = a.get("ordreArrivee", 0) or 0
            if ml_top3_prob and ml_top3_prob > 0:
                actual_top3 = 1 <= actual_place <= 3
                results["top3_ml_total"] += 1
                if actual_top3:
                    results["top3_ml_hit"] += 1
                results["top3_brier_sum"] += (ml_top3_prob - (1.0 if actual_top3 else 0.0)) ** 2

        # Super Base : meilleur chanceTop3ML parmi les 5 premiers → est-il dans le top 3 ?
        if len(analyses) >= 1:
            top5 = analyses[:5]
            with_top3 = [a for a in top5 if a.get("chanceTop3ML")]
            if with_top3:
                super_base = max(with_top3, key=lambda a: a.get("chanceTop3ML", 0))
                sb_place = super_base.get("ordreArrivee", 0) or 0
                results["super_base_total"] += 1
                if 1 <= sb_place <= 3:
                    results["super_base_hit"] += 1

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

    # NEW v8 — Stats par niveau de confiance
    confiance_stats = {}
    for key, label in [("confiance_high", "🔥 Haute (≥70)"),
                       ("confiance_bonne", "✅ Bonne (50-69)"),
                       ("confiance_moyenne", "⚠️ Moyenne (35-49)"),
                       ("confiance_faible", "❌ Faible (<35)")]:
        s = results.get(key, {"win": 0, "total": 0, "gain": 0.0})
        if s["total"] > 0:
            confiance_stats[key] = {
                "label": label,
                "total": s["total"],
                "win": s["win"],
                "taux": round(s["win"] / s["total"] * 100, 1),
                "roi": round((s["gain"] - s["total"]) / s["total"] * 100, 1),
            }
    results["confiance_stats"] = confiance_stats

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

    # ═══ DIAGNOSTIC RENTABILITÉ ═══
    # Top1 par cote range
    top1_by_cote = {"petit": {"win": 0, "total": 0},  # cote < 3
                    "moyen": {"win": 0, "total": 0},   # 3 <= cote < 8
                    "gros": {"win": 0, "total": 0}}    # cote >= 8
    # ROI par stratégie
    roi_strategies = {
        "top1_systematique": {"mise": 0, "gain": 0},
        "top1_cote_moyenne": {"mise": 0, "gain": 0},   # top1 seulement si cote >= 3
        "top1_value_bet": {"mise": 0, "gain": 0},       # top1 si valueBet
        "top3_placé": {"mise": 0, "gain": 0},           # parier le #1 placé (top3)
    }

    # On doit refaire une passe sur les analyses déjà traitées
    # → On ne les a plus. Mais on peut calculer depuis results existant.
    # Pour le diagnostic, on recalcule depuis les value_bets
    for vb in vb:
        if vb["gagne"]:
            roi_strategies["top1_value_bet"]["gain"] += vb["cote"]
        roi_strategies["top1_value_bet"]["mise"] += 1

    # Statégie "top3 placé" : le #1 est top3 dans 71.9% des cas
    # En pariant placé (cote placé ≈ cote_gagnant / 3 en moyenne)
    # On estime la cote placée à cote/2.5 (approximation PMU)
    results["diagnostic"] = {
        "taux_top1": results["taux_top1"],
        "taux_top3_du_top1": results["taux_top1_place"],
        "cote_moyenne_necessaire": round(100 / max(results["taux_top1"], 1) * 1.25, 1),
        "conseil": "PASSER EN MODE PLACÉ (TOP3)" if results["taux_top1"] < 45 else "OK GAGNANT",
        "roi_strategies": {},
    }
    for name, s in roi_strategies.items():
        if s["mise"] > 0:
            results["diagnostic"]["roi_strategies"][name] = round(
                (s["gain"] - s["mise"]) / s["mise"] * 100, 2)

    # NEW v6 — Top 4 ML stats
    n_top4 = results["top4_ml_total"]
    if n_top4 > 0:
        results["top4_ml_accuracy"] = round(results["top4_ml_hit"] / n_top4 * 100, 2)
        results["top4_ml_brier"] = round(results["top4_brier_sum"] / n_top4, 4)
    else:
        results["top4_ml_accuracy"] = None
        results["top4_ml_brier"] = None

    n_top3 = results["top3_ml_total"]
    if n_top3 > 0:
        results["top3_ml_accuracy"] = round(results["top3_ml_hit"] / n_top3 * 100, 2)
        results["top3_ml_brier"] = round(results["top3_brier_sum"] / n_top3, 4)
    else:
        results["top3_ml_accuracy"] = None
        results["top3_ml_brier"] = None

    if results["super_base_total"] > 0:
        results["super_base_accuracy"] = round(results["super_base_hit"] / results["super_base_total"] * 100, 2)
    else:
        results["super_base_accuracy"] = None
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
            "top3_total": 0, "top4_count": 0,
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
            if 1 <= place <= 4:
                day["top4_count"] += 1

            day["courses"].append(course_detail)

        if day["total"] > 0:
            day["taux_top1"] = round(day["top1"] / day["total"] * 100, 1)
            day["taux_top3"] = round(day["top3_total"] / day["total"] * 100, 1)
            day["taux_top4"] = round(day["top4_count"] / day["total"] * 100, 1)
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


@app.route("/force")
@admin_required
def force_page():
    return render_template("force.html")


@app.route("/edge")
@admin_required
def edge_page():
    return render_template("edge.html")


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


@app.route("/api/course-pdf/<int:r_num>/<int:c_num>")
@admin_required
def api_course_pdf(r_num, c_num):
    """Génère un PDF de synthèse pour une course.
    - GAGNANT : chevaux classés #1 et #2
    - SUPER BASE : meilleur % Top4 parmi les 5 premiers
    """
    from fpdf import FPDF
    from flask import send_file
    import io as _io

    date_str = request.args.get("date") or fmt_date(datetime.now())
    use_ml = request.args.get("ml") == "1"
    capital = float(request.args.get("capital", 100))

    try:
        prog = get_programme(date_str)
        parts = get_participants(date_str, r_num, c_num)
        perfs = get_performances(date_str, r_num, c_num)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    course_info = None
    reunion_info = None
    discipline = None
    hippodrome = None
    type_corde = None
    terrain = None
    heure = ""
    for r in prog["programme"]["reunions"]:
        if r["numOfficiel"] == r_num:
            hippodrome = r["hippodrome"]["libelleCourt"]
            reunion_info = {"hippodrome": hippodrome}
            for c in r["courses"]:
                if c["numOrdre"] == c_num:
                    discipline = c.get("discipline")
                    type_corde = c.get("corde", "")
                    terrain = c.get("natureTerrain") or c.get("terrain")
                    heure = datetime.fromtimestamp(c["heureDepart"] / 1000).strftime("%H:%M") if c.get("heureDepart") else ""
                    course_info = {
                        "libelle": c.get("libelle"),
                        "discipline": discipline,
                        "distance": c.get("distance"),
                        "specialite": c.get("specialite"),
                        "corde": type_corde,
                        "terrain": terrain,
                        "heure": heure,
                        "montantPrix": c.get("montantPrix"),
                        "nbPartants": c.get("nombreDeclaresPartants"),
                    }

    team_stats, horse_stats, elo, elo_hist, horse_races, pedigree = compute_all_stats(
        max_days=HISTORY_DAYS)
    analyses = analyser_course(parts, perfs,
                                course_info.get("distance") if course_info else None,
                                discipline, hippodrome, type_corde,
                                team_stats, horse_stats, elo, elo_hist,
                                horse_races, pedigree, use_ml=use_ml,
                                capital=capital, terrain=terrain)

    if not analyses:
        return jsonify({"error": "Aucune analyse"}), 404

    # GAGNANT : top 2
    gagnant = analyses[:2]
    # SUPER BASE : meilleur Top3 ML parmi les 5 premiers (fallback Top4 ML)
    top5 = analyses[:5]
    # Priorité au modèle TOP3 dédié, sinon fallback sur Top4
    def _super_base_score(a):
        if a.get("chanceTop3ML"):
            return a["chanceTop3ML"]
        return a.get("chanceTop4", 0) / 100  # fallback
    best_top4 = max(top5, key=_super_base_score)

    # Date lisible
    try:
        date_lisible = f"{date_str[0:2]}/{date_str[2:4]}/{date_str[4:8]}"
    except:
        date_lisible = date_str

    # ============ PDF ============
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    # En-tête vert foncé
    pdf.set_fill_color(15, 23, 42)
    pdf.rect(0, 0, 210, 35, 'F')
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(16, 185, 129)
    pdf.set_xy(10, 6)
    pdf.cell(0, 10, "TURF ANALYZER v6", ln=True)

    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(148, 163, 184)
    pdf.set_x(10)
    libelle = (course_info or {}).get("libelle", "") or ""
    dist = (course_info or {}).get("distance", "") or ""
    disc = (course_info or {}).get("discipline", "") or ""
    nb = (course_info or {}).get("nbPartants", "") or ""
    prix = (course_info or {}).get("montantPrix")
    prix_str = f" - {prix:,} EUR" if prix else ""
    pdf.cell(0, 5, f"{date_lisible} | {hippodrome or ''} | R{r_num}C{c_num} | {heure}", ln=True)
    pdf.set_x(10)
    pdf.cell(0, 5, f"{libelle} | {disc} {dist}m | {nb} partants{prix_str}", ln=True)
    if use_ml:
        pdf.set_x(10)
        pdf.set_text_color(139, 92, 246)
        pdf.cell(0, 5, "Mode: ML active (20% heuristique + 80% ML)", ln=True)

    # --- Section GAGNANT ---
    pdf.ln(10)
    pdf.set_fill_color(16, 185, 129)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "  GAGNANT — Top 2", fill=True, ln=True)

    for a in gagnant:
        pdf.ln(3)
        y_start = pdf.get_y()
        pdf.set_fill_color(30, 41, 59)
        pdf.rect(10, y_start, 190, 22, 'F')

        # Numéro + Nom
        pdf.set_font("Helvetica", "B", 22)
        pdf.set_text_color(16, 185, 129)
        pdf.set_xy(14, y_start + 2)
        pdf.cell(14, 14, f"#{a['rang']}")

        pdf.set_font("Helvetica", "B", 16)
        pdf.set_text_color(226, 232, 240)
        pdf.cell(75, 8, str(a["nom"])[:25])

        # Chance
        pdf.set_font("Helvetica", "B", 20)
        pdf.set_text_color(16, 185, 129)
        pdf.cell(30, 10, f"{a['chance']}%")

        # Cote
        if a.get("cote"):
            pdf.set_font("Helvetica", "B", 11)
            pdf.set_text_color(30, 64, 175)
            pdf.cell(25, 10, f"Cote {a['cote']}")

        # Edge
        if a.get("edge", 0) != 0:
            pdf.set_font("Helvetica", "B", 10)
            if a["edge"] > 0:
                pdf.set_text_color(16, 185, 129)
            else:
                pdf.set_text_color(239, 68, 68)
            pdf.cell(0, 10, f"Edge {'+' if a['edge']>0 else ''}{a['edge']}%")

        # Ligne info
        pdf.set_xy(14, y_start + 12)
        pdf.set_font("Helvetica", "", 9)
        pdf.set_text_color(148, 163, 184)
        age = a.get("age", "")
        sexe = a.get("sexe", "")
        driver = a.get("driver", "")
        entraineur = a.get("entraineur", "")
        nbC = a.get("nbCourses", 0)
        nbV = a.get("nbVictoires", 0)
        nbP = a.get("nbPlaces", 0)
        gains = a.get("gainsCarriere", 0)
        pdf.cell(0, 5, f"{age}ans {sexe} | Dr: {driver} | Ent: {entraineur} | {nbC}c {nbV}V {nbP}P | {gains:,}EUR")

        # Musique
        if a.get("musique"):
            pdf.set_x(14)
            pdf.set_font("Courier", "", 8)
            pdf.set_text_color(245, 158, 11)
            pdf.cell(0, 4, str(a["musique"])[:50])

        pdf.set_y(y_start + 24)

    # --- Section SUPER BASE ---
    pdf.ln(6)
    pdf.set_fill_color(6, 182, 212)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", "B", 14)
    pdf.cell(0, 10, "  SUPER BASE — Meilleur Top 4 (top 5)", fill=True, ln=True)

    a = best_top4
    pdf.ln(3)
    y_start = pdf.get_y()
    pdf.set_fill_color(30, 41, 59)
    pdf.rect(10, y_start, 190, 30, 'F')

    pdf.set_font("Helvetica", "B", 22)
    pdf.set_text_color(6, 182, 212)
    pdf.set_xy(14, y_start + 2)
    pdf.cell(14, 14, f"#{a['rang']}")

    pdf.set_font("Helvetica", "B", 16)
    pdf.set_text_color(226, 232, 240)
    pdf.cell(65, 8, str(a["nom"])[:25])

    # Chance Top3 ML (ou fallback Top4)
    t3ml = a.get("chanceTop3ML")
    t4 = a.get("chanceTop4", 0)
    score_display = round(t3ml) if t3ml else round(t4)
    label_display = "Top3 ML" if t3ml else "Top4"
    pdf.set_font("Helvetica", "B", 20)
    pdf.set_text_color(6, 182, 212)
    pdf.cell(30, 10, f"{score_display}%")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(148, 163, 184)
    pdf.cell(20, 10, label_display)

    # Chance Gagnant
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_text_color(16, 185, 129)
    pdf.cell(0, 10, f"Gagnant {a['chance']}%")

    # Info
    pdf.set_xy(14, y_start + 12)
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(148, 163, 184)
    driver = a.get("driver", "")
    entraineur = a.get("entraineur", "")
    pdf.cell(0, 5, f"Dr: {driver} | Ent: {entraineur} | {a.get('nbCourses',0)}c {a.get('nbVictoires',0)}V {a.get('nbPlaces',0)}P | {a.get('gainsCarriere',0):,}EUR")

    # Musique
    if a.get("musique"):
        pdf.set_x(14)
        pdf.set_font("Courier", "", 8)
        pdf.set_text_color(245, 158, 11)
        pdf.cell(0, 4, str(a["musique"])[:50])

    # Top4 ML detail
    if a.get("chanceTop4ML"):
        pdf.set_x(14)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(139, 92, 246)
        pdf.cell(0, 4, f"Top4 ML: {round(a['chanceTop4ML'])}% | Top4 Heur: {round(a.get('chanceTop4Heur',0))}%")

    # Kelly
    if a.get("kellyMise", 0) > 0:
        pdf.set_x(14)
        pdf.set_font("Helvetica", "B", 9)
        pdf.set_text_color(167, 139, 250)
        pdf.cell(0, 4, f"Kelly: {a['kellyMise']}EUR | EV: +{a.get('expectedROI',0)}%")

    # Cote
    if a.get("cote"):
        pdf.set_x(14)
        pdf.set_font("Helvetica", "", 8)
        pdf.set_text_color(30, 64, 175)
        pdf.cell(0, 4, f"Cote: {a['cote']}")

    # --- Footer ---
    pdf.set_y(y_start + 36)
    pdf.ln(6)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(71, 85, 105)
    pdf.cell(0, 5, f"Turf Analyzer v6 — Genere le {datetime.now().strftime('%d/%m/%Y %H:%M')} — 41 features, Dual Win/Top4, Stacking ML", align="C")
    pdf.ln(4)
    pdf.cell(0, 5, "Le jeu comporte des risques : joueurs-info-service.fr", align="C")

    # Output
    output = _io.BytesIO()
    pdf.output(output)
    output.seek(0)

    filename = f"turf_{date_lisible.replace('/','')}_R{r_num}C{c_num}.pdf"
    return send_file(output, mimetype='application/pdf', as_attachment=True,
                     download_name=filename)


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
                  "top3_total": 0, "top4_count": 0}
        for d in data:
            for k in totals:
                totals[k] += d.get(k, 0)
        n = totals["total"] or 1
        totals["taux_top1"] = round(totals["top1"] / n * 100, 1)
        totals["taux_top3"] = round(totals["top3_total"] / n * 100, 1)
        totals["taux_top4"] = round(totals["top4_count"] / n * 100, 1)
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


@app.route("/api/train-reset", methods=["POST"])
@admin_required
def api_train_reset():
    """Force le déblocage du training (en cas de lock mort)."""
    current = _load_ml_status()
    if current and current.get("status") == "training":
        started = current.get("started_at", "?")
        _save_ml_status({
            "status": "reset",
            "finished_at": datetime.now().isoformat(),
            "source": "manual_reset",
            "error": f"Reset forcé (était bloqué depuis {started})",
        })
        return jsonify({"ok": True, "message": "Lock de training supprimé"})
    return jsonify({"ok": True, "message": "Aucun training en cours"})


@app.route("/api/train", methods=["POST"])
@admin_required
def api_train():
    days = int(request.args.get("days", 21))
    days = min(days, 30)
    model_type = request.args.get("type", "advanced")

    # Vérifier si un training est déjà en cours
    current_status = _load_ml_status()
    if current_status and current_status.get("status") == "training":
        # Auto-reset si le training est bloqué depuis + de 30 min
        started = current_status.get("started_at", "")
        if started:
            try:
                from datetime import datetime as _dt
                start_time = _dt.fromisoformat(started)
                elapsed_min = (_dt.now() - start_time).total_seconds() / 60
                if elapsed_min > 30:
                    print(f"[Train] ⚠️ Training bloqué depuis {elapsed_min:.0f}min, auto-reset")
                    _save_ml_status({
                        "status": "timeout",
                        "finished_at": datetime.now().isoformat(),
                        "source": "manual",
                        "error": f"Auto-reset après {elapsed_min:.0f} min (probable crash)",
                    })
                else:
                    return jsonify({"error": "Un entraînement est déjà en cours"}), 409
            except Exception:
                pass
        else:
            return jsonify({"error": "Un entraînement est déjà en cours"}), 409

    # Lancer le training en arrière-plan (async)
    def _train_bg():
        now_str = datetime.now().strftime("%d/%m/%Y %H:%M")
        print(f"[Manual Train] Démarrage — {days} jours, {model_type} — {now_str}")
        _save_ml_status({
            "status": "training",
            "started_at": datetime.now().isoformat(),
            "source": "manual",
            "params": {"days_back": days, "model_type": model_type},
        })
        try:
            info = train_ml_model(days_back=days, model_type=model_type)
            if info:
                _save_ml_status({
                    "status": "ok",
                    "finished_at": datetime.now().isoformat(),
                    "source": "manual",
                    "params": {"days_back": days, "model_type": model_type},
                    "result": {k: str(v) for k, v in info.items()},
                })
                print(f"[Manual Train] ✅ Terminé — {info}")
            else:
                _save_ml_status({
                    "status": "error",
                    "finished_at": datetime.now().isoformat(),
                    "source": "manual",
                    "error": "Pas assez de données (< 100 samples)",
                })
                print("[Manual Train] ❌ Pas assez de données")
        except Exception as e:
            _save_ml_status({
                "status": "error",
                "finished_at": datetime.now().isoformat(),
                "source": "manual",
                "error": str(e),
            })
            print(f"[Manual Train] ❌ Erreur: {e}")

    import threading
    t = threading.Thread(target=_train_bg, daemon=True)
    t.start()

    return jsonify({"ok": True, "message": "Training démarré en arrière-plan", "status_url": "/api/ml-status"})


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
        "top3": load_ml_model_top3() is not None,
        "top4": load_ml_model_top4() is not None,
    }
    return jsonify(status)


if __name__ == "__main__":
    start_auto_train_scheduler()
    app.run(host="0.0.0.0", port=5000, debug=False)
