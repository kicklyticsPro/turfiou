"""
Turf Analyzer — Version épurée

Analyse des chevaux basée sur 3 piliers uniquement :
  1. Stats carrière du cheval
  2. Stats driver/jockey
  3. Stats entraîneur

Classement = moyenne simple des 3 scores.
Backtest = performance historique basé sur ce classement.

Les formules de scoring sont IDENTIQUES à l'ancien projet v6.
"""

from flask import Flask, jsonify, render_template, request, session, redirect, url_for
from functools import wraps
from datetime import datetime, timedelta
import requests
import math
import os
import pickle
import json
import threading
from functools import lru_cache
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "turf-analyzer-simple")
ADMIN_PASSWORD = os.environ.get("ADMIN_PASSWORD", "admin123")

# ── Config ──
PMU_BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/61/programme"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TurfAnalyzer/1.0)"}
CACHE_DIR = os.environ.get("CACHE_DIR", "/tmp/turf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

STATS_CACHE_FILE = os.path.join(CACHE_DIR, "stats_v6.pkl")
HISTORY_DAYS = 15
WINDOW_SHORT = 30

# ── Global state ──
_stats_ready = threading.Event()
_current_stats = None
_stats_progress = {"status": "init", "message": "Initialisation...", "progress": 0}
_progress_lock = threading.Lock()


def _norm(name):
    """Normalise un nom : uppercase, strip, collapse spaces."""
    if not name:
        return ""
    return " ".join(str(name).upper().split())

def fmt_date(d):
    return d.strftime("%d%m%Y")

def _safe(val, default=0):
    if val is None:
        return default
    try:
        if math.isnan(val) or math.isinf(val):
            return default
    except (TypeError, ValueError):
        pass
    return val

def _update_progress(done, total, msg=None):
    """Thread-safe update de la progression."""
    with _progress_lock:
        pct = round(done / max(total, 1) * 100)
        _stats_progress["status"] = "loading"
        _stats_progress["progress"] = pct
        _stats_progress["done"] = done
        _stats_progress["total"] = total
        _stats_progress["message"] = msg or f"{done}/{total} courses ({pct}%)"

def _freeze(d):
    """Convertit defaultdict en dict normal pour sérialisation."""
    if isinstance(d, defaultdict):
        return {k: _freeze(v) for k, v in d.items()}
    return d

def _empty_bucket():
    return {"c": 0, "v": 0, "p": 0}


# ═══════════════════════════════════════════════════════════
#  PMU API
# ═══════════════════════════════════════════════════════════

@lru_cache(maxsize=256)
def get_programme(date_str):
    r = requests.get(f"{PMU_BASE}/{date_str}", headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()

def get_participants_cached(date_str, r_num, c_num):
    url = f"{PMU_BASE}/{date_str}/R{r_num}/C{c_num}/participants"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()

@lru_cache(maxsize=2048)
def get_participants(date_str, r_num, c_num):
    return get_participants_cached(date_str, r_num, c_num)

def get_participants_live(date_str, r_num, c_num):
    url = f"{PMU_BASE}/{date_str}/R{r_num}/C{c_num}/participants"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()

@lru_cache(maxsize=2048)
def get_performances(date_str, r_num, c_num):
    url = f"{PMU_BASE}/{date_str}/R{r_num}/C{c_num}/performances-detaillees/pretty"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"participants": []}


# ═══════════════════════════════════════════════════════════
#  Cache persistence
# ═══════════════════════════════════════════════════════════

def load_pickle(path, max_age_hours=48):
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


# ═══════════════════════════════════════════════════════════
#  Build tasks list
# ═══════════════════════════════════════════════════════════

def _build_tasks(max_days, exclude_recent_days):
    """Construit la liste des tâches (date, r_num, c_num, discipline, hippo, delta)."""
    today = datetime.now()
    start_delta = exclude_recent_days + 1 if exclude_recent_days > 0 else 1
    tasks = []
    for delta in range(start_delta, max_days + 1):
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
                                  c.get("discipline", ""), hippo, delta))
    return tasks


# ═══════════════════════════════════════════════════════════
#  Background stats loading — PARALLÈLE avec progression
# ═══════════════════════════════════════════════════════════

def _process_task(task, ts_drivers, ts_drivers_short, ts_drivers_disc, ts_drivers_hippo,
                  ts_entraineurs, ts_entraineurs_short, ts_entraineurs_disc,
                  hs_global, hs_with_driver, hs_hippo, hs_disc):
    """Traite une seule course et met à jour les stats."""
    date_str, r_num, c_num, discipline, hippo, delta = task
    is_short = delta <= WINDOW_SHORT

    try:
        parts = get_participants_cached(date_str, r_num, c_num)
    except Exception:
        return

    for p in parts.get("participants", []):
        if p.get("statut") != "PARTANT":
            continue
        cheval = _norm(p.get("nom"))
        driver = _norm(p.get("driver"))
        entraineur = _norm(p.get("entraineur"))
        place = p.get("ordreArrivee", 0) or 0
        won = 1 if place == 1 else 0
        placed = 1 if 1 <= place <= 3 else 0

        # Stats cheval
        if cheval:
            hs_global[cheval]["c"] += 1
            hs_global[cheval]["v"] += won
            hs_global[cheval]["p"] += placed
            if driver:
                hs_with_driver[cheval][driver]["c"] += 1
                hs_with_driver[cheval][driver]["v"] += won
                hs_with_driver[cheval][driver]["p"] += placed
            if hippo:
                hs_hippo[cheval][hippo]["c"] += 1
                hs_hippo[cheval][hippo]["v"] += won
                hs_hippo[cheval][hippo]["p"] += placed
            if discipline:
                hs_disc[cheval][discipline]["c"] += 1
                hs_disc[cheval][discipline]["v"] += won
                hs_disc[cheval][discipline]["p"] += placed

        # Stats driver
        if driver:
            ts_drivers[driver]["c"] += 1
            ts_drivers[driver]["v"] += won
            ts_drivers[driver]["p"] += placed
            if is_short:
                ts_drivers_short[driver]["c"] += 1
                ts_drivers_short[driver]["v"] += won
                ts_drivers_short[driver]["p"] += placed
            if discipline:
                ts_drivers_disc[driver][discipline]["c"] += 1
                ts_drivers_disc[driver][discipline]["v"] += won
                ts_drivers_disc[driver][discipline]["p"] += placed
            if hippo:
                ts_drivers_hippo[driver][hippo]["c"] += 1
                ts_drivers_hippo[driver][hippo]["v"] += won
                ts_drivers_hippo[driver][hippo]["p"] += placed

        # Stats entraîneur
        if entraineur:
            ts_entraineurs[entraineur]["c"] += 1
            ts_entraineurs[entraineur]["v"] += won
            ts_entraineurs[entraineur]["p"] += placed
            if is_short:
                ts_entraineurs_short[entraineur]["c"] += 1
                ts_entraineurs_short[entraineur]["v"] += won
                ts_entraineurs_short[entraineur]["p"] += placed
            if discipline:
                ts_entraineurs_disc[entraineur][discipline]["c"] += 1
                ts_entraineurs_disc[entraineur][discipline]["v"] += won
                ts_entraineurs_disc[entraineur][discipline]["p"] += placed

def _background_load_stats():
    """Charge les stats en arrière-plan avec progression visible."""
    global _current_stats, _stats_ready, _stats_progress

    # Vérifier le cache d'abord
    cached = load_pickle(STATS_CACHE_FILE)
    if cached:
        _current_stats = cached
        _stats_ready.set()
        _stats_progress = {"status": "ready", "message": "Chargé depuis le cache", "progress": 100,
                           "horses": len(cached.get("horse_stats", {}).get("global", {})),
                           "drivers": len(cached.get("team_stats", {}).get("drivers", {})),
                           "trainers": len(cached.get("team_stats", {}).get("entraineurs", {}))}
        print(f"[Stats] ✅ Cache chargé")
        return

    # Pas de cache → calculer en parallèle
    _stats_progress = {"status": "loading", "message": "Dénombrement des courses...", "progress": 0}

    try:
        # Étape 1 : compter les tâches
        all_tasks = _build_tasks(HISTORY_DAYS, 0)
        total = len(all_tasks)
        _stats_progress = {"status": "loading", "message": f"{total} courses à analyser",
                           "progress": 0, "total": total, "done": 0}

        if total == 0:
            _stats_progress = {"status": "error", "message": "Aucune course trouvée", "progress": 0}
            _stats_ready.set()
            return

        # Étape 2 : init les compteurs de stats
        ts_drivers = defaultdict(_empty_bucket)
        ts_drivers_short = defaultdict(_empty_bucket)
        ts_drivers_disc = defaultdict(lambda: defaultdict(_empty_bucket))
        ts_drivers_hippo = defaultdict(lambda: defaultdict(_empty_bucket))
        ts_entraineurs = defaultdict(_empty_bucket)
        ts_entraineurs_short = defaultdict(_empty_bucket)
        ts_entraineurs_disc = defaultdict(lambda: defaultdict(_empty_bucket))

        hs_global = defaultdict(_empty_bucket)
        hs_with_driver = defaultdict(lambda: defaultdict(_empty_bucket))
        hs_hippo = defaultdict(lambda: defaultdict(_empty_bucket))
        hs_disc = defaultdict(lambda: defaultdict(_empty_bucket))

        # Étape 3 : fetch + traitement en parallèle avec compteur
        done = [0]
        done_lock = threading.Lock()

        def track_and_process(task):
            _process_task(task, ts_drivers, ts_drivers_short, ts_drivers_disc, ts_drivers_hippo,
                         ts_entraineurs, ts_entraineurs_short, ts_entraineurs_disc,
                         hs_global, hs_with_driver, hs_hippo, hs_disc)
            with done_lock:
                done[0] += 1
                if done[0] % 10 == 0:
                    _update_progress(done[0], total)

        print(f"[Stats] 🚀 Calcul en parallèle ({total} courses, 15 workers)...")
        with ThreadPoolExecutor(max_workers=15) as ex:
            list(ex.map(track_and_process, all_tasks))

        # Étape 4 : finaliser
        team_stats = {
            "drivers": _freeze(ts_drivers),
            "drivers_short": _freeze(ts_drivers_short),
            "drivers_disc": _freeze(ts_drivers_disc),
            "drivers_hippo": _freeze(ts_drivers_hippo),
            "entraineurs": _freeze(ts_entraineurs),
            "entraineurs_short": _freeze(ts_entraineurs_short),
            "entraineurs_disc": _freeze(ts_entraineurs_disc),
        }
        horse_stats = {
            "global": _freeze(hs_global),
            "with_driver": _freeze(hs_with_driver),
            "hippo": _freeze(hs_hippo),
            "disc": _freeze(hs_disc),
        }

        _current_stats = {"team_stats": team_stats, "horse_stats": horse_stats}
        save_pickle(STATS_CACHE_FILE, _current_stats)
        _stats_ready.set()
        _stats_progress = {"status": "ready", "message": "Prêt", "progress": 100,
                           "horses": len(hs_global), "drivers": len(ts_drivers),
                           "trainers": len(ts_entraineurs)}
        print(f"[Stats] ✅ Terminé : {len(hs_global)} chevaux, {len(ts_drivers)} drivers, "
              f"{len(ts_entraineurs)} entraîneurs")

    except Exception as e:
        import traceback
        traceback.print_exc()
        _stats_ready.set()
        _stats_progress = {"status": "error", "message": str(e), "progress": 0}


# ═══════════════════════════════════════════════════════════
#  Stats accessors
# ═══════════════════════════════════════════════════════════

def get_stats():
    if _stats_ready.is_set():
        return _current_stats
    return None

def compute_all_stats(max_days=HISTORY_DAYS, exclude_recent_days=0):
    """Version synchrone pour backtest/bilan (utilise le cache si dispo)."""
    if exclude_recent_days == 0:
        cached = load_pickle(STATS_CACHE_FILE)
        if cached:
            return cached

    # Calculer de manière synchrone (pour backtest avec exclude)
    all_tasks = _build_tasks(max_days, exclude_recent_days)

    ts_drivers = defaultdict(_empty_bucket)
    ts_drivers_short = defaultdict(_empty_bucket)
    ts_drivers_disc = defaultdict(lambda: defaultdict(_empty_bucket))
    ts_drivers_hippo = defaultdict(lambda: defaultdict(_empty_bucket))
    ts_entraineurs = defaultdict(_empty_bucket)
    ts_entraineurs_short = defaultdict(_empty_bucket)
    ts_entraineurs_disc = defaultdict(lambda: defaultdict(_empty_bucket))
    hs_global = defaultdict(_empty_bucket)
    hs_with_driver = defaultdict(lambda: defaultdict(_empty_bucket))
    hs_hippo = defaultdict(lambda: defaultdict(_empty_bucket))
    hs_disc = defaultdict(lambda: defaultdict(_empty_bucket))

    for task in all_tasks:
        _process_task(task, ts_drivers, ts_drivers_short, ts_drivers_disc, ts_drivers_hippo,
                     ts_entraineurs, ts_entraineurs_short, ts_entraineurs_disc,
                     hs_global, hs_with_driver, hs_hippo, hs_disc)

    out = {
        "team_stats": {
            "drivers": _freeze(ts_drivers),
            "drivers_short": _freeze(ts_drivers_short),
            "drivers_disc": _freeze(ts_drivers_disc),
            "drivers_hippo": _freeze(ts_drivers_hippo),
            "entraineurs": _freeze(ts_entraineurs),
            "entraineurs_short": _freeze(ts_entraineurs_short),
            "entraineurs_disc": _freeze(ts_entraineurs_disc),
        },
        "horse_stats": {
            "global": _freeze(hs_global),
            "with_driver": _freeze(hs_with_driver),
            "hippo": _freeze(hs_hippo),
            "disc": _freeze(hs_disc),
        },
    }

    if exclude_recent_days == 0:
        save_pickle(STATS_CACHE_FILE, out)

    return out

def start_stats_loader():
    t = threading.Thread(target=_background_load_stats, daemon=True)
    t.start()
    print("[Stats] 🔄 Chargement en arrière-plan...")


# ═══════════════════════════════════════════════════════════
#  Scoring — IDENTIQUE à l'ancien projet v6
# ═══════════════════════════════════════════════════════════

def get_bucket_score(bucket, max_score=100, min_courses=5):
    """Score 0-100. IDENTIQUE à l'ancien get_bucket_score."""
    if not bucket or bucket["c"] < min_courses:
        return None
    c, v, p = bucket["c"], bucket["v"], bucket["p"]
    tv, tp = v / c, p / c
    confiance = min(1.0, c / 30)
    raw = tv * 200 + tp * 60
    return min(max_score, raw * confiance + 30 * (1 - confiance))

def get_team_score_multi(name, kind, team_stats, discipline=None, hippodrome=None):
    """Score driver ou entraîneur avec pondération multi-contexte.
    IDENTIQUE à l'ancien get_team_score_multi."""
    if not team_stats or not name:
        return 50
    name = _norm(name)

    if kind == "drivers":
        gb = team_stats.get("drivers", {}).get(name)
        sb = team_stats.get("drivers_short", {}).get(name)
        db = team_stats.get("drivers_disc", {}).get(name, {}).get(discipline) if discipline else None
        hb = team_stats.get("drivers_hippo", {}).get(name, {}).get(hippodrome) if hippodrome else None
    else:
        gb = team_stats.get("entraineurs", {}).get(name)
        sb = team_stats.get("entraineurs_short", {}).get(name)
        db = team_stats.get("entraineurs_disc", {}).get(name, {}).get(discipline) if discipline else None
        hb = None

    s_g = get_bucket_score(gb) or 50
    s_s = get_bucket_score(sb, min_courses=3)
    s_d = get_bucket_score(db, min_courses=3)
    s_h = get_bucket_score(hb, min_courses=3)

    parts = [(s_g, 0.35)]
    if s_s is not None:
        parts.append((s_s, 0.30))
    if s_d is not None:
        parts.append((s_d, 0.20))
    if s_h is not None:
        parts.append((s_h, 0.15))

    tw = sum(w for _, w in parts)
    return sum(s * w for s, w in parts) / tw

def get_horse_score(cheval, driver, hippodrome, discipline, horse_stats):
    """Score cheval avec pondération multi-contexte.
    IDENTIQUE à l'ancien get_horse_score."""
    if not horse_stats or not cheval:
        return 50
    cheval = _norm(cheval)

    s_g = get_bucket_score(horse_stats.get("global", {}).get(cheval)) or 50
    s_d = get_bucket_score(
        horse_stats.get("with_driver", {}).get(cheval, {}).get(_norm(driver)),
        min_courses=2) if driver else None
    s_h = get_bucket_score(
        horse_stats.get("hippo", {}).get(cheval, {}).get(hippodrome),
        min_courses=2) if hippodrome else None
    s_di = get_bucket_score(
        horse_stats.get("disc", {}).get(cheval, {}).get(discipline),
        min_courses=2) if discipline else None

    parts = [(s_g, 0.40)]
    if s_d is not None:
        parts.append((s_d, 0.25))
    if s_h is not None:
        parts.append((s_h, 0.20))
    if s_di is not None:
        parts.append((s_di, 0.15))

    tw = sum(w for _, w in parts)
    return sum(s * w for s, w in parts) / tw

def _composite_score(horse_s, driver_s, trainer_s):
    """Moyenne simple des 3 scores."""
    parts = []
    if horse_s is not None:
        parts.append(horse_s)
    if driver_s is not None:
        parts.append(driver_s)
    if trainer_s is not None:
        parts.append(trainer_s)
    if not parts:
        return 50.0
    return sum(parts) / len(parts)


# ═══════════════════════════════════════════════════════════
#  Analyse d'une course
# ═══════════════════════════════════════════════════════════

def analyser_course(parts_data, perfs_data, team_stats, horse_stats, discipline, hippodrome):
    """Analyse une course et retourne la liste triée par score composite."""
    partants = [p for p in parts_data.get("participants", [])
                if p.get("statut") == "PARTANT"]
    if not partants:
        return []

    # Cotes
    cotes = []
    for p in partants:
        rap = p.get("dernierRapportDirect") or p.get("dernierRapportReference")
        cotes.append(float(rap["rapport"]) if rap and rap.get("rapport") else None)

    # Probabilités marché
    inv_cotes = [1.0 / c if c and c > 0 else 0 for c in cotes]
    total_inv = sum(inv_cotes) or 1.0
    proba_marche = [x / total_inv * 100 for x in inv_cotes]

    analyses = []
    for i, p in enumerate(partants):
        cheval = p.get("nom") or ""
        driver = p.get("driver") or ""
        entraineur = p.get("entraineur") or ""
        nb_courses = p.get("nombreCourses", 0) or 0
        nb_victoires = p.get("nombreVictoires", 0) or 0
        nb_places = p.get("nombrePlaces", 0) or 0

        gains = p.get("gainsParticipant", {}) or {}
        gains_carriere = gains.get("gainsCarriere", 0) or 0

        # Scores des 3 piliers (formules identiques à l'ancien projet)
        s_horse = get_horse_score(cheval, driver, hippodrome, discipline, horse_stats)
        s_driver = get_team_score_multi(driver, "drivers", team_stats, discipline, hippodrome)
        s_trainer = get_team_score_multi(entraineur, "entraineurs", team_stats, discipline)

        # Score composite = moyenne des 3
        s_composite = _composite_score(s_horse, s_driver, s_trainer)

        # Edge vs marché
        cote = cotes[i]
        proba_m = proba_marche[i]
        edge = round(s_composite - proba_m, 2) if cote else 0

        analyses.append({
            "numPmu": p.get("numPmu"),
            "nom": cheval,
            "age": p.get("age"),
            "sexe": p.get("sexe"),
            "driver": driver or "—",
            "entraineur": entraineur or "—",
            "musique": p.get("musique", ""),
            "nbCourses": nb_courses,
            "nbVictoires": nb_victoires,
            "nbPlaces": nb_places,
            "cote": cote,
            "probaMarche": round(proba_m, 2),
            "gainsCarriere": gains_carriere // 100,
            "deferre": p.get("deferre", ""),
            "oeilleres": p.get("oeilleres", ""),
            "urlCasaque": p.get("urlCasaque"),
            "ordreArrivee": p.get("ordreArrivee"),
            "scores": {
                "cheval": round(s_horse, 1),
                "driver": round(s_driver, 1),
                "entraineur": round(s_trainer, 1),
                "composite": round(s_composite, 1),
                "marche": round(proba_m, 1),
            },
            "edge": edge,
            "chance": round(s_composite, 2),
        })

    # Trier par score composite décroissant
    analyses.sort(key=lambda x: -x["scores"]["composite"])
    for rank, a in enumerate(analyses, 1):
        a["rang"] = rank

    return analyses


# ═══════════════════════════════════════════════════════════
#  Backtest
# ═══════════════════════════════════════════════════════════

def backtest(days_back=7):
    stats = compute_all_stats(max_days=HISTORY_DAYS, exclude_recent_days=days_back)
    team_stats = stats["team_stats"]
    horse_stats = stats["horse_stats"]
    today = datetime.now()

    results = {
        "total_courses": 0, "top1_winner": 0, "top1_top3": 0,
        "mise_totale": 0.0, "gain_total": 0.0, "value_bets": [],
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
                    tasks.append((date_str, r["numOfficiel"], c["numOrdre"]))
                    metas.append({
                        "date": d.strftime("%d/%m"),
                        "course": f"R{r['numOfficiel']}C{c['numOrdre']}",
                        "hippodrome": hippo,
                        "discipline": c.get("discipline", ""),
                    })

    with ThreadPoolExecutor(max_workers=20) as ex:
        fetched = list(ex.map(
            lambda args: (get_participants_cached(*args), get_performances(*args)),
            tasks
        ))

    for (parts, perfs), meta in zip(fetched, metas):
        if not parts:
            continue
        try:
            analyses = analyser_course(parts, perfs, team_stats, horse_stats,
                                        meta.get("discipline", ""),
                                        meta.get("hippodrome", ""))
        except Exception:
            continue
        if not analyses:
            continue

        results["total_courses"] += 1
        top1 = analyses[0]
        top1_place = top1.get("ordreArrivee", 0) or 0
        top1_cote = top1.get("cote") or 0

        if top1_place == 1:
            results["top1_winner"] += 1
        if 1 <= top1_place <= 3:
            results["top1_top3"] += 1

        results["mise_totale"] += 1
        if top1_place == 1 and top1_cote:
            results["gain_total"] += top1_cote

        for a in analyses:
            if a.get("edge", 0) >= 3 and a.get("cote"):
                results["value_bets"].append({
                    "course": meta["course"], "date": meta["date"],
                    "cheval": a["nom"], "cote": a["cote"], "edge": a["edge"],
                    "gagne": a.get("ordreArrivee", 0) == 1,
                })

    n = results["total_courses"] or 1
    results["taux_top1"] = round(results["top1_winner"] / n * 100, 2)
    results["taux_top1_place"] = round(results["top1_top3"] / n * 100, 2)
    results["roi"] = round(
        (results["gain_total"] - results["mise_totale"]) / max(results["mise_totale"], 1) * 100, 2)
    results["mise_totale"] = round(results["mise_totale"], 2)
    results["gain_total"] = round(results["gain_total"], 2)

    vb = results["value_bets"]
    if vb:
        gains_vb = sum(b["cote"] for b in vb if b["gagne"])
        results["vb_nb"] = len(vb)
        results["vb_winrate"] = round(sum(1 for b in vb if b["gagne"]) / len(vb) * 100, 2)
        results["vb_roi"] = round((gains_vb - len(vb)) / len(vb) * 100, 2)
    else:
        results["vb_nb"] = 0
        results["vb_winrate"] = 0
        results["vb_roi"] = 0

    return results


# ═══════════════════════════════════════════════════════════
#  Bilan quotidien
# ═══════════════════════════════════════════════════════════

def bilan(days_back=7):
    stats = compute_all_stats(max_days=HISTORY_DAYS, exclude_recent_days=days_back)
    team_stats = stats["team_stats"]
    horse_stats = stats["horse_stats"]
    today = datetime.now()
    daily_results = []

    for delta in range(1, days_back + 1):
        d = today - timedelta(days=delta)
        date_str = fmt_date(d)
        day = {
            "date": d.strftime("%d/%m/%Y"), "date_short": d.strftime("%d/%m"),
            "jour": ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"][d.weekday()],
            "total": 0, "top1": 0, "top2": 0, "top3": 0, "hors": 0, "top3_total": 0,
            "courses": [],
        }

        try:
            prog = get_programme(date_str)
        except Exception:
            continue

        tasks = []
        course_metas = []
        for r in prog["programme"]["reunions"]:
            hippo = r["hippodrome"]["libelleCourt"]
            for c in r["courses"]:
                if c.get("arriveeDefinitive"):
                    tasks.append((date_str, r["numOfficiel"], c["numOrdre"]))
                    course_metas.append({
                        "hippodrome": hippo,
                        "discipline": c.get("discipline", ""),
                    })

        with ThreadPoolExecutor(max_workers=20) as ex:
            fetched = list(ex.map(
                lambda args: (get_participants_cached(*args), get_performances(*args)),
                tasks
            ))

        for (parts, perfs), meta in zip(fetched, course_metas):
            if not parts:
                continue
            try:
                analyses = analyser_course(parts, perfs, team_stats, horse_stats,
                                            meta.get("discipline", ""),
                                            meta.get("hippodrome", ""))
            except Exception:
                continue
            if not analyses:
                continue

            day["total"] += 1
            top1 = analyses[0]
            place = top1.get("ordreArrivee", 0) or 0

            course_detail = {"nom": top1["nom"], "place": place, "cote": top1.get("cote")}
            if place == 1:
                day["top1"] += 1; course_detail["resultat"] = "🥇"
            elif place == 2:
                day["top2"] += 1; course_detail["resultat"] = "🥈"
            elif place == 3:
                day["top3"] += 1; course_detail["resultat"] = "🥉"
            else:
                day["hors"] += 1
                course_detail["resultat"] = f"#{place}" if place > 0 else "—"

            if 1 <= place <= 3:
                day["top3_total"] += 1
            day["courses"].append(course_detail)

        if day["total"] > 0:
            day["taux_top1"] = round(day["top1"] / day["total"] * 100, 1)
            day["taux_top3"] = round(day["top3_total"] / day["total"] * 100, 1)
            daily_results.append(day)

    return daily_results


# ═══════════════════════════════════════════════════════════
#  Auth
# ═══════════════════════════════════════════════════════════

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin"):
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated


# ═══════════════════════════════════════════════════════════
#  Routes
# ═══════════════════════════════════════════════════════════

@app.route("/")
def public_home():
    return render_template("public.html")

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

@app.route("/admin")
@admin_required
def admin_home():
    return render_template("index.html")

@app.route("/backtest")
@admin_required
def backtest_page():
    return render_template("backtest.html")

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
                "heure": datetime.fromtimestamp(
                    c["heureDepart"] / 1000
                ).strftime("%H:%M") if c.get("heureDepart") else "",
                "nbPartants": c.get("nombreDeclaresPartants"),
                "arriveeDefinitive": c.get("arriveeDefinitive", False),
            } for c in r["courses"]],
        })
    return jsonify({"date": date_str, "reunions": out})

@app.route("/api/stats-status")
@admin_required
def api_stats_status():
    stats = get_stats()
    progress = dict(_stats_progress)
    if stats:
        progress["ready"] = True
        progress["horses"] = progress.get("horses", len(stats.get("horse_stats", {}).get("global", {})))
        progress["drivers"] = progress.get("drivers", len(stats.get("team_stats", {}).get("drivers", {})))
        progress["trainers"] = progress.get("trainers", len(stats.get("team_stats", {}).get("entraineurs", {})))
    else:
        progress["ready"] = False
    return jsonify(progress)

@app.route("/api/course/<int:r_num>/<int:c_num>")
@admin_required
def api_course(r_num, c_num):
    date_str = request.args.get("date") or fmt_date(datetime.now())
    live = request.args.get("live") == "1"

    stats = get_stats()
    if not stats:
        return jsonify({"error": "Stats pas encore prêtes", "status": "loading"}), 503

    try:
        prog = get_programme(date_str)
        parts = get_participants_live(date_str, r_num, c_num) if live else get_participants(date_str, r_num, c_num)
        perfs = get_performances(date_str, r_num, c_num)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    if not parts:
        return jsonify({"error": "Aucun participant"}), 404

    course_info = None
    hippodrome = None
    discipline = None
    for r in prog["programme"]["reunions"]:
        if r["numOfficiel"] == r_num:
            hippodrome = r["hippodrome"]["libelleCourt"]
            for c in r["courses"]:
                if c["numOrdre"] == c_num:
                    discipline = c.get("discipline", "")
                    course_info = {
                        "libelle": c.get("libelle"),
                        "discipline": discipline,
                        "distance": c.get("distance"),
                        "corde": c.get("corde", ""),
                        "heure": datetime.fromtimestamp(
                            c["heureDepart"] / 1000
                        ).strftime("%H:%M") if c.get("heureDepart") else "",
                        "nbPartants": c.get("nombreDeclaresPartants"),
                        "arriveeDefinitive": c.get("arriveeDefinitive", False),
                    }

    team_stats = stats["team_stats"]
    horse_stats = stats["horse_stats"]
    analyses = analyser_course(parts, perfs, team_stats, horse_stats, discipline, hippodrome)

    return jsonify({
        "date": date_str, "hippodrome": hippodrome, "course": course_info,
        "analyses": analyses, "timestamp": datetime.now().isoformat(),
    })

@app.route("/api/backtest")
@admin_required
def api_backtest():
    days = int(request.args.get("days", 7))
    days = min(days, 14)
    try:
        return jsonify(backtest(days_back=days))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/bilan")
@admin_required
def api_bilan():
    days = int(request.args.get("days", 7))
    days = min(days, 14)
    try:
        data = bilan(days_back=days)
        totals = {"total": 0, "top1": 0, "top2": 0, "top3": 0, "hors": 0, "top3_total": 0}
        for d in data:
            for k in totals:
                totals[k] += d.get(k, 0)
        n = totals["total"] or 1
        totals["taux_top1"] = round(totals["top1"] / n * 100, 1)
        totals["taux_top3"] = round(totals["top3_total"] / n * 100, 1)
        return jsonify({"daily": data, "totals": totals})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/team-stats")
@admin_required
def api_team_stats():
    stats = get_stats()
    if not stats:
        return jsonify({"error": "Stats pas encore prêtes"}), 503

    team_stats = stats["team_stats"]
    drivers = sorted(team_stats["drivers"].items(),
                    key=lambda x: -(x[1]["v"] if x[1]["c"] >= 5 else 0))[:30]
    trainers = sorted(team_stats["entraineurs"].items(),
                     key=lambda x: -(x[1]["v"] if x[1]["c"] >= 5 else 0))[:30]
    return jsonify({
        "drivers": [{"nom": k, "courses": v["c"], "victoires": v["v"], "places": v["p"],
                     "taux_victoire": round(v["v"]/v["c"]*100, 1) if v["c"] else 0}
                    for k, v in drivers],
        "entraineurs": [{"nom": k, "courses": v["c"], "victoires": v["v"], "places": v["p"],
                         "taux_victoire": round(v["v"]/v["c"]*100, 1) if v["c"] else 0}
                        for k, v in trainers],
    })

@app.route("/api/force-refresh-stats", methods=["POST"])
@admin_required
def api_force_refresh():
    global _current_stats, _stats_ready
    _current_stats = None
    _stats_ready.clear()
    if os.path.exists(STATS_CACHE_FILE):
        os.remove(STATS_CACHE_FILE)
    _stats_progress = {"status": "loading", "message": "Recalcul...", "progress": 0}
    t = threading.Thread(target=_background_load_stats, daemon=True)
    t.start()
    return jsonify({"status": "refreshing"})

@app.route("/api/health")
def api_health():
    stats = get_stats()
    return jsonify({"status": "ok", "version": "simple",
                    "stats_ready": stats is not None})


if __name__ == "__main__":
    start_stats_loader()
    app.run(host="0.0.0.0", port=5000, debug=False)
