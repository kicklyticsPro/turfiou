"""
Turf Analyzer — Version épurée

Analyse des chevaux basée sur 3 piliers uniquement :
  1. Stats carrière du cheval (nb courses, victoires, places)
  2. Stats driver/jockey (nb courses, victoires, places)
  3. Stats entraîneur (nb courses, victoires, places)

Classement = moyenne simple des 3 scores.
Backtest = performance historique basé sur ce classement.
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
import time
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

STATS_CACHE_FILE = os.path.join(CACHE_DIR, "stats_v3.pkl")
STATS_STATUS_FILE = os.path.join(CACHE_DIR, "stats_status.json")
HISTORY_DAYS = 15  # Période de calcul des stats

# ── Global state ──
_stats_ready = threading.Event()
_current_stats = None
_stats_progress = {"status": "init", "message": "Initialisation..."}


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


# ═══════════════════════════════════════════════════════════
#  PMU API (avec cache lru)
# ═══════════════════════════════════════════════════════════

@lru_cache(maxsize=256)
def get_programme(date_str):
    r = requests.get(f"{PMU_BASE}/{date_str}", headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()

def get_participants_cached(date_str, r_num, c_num):
    """Version cache pour le calcul des stats historiques."""
    url = f"{PMU_BASE}/{date_str}/R{r_num}/C{c_num}/participants"
    r = requests.get(url, headers=HEADERS, timeout=15)
    r.raise_for_status()
    return r.json()

@lru_cache(maxsize=2048)
def get_participants(date_str, r_num, c_num):
    return get_participants_cached(date_str, r_num, c_num)

def get_participants_live(date_str, r_num, c_num):
    """Version SANS cache pour les courses du jour (résultats en temps réel)."""
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

def _save_stats_status(status):
    global _stats_progress
    _stats_progress = status
    try:
        with open(STATS_STATUS_FILE, "w") as f:
            json.dump(status, f)
    except Exception:
        pass

def _load_stats_status():
    try:
        if os.path.exists(STATS_STATUS_FILE):
            with open(STATS_STATUS_FILE, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return {"status": "none"}


# ═══════════════════════════════════════════════════════════
#  Construction des stats (cheval/driver/entraineur)
# ═══════════════════════════════════════════════════════════

def _empty_bucket():
    return {"c": 0, "v": 0, "p": 0}

def _fetch_course_simple(args):
    date_str, r_num, c_num = args
    try:
        parts = get_participants_cached(date_str, r_num, c_num)
        return (parts, date_str)
    except Exception:
        return None

def compute_all_stats(max_days=HISTORY_DAYS, exclude_recent_days=0):
    """Construit les stats carrière cheval + driver + entraineur."""
    if exclude_recent_days == 0:
        cached = load_pickle(STATS_CACHE_FILE)
        if cached:
            return cached

    horse_stats = defaultdict(_empty_bucket)
    driver_stats = defaultdict(_empty_bucket)
    trainer_stats = defaultdict(_empty_bucket)

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
            for c in r["courses"]:
                if c.get("arriveeDefinitive"):
                    tasks.append((date_str, r["numOfficiel"], c["numOrdre"]))

    print(f"[Stats] {len(tasks)} courses à analyser sur {max_days - start_delta + 1} jours...")

    with ThreadPoolExecutor(max_workers=30) as ex:
        results = list(ex.map(_fetch_course_simple, tasks))

    for parts_data, date_str in results:
        if not parts_data:
            continue

        partants = [p for p in parts_data.get("participants", [])
                    if p.get("statut") == "PARTANT"]

        for p in partants:
            cheval = _norm(p.get("nom"))
            driver = _norm(p.get("driver"))
            entraineur = _norm(p.get("entraineur"))
            place = p.get("ordreArrivee", 0) or 0
            won = 1 if place == 1 else 0
            placed = 1 if 1 <= place <= 3 else 0

            if cheval:
                horse_stats[cheval]["c"] += 1
                horse_stats[cheval]["v"] += won
                horse_stats[cheval]["p"] += placed
            if driver:
                driver_stats[driver]["c"] += 1
                driver_stats[driver]["v"] += won
                driver_stats[driver]["p"] += placed
            if entraineur:
                trainer_stats[entraineur]["c"] += 1
                trainer_stats[entraineur]["v"] += won
                trainer_stats[entraineur]["p"] += placed

    out = {
        "horse_stats": dict(horse_stats),
        "driver_stats": dict(driver_stats),
        "trainer_stats": dict(trainer_stats),
    }

    if exclude_recent_days == 0:
        save_pickle(STATS_CACHE_FILE, out)
        print(f"[Stats] ✅ Calculé et sauvegardé : {len(out['horse_stats'])} chevaux, "
              f"{len(out['driver_stats'])} drivers, {len(out['trainer_stats'])} entraîneurs")

    return out


# ═══════════════════════════════════════════════════════════
#  Background stats loading
# ═══════════════════════════════════════════════════════════

def _background_load_stats():
    """Charge les stats en arrière-plan au démarrage."""
    global _current_stats, _stats_ready

    # Vérifier le cache d'abord
    cached = load_pickle(STATS_CACHE_FILE)
    if cached:
        _current_stats = cached
        _stats_ready.set()
        print(f"[Stats] ✅ Chargé depuis le cache: {len(cached['horse_stats'])} chevaux")
        _save_stats_status({"status": "ready", "source": "cache",
                            "at": datetime.now().isoformat(),
                            "horses": len(cached["horse_stats"]),
                            "drivers": len(cached["driver_stats"]),
                            "trainers": len(cached["trainer_stats"])})
        return

    # Pas de cache → calculer avec progression
    _save_stats_status({"status": "loading", "started_at": datetime.now().isoformat(),
                        "message": "Calcul des stats en cours..."})
    try:
        today = datetime.now()
        horse_stats = defaultdict(_empty_bucket)
        driver_stats = defaultdict(_empty_bucket)
        trainer_stats = defaultdict(_empty_bucket)

        # Compter le total des courses
        total_tasks = 0
        for delta in range(1, HISTORY_DAYS + 1):
            d = today - timedelta(days=delta)
            date_str = fmt_date(d)
            try:
                prog = get_programme(date_str)
                for r in prog["programme"]["reunions"]:
                    for c in r["courses"]:
                        if c.get("arriveeDefinitive"):
                            total_tasks += 1
            except Exception:
                continue

        print(f"[Stats] {total_tasks} courses à analyser...")

        done = 0
        for delta in range(1, HISTORY_DAYS + 1):
            d = today - timedelta(days=delta)
            date_str = fmt_date(d)
            try:
                prog = get_programme(date_str)
            except Exception:
                continue
            for r in prog["programme"]["reunions"]:
                for c in r["courses"]:
                    if not c.get("arriveeDefinitive"):
                        continue
                    try:
                        parts = get_participants_cached(date_str, r["numOfficiel"], c["numOrdre"])
                        for p in parts.get("participants", []):
                            if p.get("statut") != "PARTANT":
                                continue
                            cheval = _norm(p.get("nom"))
                            driver = _norm(p.get("driver"))
                            entr = _norm(p.get("entraineur"))
                            place = p.get("ordreArrivee", 0) or 0
                            won = 1 if place == 1 else 0
                            placed = 1 if 1 <= place <= 3 else 0
                            if cheval:
                                horse_stats[cheval]["c"] += 1
                                horse_stats[cheval]["v"] += won
                                horse_stats[cheval]["p"] += placed
                            if driver:
                                driver_stats[driver]["c"] += 1
                                driver_stats[driver]["v"] += won
                                driver_stats[driver]["p"] += placed
                            if entr:
                                trainer_stats[entr]["c"] += 1
                                trainer_stats[entr]["v"] += won
                                trainer_stats[entr]["p"] += placed
                        done += 1
                        if done % 20 == 0:
                            pct = round(done / total_tasks * 100) if total_tasks else 0
                            _save_stats_status({"status": "loading", "progress": pct,
                                                "message": f"{done}/{total_tasks} courses ({pct}%)"})
                    except Exception:
                        done += 1
                        continue

        _current_stats = {
            "horse_stats": dict(horse_stats),
            "driver_stats": dict(driver_stats),
            "trainer_stats": dict(trainer_stats),
        }
        save_pickle(STATS_CACHE_FILE, _current_stats)
        _stats_ready.set()
        print(f"[Stats] ✅ Terminé : {len(_current_stats['horse_stats'])} chevaux")
        _save_stats_status({"status": "ready", "source": "computed",
                            "at": datetime.now().isoformat(),
                            "horses": len(_current_stats["horse_stats"]),
                            "drivers": len(_current_stats["driver_stats"]),
                            "trainers": len(_current_stats["trainer_stats"])})
    except Exception as e:
        import traceback
        traceback.print_exc()
        _stats_ready.set()
        _save_stats_status({"status": "error", "error": str(e),
                            "at": datetime.now().isoformat()})

def get_stats():
    """Retourne les stats si prêtes, sinon None."""
    if _stats_ready.is_set():
        return _current_stats
    return None

def start_stats_loader():
    """Démarre le chargement des stats dans un thread séparé."""
    t = threading.Thread(target=_background_load_stats, daemon=True)
    t.start()
    print("[Stats] 🔄 Chargement en arrière-plan...")


# ═══════════════════════════════════════════════════════════
#  Scoring — 3 piliers → moyenne simple
# ═══════════════════════════════════════════════════════════

def _bucket_score(bucket, min_courses=2):
    """Score 0-100 à partir d'un bucket {c, v, p}."""
    if not bucket or bucket["c"] < min_courses:
        return None
    c, v, p = bucket["c"], bucket["v"], bucket["p"]
    win_rate = v / c
    place_rate = p / c
    confiance = min(1.0, c / 20)
    raw = win_rate * 200 + place_rate * 60
    return min(100, raw * confiance + 30 * (1 - confiance))

def _score_from_career(nb_courses, nb_victoires, nb_places):
    """Score 0-100 à partir des stats carrière brutes du PMU."""
    if not nb_courses or nb_courses < 2:
        return None
    win_rate = nb_victoires / nb_courses
    place_rate = nb_places / nb_courses
    confiance = min(1.0, nb_courses / 20)
    raw = win_rate * 200 + place_rate * 60
    return min(100, raw * confiance + 30 * (1 - confiance))

def _horse_score(nom, horse_stats, nb_courses=0, nb_victoires=0, nb_places=0, min_courses=2):
    """Score du cheval : historique d'abord, fallback sur stats carrière PMU."""
    bucket = horse_stats.get(_norm(nom))
    s = _bucket_score(bucket, min_courses)
    if s is not None:
        return s
    # Fallback sur stats carrière du PMU
    return _score_from_career(nb_courses, nb_victoires, nb_places)

def _driver_score(nom, driver_stats, min_courses=2):
    bucket = driver_stats.get(_norm(nom))
    return _bucket_score(bucket, min_courses)

def _trainer_score(nom, trainer_stats, min_courses=2):
    bucket = trainer_stats.get(_norm(nom))
    return _bucket_score(bucket, min_courses)

def _composite_score(horse_s, driver_s, trainer_s):
    """Moyenne simple des 3 scores. None → 50 (neutre)."""
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

def analyser_course(parts_data, perfs_data, team_stats):
    """Analyse une course et retourne la liste triée par score composite.

    Utilise l'historique PMU (15 derniers jours) pour les stats driver/entraineur,
    et les stats carrière du PMU (nombreCourses/Victoires/Places) pour le cheval.
    """
    horse_stats = team_stats["horse_stats"]
    driver_stats = team_stats["driver_stats"]
    trainer_stats = team_stats["trainer_stats"]

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

        # ── Score cheval : stats carrière PMU directement ──
        # On utilise les stats carrière du cheval (nombreCourses/Victoires/Places)
        # car c'est la donnée la plus fiable et toujours disponible
        if nb_courses >= 2:
            s_horse = _score_from_career(nb_courses, nb_victoires, nb_places)
        else:
            # Fallback sur l'historique si pas assez de données carrière
            s_horse = _horse_score(cheval, horse_stats, nb_courses, nb_victoires, nb_places)

        # ── Score driver : historique des 15 derniers jours ──
        # Fallback sur carrière si pas d'historique
        s_driver = _driver_score(driver, driver_stats)
        if s_driver is None:
            # Essayer de trouver les stats carrière du driver
            dr_c = 0
            dr_v = 0
            dr_p = 0
            for h_name, h_bucket in horse_stats.items():
                # On ne peut pas savoir directement les stats carrière d'un driver
                # depuis les données du cheval, donc on garde None → 50
                pass
            s_driver = 50.0  # Neutre si pas de données

        # ── Score entraîneur : historique des 15 derniers jours ──
        s_trainer = _trainer_score(entraineur, trainer_stats)
        if s_trainer is None:
            s_trainer = 50.0  # Neutre si pas de données

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
                "cheval": round(s_horse, 1) if s_horse is not None else None,
                "driver": round(s_driver, 1) if s_driver is not None else None,
                "entraineur": round(s_trainer, 1) if s_trainer is not None else None,
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
    """Backtest simple : vérifie si le #1 du composite gagne ou se place."""
    team_stats = compute_all_stats(max_days=HISTORY_DAYS, exclude_recent_days=days_back)
    today = datetime.now()

    results = {
        "total_courses": 0,
        "top1_winner": 0,
        "top1_top3": 0,
        "mise_totale": 0.0,
        "gain_total": 0.0,
        "value_bets": [],
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
            for c in r["courses"]:
                if c.get("arriveeDefinitive"):
                    tasks.append((date_str, r["numOfficiel"], c["numOrdre"]))
                    metas.append({
                        "date": d.strftime("%d/%m"),
                        "course": f"R{r['numOfficiel']}C{c['numOrdre']}",
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
            analyses = analyser_course(parts, perfs, team_stats)
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

        # Value bets
        for a in analyses:
            if a.get("edge", 0) >= 3 and a.get("cote"):
                won = a.get("ordreArrivee", 0) == 1
                results["value_bets"].append({
                    "course": meta["course"],
                    "date": meta["date"],
                    "cheval": a["nom"],
                    "cote": a["cote"],
                    "edge": a["edge"],
                    "gagne": won,
                })

    n = results["total_courses"] or 1
    results["taux_top1"] = round(results["top1_winner"] / n * 100, 2)
    results["taux_top1_place"] = round(results["top1_top3"] / n * 100, 2)
    results["roi"] = round(
        (results["gain_total"] - results["mise_totale"]) / max(results["mise_totale"], 1) * 100, 2
    )
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
    """Stats par jour : où finit le #1 du composite."""
    team_stats = compute_all_stats(max_days=HISTORY_DAYS, exclude_recent_days=days_back)
    today = datetime.now()
    daily_results = []

    for delta in range(1, days_back + 1):
        d = today - timedelta(days=delta)
        date_str = fmt_date(d)
        day = {
            "date": d.strftime("%d/%m/%Y"),
            "date_short": d.strftime("%d/%m"),
            "jour": ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"][d.weekday()],
            "total": 0,
            "top1": 0, "top2": 0, "top3": 0, "hors": 0,
            "top3_total": 0,
            "courses": [],
        }

        try:
            prog = get_programme(date_str)
        except Exception:
            continue

        tasks = []
        for r in prog["programme"]["reunions"]:
            for c in r["courses"]:
                if c.get("arriveeDefinitive"):
                    tasks.append((date_str, r["numOfficiel"], c["numOrdre"]))

        with ThreadPoolExecutor(max_workers=20) as ex:
            fetched = list(ex.map(
                lambda args: (get_participants_cached(*args), get_performances(*args)),
                tasks
            ))

        for parts, perfs in fetched:
            if not parts:
                continue
            try:
                analyses = analyser_course(parts, perfs, team_stats)
            except Exception:
                continue
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
#  Routes publiques
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


# ═══════════════════════════════════════════════════════════
#  Routes admin
# ═══════════════════════════════════════════════════════════

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
    """Liste des courses du jour."""
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
    """Statut du chargement des stats."""
    stats = get_stats()
    progress = dict(_stats_progress)
    if stats:
        progress["ready"] = True
        progress["horses"] = len(stats.get("horse_stats", {}))
        progress["drivers"] = len(stats.get("driver_stats", {}))
        progress["trainers"] = len(stats.get("trainer_stats", {}))
    else:
        progress["ready"] = False
    return jsonify(progress)

@app.route("/api/course/<int:r_num>/<int:c_num>")
@admin_required
def api_course(r_num, c_num):
    """Analyse d'une course avec classement."""
    date_str = request.args.get("date") or fmt_date(datetime.now())
    live = request.args.get("live") == "1"  # Pour les courses en cours

    # Vérifier que les stats sont prêtes
    stats = get_stats()
    if not stats:
        return jsonify({"error": "Stats pas encore prêtes. Patientez quelques secondes ou rechargez.",
                        "status": "loading"}), 503

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
    for r in prog["programme"]["reunions"]:
        if r["numOfficiel"] == r_num:
            hippodrome = r["hippodrome"]["libelleCourt"]
            for c in r["courses"]:
                if c["numOrdre"] == c_num:
                    course_info = {
                        "libelle": c.get("libelle"),
                        "discipline": c.get("discipline"),
                        "distance": c.get("distance"),
                        "corde": c.get("corde", ""),
                        "heure": datetime.fromtimestamp(
                            c["heureDepart"] / 1000
                        ).strftime("%H:%M") if c.get("heureDepart") else "",
                        "nbPartants": c.get("nombreDeclaresPartants"),
                        "arriveeDefinitive": c.get("arriveeDefinitive", False),
                    }

    analyses = analyser_course(parts, perfs, stats)

    return jsonify({
        "date": date_str,
        "hippodrome": hippodrome,
        "course": course_info,
        "analyses": analyses,
        "timestamp": datetime.now().isoformat(),
    })

@app.route("/api/backtest")
@admin_required
def api_backtest():
    days = int(request.args.get("days", 7))
    days = min(days, 14)
    try:
        result = backtest(days_back=days)
        return jsonify(result)
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
    """Retourne les stats driver et entraineur."""
    stats = get_stats()
    if not stats:
        return jsonify({"error": "Stats pas encore prêtes"}), 503

    drivers = sorted(stats["driver_stats"].items(),
                    key=lambda x: -(x[1]["v"] if x[1]["c"] >= 5 else 0))[:30]
    trainers = sorted(stats["trainer_stats"].items(),
                     key=lambda x: -(x[1]["v"] if x[1]["c"] >= 5 else 0))[:30]
    return jsonify({
        "drivers": [{
            "nom": k,
            "courses": v["c"],
            "victoires": v["v"],
            "places": v["p"],
            "taux_victoire": round(v["v"]/v["c"]*100, 1) if v["c"] else 0
        } for k, v in drivers],
        "entraineurs": [{
            "nom": k,
            "courses": v["c"],
            "victoires": v["v"],
            "places": v["p"],
            "taux_victoire": round(v["v"]/v["c"]*100, 1) if v["c"] else 0
        } for k, v in trainers],
    })

@app.route("/api/force-refresh-stats", methods=["POST"])
@admin_required
def api_force_refresh():
    """Force le recalcul des stats."""
    global _current_stats, _stats_ready
    _current_stats = None
    _stats_ready.clear()
    # Supprimer le cache
    if os.path.exists(STATS_CACHE_FILE):
        os.remove(STATS_CACHE_FILE)
    _save_stats_status({"status": "loading", "progress": 0,
                        "message": "Recalcul en cours..."})
    t = threading.Thread(target=_background_load_stats, daemon=True)
    t.start()
    return jsonify({"status": "refreshing"})

@app.route("/api/health")
def api_health():
    stats = get_stats()
    return jsonify({
        "status": "ok",
        "version": "simple",
        "stats_ready": stats is not None,
        "stats_horses": len(stats["horse_stats"]) if stats else 0,
    })


if __name__ == "__main__":
    start_stats_loader()  # Charge les stats en arrière-plan au démarrage
    app.run(host="0.0.0.0", port=5000, debug=False)
