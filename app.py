"""
Turf Analyzer — Version épurée

Analyse des chevaux basée sur 3 piliers uniquement :
  1. Stats carrière du cheval (nb courses, victoires, places → win_rate, place_rate)
  2. Stats driver/jockey (courses, victoires, places → win_rate, place_rate)
  3. Stats entraîneur (courses, victoires, places → win_rate, place_rate)

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

STATS_CACHE_FILE = os.path.join(CACHE_DIR, "stats.pkl")
HISTORY_DAYS = 180

# ── Helpers ──
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
#  PMU API
# ═══════════════════════════════════════════════════════════

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

@lru_cache(maxsize=1024)
def get_performances(date_str, r_num, c_num):
    url = f"{PMU_BASE}/{date_str}/R{r_num}/C{c_num}/performances-detaillees/pretty"
    try:
        r = requests.get(url, headers=HEADERS, timeout=15)
        r.raise_for_status()
        return r.json()
    except Exception:
        return {"participants": []}


# ═══════════════════════════════════════════════════════════
#  Cache
# ═══════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════
#  Construction des stats (uniquement cheval/driver/entraineur)
# ═══════════════════════════════════════════════════════════

def _empty_bucket():
    return {"c": 0, "v": 0, "p": 0}

def _fetch_course_simple(args):
    date_str, r_num, c_num = args
    try:
        parts = get_participants(date_str, r_num, c_num)
        return (parts, date_str)
    except Exception:
        return None

def compute_all_stats(max_days=HISTORY_DAYS, exclude_recent_days=0):
    """Construit les stats carrière cheval + driver + entraineur.
    
    exclude_recent_days : exclut les N derniers jours (anti data leak pour backtest).
    """
    if exclude_recent_days == 0:
        cached = load_pickle(STATS_CACHE_FILE)
        if cached:
            return cached

    # Statistiques brutes : {nom: {"c": courses, "v": victoires, "p": places (top3)}}
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

    with ThreadPoolExecutor(max_workers=30) as ex:
        results = list(ex.map(_fetch_course_simple, tasks))

    for parts_data, date_str in results:
        if not parts_data:
            continue

        partants = [p for p in parts_data.get("participants", [])
                    if p.get("statut") == "PARTANT"]

        for p in partants:
            cheval = p.get("nom") or ""
            driver = p.get("driver") or ""
            entraineur = p.get("entraineur") or ""
            place = p.get("ordreArrivee", 0) or 0
            won = 1 if place == 1 else 0
            placed = 1 if 1 <= place <= 3 else 0

            # Stats cheval
            if cheval:
                horse_stats[cheval]["c"] += 1
                horse_stats[cheval]["v"] += won
                horse_stats[cheval]["p"] += placed

            # Stats driver
            if driver:
                driver_stats[driver]["c"] += 1
                driver_stats[driver]["v"] += won
                driver_stats[driver]["p"] += placed

            # Stats entraineur
            if entraineur:
                trainer_stats[entraineur]["c"] += 1
                trainer_stats[entraineur]["v"] += won
                trainer_stats[entraineur]["p"] += placed

    # Convertir en dicts simples pour sérialisation
    out = {
        "horse_stats": dict(horse_stats),
        "driver_stats": dict(driver_stats),
        "trainer_stats": dict(trainer_stats),
    }

    if exclude_recent_days == 0:
        save_pickle(STATS_CACHE_FILE, out)

    return out


# ═══════════════════════════════════════════════════════════
#  Scoring — 3 piliers → moyenne simple
# ═══════════════════════════════════════════════════════════

def _bucket_score(bucket, min_courses=5):
    """Score 0-100 à partir d'un bucket {c, v, p}."""
    if not bucket or bucket["c"] < min_courses:
        return None
    c, v, p = bucket["c"], bucket["v"], bucket["p"]
    win_rate = v / c
    place_rate = p / c
    confiance = min(1.0, c / 30)
    raw = win_rate * 200 + place_rate * 60
    return min(100, raw * confiance + 30 * (1 - confiance))

def _horse_score(nom, horse_stats, min_courses=5):
    """Score carrière du cheval."""
    bucket = horse_stats.get(nom)
    return _bucket_score(bucket, min_courses)

def _driver_score(nom, driver_stats, min_courses=5):
    """Score du driver/jockey."""
    bucket = driver_stats.get(nom)
    return _bucket_score(bucket, min_courses)

def _trainer_score(nom, trainer_stats, min_courses=5):
    """Score de l'entraîneur."""
    bucket = trainer_stats.get(nom)
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
    
    team_stats = {"horse_stats": ..., "driver_stats": ..., "trainer_stats": ...}
    """
    horse_stats = team_stats["horse_stats"]
    driver_stats = team_stats["driver_stats"]
    trainer_stats = team_stats["trainer_stats"]

    partants = [p for p in parts_data.get("participants", [])
                if p.get("statut") == "PARTANT"]
    if not partants:
        return []

    # Perfs par numéro
    perfs_by_num = {}
    for pp in (perfs_data or {}).get("participants", []):
        perfs_by_num[pp.get("numPmu")] = pp.get("coursesCourues", [])

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

        # Scores des 3 piliers
        s_horse = _horse_score(cheval, horse_stats) if cheval else None
        s_driver = _driver_score(driver, driver_stats) if driver else None
        s_trainer = _trainer_score(entraineur, trainer_stats) if entraineur else None

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
            # Scores des 3 piliers
            "scores": {
                "cheval": round(s_horse, 1) if s_horse is not None else None,
                "driver": round(s_driver, 1) if s_driver is not None else None,
                "entraineur": round(s_trainer, 1) if s_trainer is not None else None,
                "composite": round(s_composite, 1),
                "marche": round(proba_m, 1),
            },
            "edge": edge,
            "chance": round(s_composite, 2),  # Pour compatibilité templates
        })

    # Trier par score composite décroissant
    analyses.sort(key=lambda x: -x["scores"]["composite"])
    for rank, a in enumerate(analyses, 1):
        a["rang"] = rank

    # Nettoyage NaN
    for a in analyses:
        for key, val in list(a.items()):
            if isinstance(val, float):
                import math
                if math.isnan(val) or math.isinf(val):
                    a[key] = 0
        for key, val in list(a.get("scores", {}).items()):
            if isinstance(val, float):
                import math
                if math.isnan(val) or math.isinf(val):
                    a["scores"][key] = 0

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
            hippo = r["hippodrome"]["libelleCourt"]
            for c in r["courses"]:
                if c.get("arriveeDefinitive"):
                    tasks.append((date_str, r["numOfficiel"], c["numOrdre"]))
                    metas.append({
                        "date": d.strftime("%d/%m"),
                        "course": f"R{r['numOfficiel']}C{c['numOrdre']}",
                    })

    with ThreadPoolExecutor(max_workers=20) as ex:
        fetched = list(ex.map(
            lambda args: (get_participants(*args), get_performances(*args)),
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

        # Value bets : chevaux où composite > proba marché de 3+ points
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

    # Stats value bets
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
                lambda args: (get_participants(*args), get_performances(*args)),
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
#  Routes publiques
# ═══════════════════════════════════════════════════════════

@app.route("/")
def public_home():
    return render_template("public.html")

@app.route("/api/programme")
def api_programme():
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

@app.route("/api/course/<int:r_num>/<int:c_num>")
def api_course(r_num, c_num):
    """Analyse d'une course."""
    date_str = request.args.get("date") or fmt_date(datetime.now())
    try:
        prog = get_programme(date_str)
        parts = get_participants(date_str, r_num, c_num)
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

    # Anti-leak : stats ne contiennent pas les jours analysés
    today = datetime.now()
    try:
        analysed_date = datetime.strptime(date_str, "%d%m%Y")
    except (ValueError, TypeError):
        analysed_date = today
    exclude_days = max((today - analysed_date).days, 0)

    team_stats = compute_all_stats(max_days=HISTORY_DAYS, exclude_recent_days=exclude_days)
    analyses = analyser_course(parts, perfs, team_stats)

    result = {
        "date": date_str,
        "hippodrome": hippodrome,
        "course": course_info,
        "analyses": analyses,
        "timestamp": datetime.now().isoformat(),
    }
    return jsonify(result)


# ═══════════════════════════════════════════════════════════
#  Auth admin
# ═══════════════════════════════════════════════════════════

def admin_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        if not session.get("admin"):
            return redirect(url_for("login_page"))
        return f(*args, **kwargs)
    return decorated

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
    """Liste des courses du jour (admin)."""
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

@app.route("/api/course-detail/<int:r_num>/<int:c_num>")
@admin_required
def api_course_detail(r_num, c_num):
    """Analyse complète d'une course (admin, réutilise api_course)."""
    return api_course(r_num, c_num)

@app.route("/api/backtest")
@admin_required
def api_backtest():
    days = int(request.args.get("days", 7))
    days = min(days, 30)
    try:
        result = backtest(days_back=days)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/bilan")
@admin_required
def api_bilan():
    days = int(request.args.get("days", 7))
    days = min(days, 30)
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
    team_stats = compute_all_stats(max_days=HISTORY_DAYS)
    drivers = sorted(team_stats["driver_stats"].items(),
                    key=lambda x: -(x[1]["v"] if x[1]["c"] >= 10 else 0))[:30]
    trainers = sorted(team_stats["trainer_stats"].items(),
                     key=lambda x: -(x[1]["v"] if x[1]["c"] >= 10 else 0))[:30]
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

@app.route("/api/health")
def api_health():
    return jsonify({"status": "ok", "version": "simple"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
