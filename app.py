"""
Turf Analyzer — Moteur de calcul v8

Classement hippique fondé sur 3 piliers (cheval / driver / entraîneur) traités
comme des **multiplicateurs de force** (modèle de Bradley-Terry) :

  • normalisés par la taille du champ
  • régressés vers la moyenne (shrinkage empirique)
  • combinés en probabilités calibrées (gagnant + top 3)
  • comparés aux cotes du marché déoverroundées (Shin) → edge / cote juste

Le moteur vit dans lib/scoring.py (testable isolément).
"""

from flask import Flask, jsonify, render_template, request, session, redirect, url_for
from functools import wraps
from datetime import datetime, timedelta
import requests
import os
import pickle
import threading
from functools import lru_cache
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

# Moteur de calcul v8 (multiplicateurs Bradley-Terry + Shin + Harville)
from lib.scoring import analyze_course as _score_course, PLACE_TOP

app = Flask(__name__)
app.secret_key = "turf-analyzer-simple"
ADMIN_PASSWORD = "admin123"

PMU_BASE = "https://offline.turfinfo.api.pmu.fr/rest/client/61/programme"
HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TurfAnalyzer/1.0)"}
CACHE_DIR = os.environ.get("CACHE_DIR", "/tmp/turf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

STATS_CACHE_FILE = os.path.join(CACHE_DIR, "stats_v8.pkl")  # v8: buckets {c,v,p,dw,dp}
HISTORY_DAYS = 180
WINDOW_SHORT = 30

# ── Global state ──
_stats_ready = threading.Event()
_current_stats = None
_stats_progress = {"status": "init", "message": "Initialisation...", "progress": 0}
_progress_lock = threading.Lock()


def _norm(name):
    if not name:
        return ""
    return " ".join(str(name).upper().split())

def fmt_date(d):
    return d.strftime("%d%m%Y")

def _update_progress(done, total, msg=None):
    with _progress_lock:
        pct = round(done / max(total, 1) * 100)
        _stats_progress["status"] = "loading"
        _stats_progress["progress"] = pct
        _stats_progress["done"] = done
        _stats_progress["total"] = total
        _stats_progress["message"] = msg or f"{done}/{total} courses ({pct}%)"

def _freeze(d):
    if isinstance(d, defaultdict):
        return {k: _freeze(v) for k, v in d.items()}
    return d

def _empty_bucket():
    """Bucket de stats : c=courses, v=victoires, p=places(top3),
    dw=Σ1/N (difficulté victoire), dp=Σmin(3,N)/N (difficulté place)."""
    return {"c": 0, "v": 0, "p": 0, "dw": 0.0, "dp": 0.0}


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
#  Build tasks
# ═══════════════════════════════════════════════════════════

def _build_tasks(max_days, exclude_recent_days):
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
#  Process one task (stats brutes PMU uniquement)
# ═══════════════════════════════════════════════════════════

def _process_task(task, ts_drivers, ts_drivers_short, ts_drivers_disc, ts_drivers_hippo,
                  ts_entraineurs, ts_entraineurs_short, ts_entraineurs_disc,
                  hs_global, hs_with_driver, hs_hippo, hs_disc):
    """Traite une course et met à jour les compteurs (c,v,p) + difficultés
    de taille de champ (dw = Σ1/N, dp = Σ min(3,N)/N)."""
    date_str, r_num, c_num, discipline, hippo, delta = task
    is_short = delta <= WINDOW_SHORT

    try:
        parts = get_participants_cached(date_str, r_num, c_num)
    except Exception:
        return

    # taille du champ = nombre de partants (clé de la normalisation v8)
    partants = [p for p in parts.get("participants", []) if p.get("statut") == "PARTANT"]
    n_field = len(partants) or 1
    dw_inc = 1.0 / n_field                      # difficulté victoire d'un moyen
    dp_inc = min(PLACE_TOP, n_field) / n_field  # difficulté place d'un moyen

    for p in partants:
        cheval = _norm(p.get("nom"))
        driver = _norm(p.get("driver"))
        entraineur = _norm(p.get("entraineur"))
        place = p.get("ordreArrivee", 0) or 0
        won = 1 if place == 1 else 0
        placed = 1 if 1 <= place <= 3 else 0

        # ── Stats cheval ──
        if cheval:
            hb = hs_global[cheval]
            hb["c"] += 1; hb["v"] += won; hb["p"] += placed
            hb["dw"] += dw_inc; hb["dp"] += dp_inc
            if driver:
                d = hs_with_driver[cheval][driver]
                d["c"] += 1; d["v"] += won; d["p"] += placed
                d["dw"] += dw_inc; d["dp"] += dp_inc
            if hippo:
                h = hs_hippo[cheval][hippo]
                h["c"] += 1; h["v"] += won; h["p"] += placed
                h["dw"] += dw_inc; h["dp"] += dp_inc
            if discipline:
                di = hs_disc[cheval][discipline]
                di["c"] += 1; di["v"] += won; di["p"] += placed
                di["dw"] += dw_inc; di["dp"] += dp_inc

        # ── Stats driver ──
        if driver:
            d = ts_drivers[driver]
            d["c"] += 1; d["v"] += won; d["p"] += placed
            d["dw"] += dw_inc; d["dp"] += dp_inc
            if is_short:
                s = ts_drivers_short[driver]
                s["c"] += 1; s["v"] += won; s["p"] += placed
                s["dw"] += dw_inc; s["dp"] += dp_inc
            if discipline:
                dd = ts_drivers_disc[driver][discipline]
                dd["c"] += 1; dd["v"] += won; dd["p"] += placed
                dd["dw"] += dw_inc; dd["dp"] += dp_inc
            if hippo:
                dh = ts_drivers_hippo[driver][hippo]
                dh["c"] += 1; dh["v"] += won; dh["p"] += placed
                dh["dw"] += dw_inc; dh["dp"] += dp_inc

        # ── Stats entraîneur ──
        if entraineur:
            e = ts_entraineurs[entraineur]
            e["c"] += 1; e["v"] += won; e["p"] += placed
            e["dw"] += dw_inc; e["dp"] += dp_inc
            if is_short:
                es = ts_entraineurs_short[entraineur]
                es["c"] += 1; es["v"] += won; es["p"] += placed
                es["dw"] += dw_inc; es["dp"] += dp_inc
            if discipline:
                ed = ts_entraineurs_disc[entraineur][discipline]
                ed["c"] += 1; ed["v"] += won; ed["p"] += placed
                ed["dw"] += dw_inc; ed["dp"] += dp_inc


# ═══════════════════════════════════════════════════════════
#  Background stats loading
# ═══════════════════════════════════════════════════════════

def _background_load_stats():
    global _current_stats, _stats_ready, _stats_progress

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

    _stats_progress = {"status": "loading", "message": "Dénombrement...", "progress": 0}

    try:
        all_tasks = _build_tasks(HISTORY_DAYS, 0)
        total = len(all_tasks)
        _stats_progress = {"status": "loading", "message": f"{total} courses à analyser",
                           "progress": 0, "total": total, "done": 0}

        if total == 0:
            _stats_progress = {"status": "error", "message": "Aucune course", "progress": 0}
            _stats_ready.set()
            return

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

        done = [0]
        done_lock = threading.Lock()

        def track_process(task):
            _process_task(task, ts_drivers, ts_drivers_short, ts_drivers_disc, ts_drivers_hippo,
                         ts_entraineurs, ts_entraineurs_short, ts_entraineurs_disc,
                         hs_global, hs_with_driver, hs_hippo, hs_disc)
            with done_lock:
                done[0] += 1
                if done[0] % 20 == 0:
                    _update_progress(done[0], total)

        print(f"[Stats] 🚀 {total} courses, 15 workers...")
        with ThreadPoolExecutor(max_workers=15) as ex:
            list(ex.map(track_process, all_tasks))

        _current_stats = {
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

def get_stats():
    if _stats_ready.is_set():
        return _current_stats
    return None

def compute_all_stats(max_days=HISTORY_DAYS, exclude_recent_days=0):
    """Version synchrone pour backtest/bilan avec exclusion."""
    if exclude_recent_days == 0:
        cached = load_pickle(STATS_CACHE_FILE)
        if cached:
            return cached

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
#  Analyse d'une course — délègue au moteur v8 (lib/scoring.py)
# ═══════════════════════════════════════════════════════════

def analyser_course(parts_data, perfs_data, team_stats, horse_stats, discipline, hippodrome):
    """Analyse une course via le moteur de multiplicateurs de force (Bradley-Terry).

    Signature conservée (perfs_data ignoré : la musique vient des participants).
    Voir lib/scoring.analyze_course pour le détail du calcul.
    """
    return _score_course(parts_data, team_stats or {}, horse_stats or {}, discipline, hippodrome)


# ═══════════════════════════════════════════════════════════
#  Backtest
# ═══════════════════════════════════════════════════════════

def backtest(days_back=7):
    stats = compute_all_stats(max_days=HISTORY_DAYS, exclude_recent_days=days_back)
    team_stats = stats["team_stats"]
    horse_stats = stats["horse_stats"]
    today = datetime.now()

    results = {"total_courses": 0, "top1_winner": 0, "top1_top3": 0,
               "mise_totale": 0.0, "gain_total": 0.0, "value_bets": []}

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
                    metas.append({"date": d.strftime("%d/%m"),
                                  "course": f"R{r['numOfficiel']}C{c['numOrdre']}",
                                  "hippodrome": hippo,
                                  "discipline": c.get("discipline", "")})

    with ThreadPoolExecutor(max_workers=20) as ex:
        fetched = list(ex.map(
            lambda args: (get_participants_cached(*args), get_performances(*args)), tasks))

    for (parts, perfs), meta in zip(fetched, metas):
        if not parts:
            continue
        try:
            analyses = analyser_course(parts, perfs, team_stats, horse_stats,
                                        meta["discipline"], meta["hippodrome"])
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
    results["roi"] = round((results["gain_total"] - results["mise_totale"]) / max(results["mise_totale"], 1) * 100, 2)
    results["mise_totale"] = round(results["mise_totale"], 2)
    results["gain_total"] = round(results["gain_total"], 2)

    vb = results["value_bets"]
    if vb:
        gains_vb = sum(b["cote"] for b in vb if b["gagne"])
        results["vb_nb"] = len(vb)
        results["vb_winrate"] = round(sum(1 for b in vb if b["gagne"]) / len(vb) * 100, 2)
        results["vb_roi"] = round((gains_vb - len(vb)) / len(vb) * 100, 2)
    else:
        results["vb_nb"] = 0; results["vb_winrate"] = 0; results["vb_roi"] = 0

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
        day = {"date": d.strftime("%d/%m/%Y"), "date_short": d.strftime("%d/%m"),
               "jour": ["Lun", "Mar", "Mer", "Jeu", "Ven", "Sam", "Dim"][d.weekday()],
               "total": 0, "top1": 0, "top2": 0, "top3": 0, "hors": 0,
               "top3_total": 0, "courses": []}

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
                    course_metas.append({"hippodrome": hippo, "discipline": c.get("discipline", "")})

        with ThreadPoolExecutor(max_workers=20) as ex:
            fetched = list(ex.map(
                lambda args: (get_participants_cached(*args), get_performances(*args)), tasks))

        for (parts, perfs), meta in zip(fetched, course_metas):
            if not parts:
                continue
            try:
                analyses = analyser_course(parts, perfs, team_stats, horse_stats,
                                            meta["discipline"], meta["hippodrome"])
            except Exception:
                continue
            if not analyses:
                continue

            day["total"] += 1
            top1 = analyses[0]
            place = top1.get("ordreArrivee", 0) or 0
            course_detail = {"nom": top1["nom"], "place": place, "cote": top1.get("cote")}

            if place == 1: day["top1"] += 1; course_detail["resultat"] = "🥇"
            elif place == 2: day["top2"] += 1; course_detail["resultat"] = "🥈"
            elif place == 3: day["top3"] += 1; course_detail["resultat"] = "🥉"
            else: day["hors"] += 1; course_detail["resultat"] = f"#{place}" if place > 0 else "—"

            if 1 <= place <= 3: day["top3_total"] += 1
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
                        "heure": datetime.fromtimestamp(
                            c["heureDepart"] / 1000
                        ).strftime("%H:%M") if c.get("heureDepart") else "",
                        "nbPartants": c.get("nombreDeclaresPartants"),
                        "arriveeDefinitive": c.get("arriveeDefinitive", False),
                    }

    stats = get_stats()
    if stats:
        team_stats = stats["team_stats"]
        horse_stats = stats["horse_stats"]
    else:
        # Stats pas encore chargées → utiliser des structures vides
        # Le score cheval fonctionnera quand même (stats PMU directes)
        team_stats = {}
        horse_stats = {}
        stats = {"team_stats": team_stats, "horse_stats": horse_stats}

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
    return jsonify({"status": "ok", "version": "v8",
                    "stats_ready": stats is not None})


if __name__ == "__main__":
    start_stats_loader()
    app.run(host="0.0.0.0", port=5000, debug=False)
