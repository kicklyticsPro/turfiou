"""
lib/scoring.py — Moteur de calcul v8 (précision prédictive)

Modèle à multiplicateurs de force (Bradley-Terry) avec :
  • normalisation par la taille du champ (field size)
  • shrinkage empirique vers la moyenne de population (m = 1)
  • probabilités calibrées (softmax + Harville pour le top 3)
  • déoverround du marché (méthode de Shin)
  • edge = proba_modèle − proba_marché (vrais "value bets")

Aucune dépendance externe (stdandar lib uniquement) → testable isolément.
Voir SCORING_DESIGN.md pour la justification mathématique.
"""

from __future__ import annotations
import math
from typing import Optional

# ════════════════════════════════════════════════════════════
#  Constantes réglables
# ════════════════════════════════════════════════════════════

DEFAULT_FIELD = 12        # taille de champ moyenne si inconnue (stats carrière PMU)
KAPPA_WIN = 2.0           # shrinkage victoire (calibré par backtest 178 courses)
KAPPA_PLACE = 3.0         # shrinkage place     (×3 car ~3 places/victoire)
POWER_SCALE = 28.0        # points par unité de ln(m) ; 50 = moyenne
MUSIQUE_DECAY = 0.78      # décroissance temporelle (forme récente)

# Température (Platt scaling) : applique un exposant global aux multiplicateurs
# de force avant le softmax. T<1 = probas plus uniformes (moins confiant),
# T>1 = probas plus tranchées. Calibré par backtest (voir calibrate.py) :
# le modèle est sous-confiant à T=1.0 ; T=2.5 corrige la calibration (log-loss).
TEMPERATURE = 2.5

# Pondération des piliers (calibrée : cheval renforcé, entraîneur réduit)
W_HORSE, W_DRIVER, W_TRAINER = 0.55, 0.30, 0.15

# ── Améliorations v9 ──
# Blend modèle + marché : λ_final = λ_modèle^(1-β) · λ_marché^β.
# Le marché (cotes) est le prédicteur le plus puissant ; le modèle ajoute le
# signal de forme/compétence. β=0 → modèle seul, β=1 → marché seul.
# Calibré sur 178 courses : β=0.85 maximise la précision tout en gardant une
# contribution utile du modèle (le blend BAT le marché seul sur le Top3 :
# 64% vs 60.7%). L'edge (value bet) reste calculé sur le modèle PUR (β-indépendant).
MARKET_BLEND = 0.85

# Effet de la corde (draw). Désactivé (0.0) après calibration : le marché encode
# déjà la position à la corde, et un pénalty générique sans historique par cheval
# n'ajoute que du bruit (Top3 64.0% à 0.0 vs 62.4% à 0.10).
DRAW_COEF = 0.0

# Pondération des contextes par pilier
TEAM_CTX = {"global": 0.40, "short": 0.30, "disc": 0.20, "hippo": 0.10}
TRAINER_CTX = {"global": 0.45, "short": 0.30, "disc": 0.25}
HORSE_CTX = {
    "global": 0.30, "career": 0.18, "with_driver": 0.18,
    "hippo": 0.12, "disc": 0.12, "musique": 0.10,
}

PLACE_TOP = 3  # un "placé" = dans les 3 premiers


# ════════════════════════════════════════════════════════════
#  Utilitaires
# ════════════════════════════════════════════════════════════

def _clamp(x, lo, hi):
    return max(lo, min(hi, x))


def _safe_log(x):
    """log naturel, borné pour éviter les extrêmes."""
    return math.log(_clamp(x, 1e-3, 1e3))


def _bucket_get(bucket, *keys, default=None):
    """Récupère une valeur dans un bucket en tolérant les anciens caches."""
    if not bucket:
        return default
    cur = bucket
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _bucket_cvp(bucket):
    """Extrait (c, v, p) d'un bucket, robuste."""
    if not bucket:
        return 0, 0, 0
    return (bucket.get("c", 0) or 0,
            bucket.get("v", 0) or 0,
            bucket.get("p", 0) or 0)


def _bucket_dw(bucket, c):
    """Difficulté victoire Σ(1/N). Si absente (vieux cache) → c / DEFAULT_FIELD."""
    dw = _bucket_get(bucket, "dw")
    if dw is None or dw <= 0:
        return max(c, 0) / DEFAULT_FIELD
    return dw


def _bucket_dp(bucket, c):
    """Difficulté place Σ min(3,N)/N. Si absente → approximation."""
    dp = _bucket_get(bucket, "dp")
    if dp is None or dp <= 0:
        return max(c, 0) * min(PLACE_TOP, DEFAULT_FIELD) / DEFAULT_FIELD
    return dp


# ════════════════════════════════════════════════════════════
#  A. Multiplicateurs normalisés + shrinkage Bayésien
# ════════════════════════════════════════════════════════════

def win_multiplier(c, v, dw, kappa=None):
    """Multiplicateur de victoire, normalisé par la taille du champ,
    régressé vers 1.0 (moyenne de population).

    m = (v + κ) / (dw + κ)   avec dw = Σ 1/N_i (victoires attendues d'un moyen).
    Un cheval moyen → v ≈ dw → m ≈ 1. Le κ régresse vers 1 sur petits échantillons.

    NB : kappa=None → lit la constante module KAPPA_WIN à l'appel (calibrable
    à chaud, cf calibrate.py).
    """
    if c <= 0 or dw <= 0:
        return 1.0
    if kappa is None:
        kappa = KAPPA_WIN
    return (v + kappa) / (dw + kappa)


def place_multiplier(c, p, dp, kappa=None):
    """Multiplicateur de place (top 3), normalisé + shrinkage vers 1.0."""
    if c <= 0 or dp <= 0:
        return 1.0
    if kappa is None:
        kappa = KAPPA_PLACE
    return (p + kappa) / (dp + kappa)


def bucket_win_mult(bucket, kappa=None, min_courses=1):
    """Multiplicateur victoire d'un bucket {c,v,p,dw}."""
    c, v, _ = _bucket_cvp(bucket)
    if c < min_courses:
        return None
    return win_multiplier(c, v, _bucket_dw(bucket, c), kappa)


def bucket_place_mult(bucket, kappa=None, min_courses=1):
    c, _, p = _bucket_cvp(bucket)
    if c < min_courses:
        return None
    return place_multiplier(c, p, _bucket_dp(bucket, c), kappa)


# ════════════════════════════════════════════════════════════
#  B. Combinaison log-space (multiplicative des odds ratios)
# ════════════════════════════════════════════════════════════

def blend_multipliers(parts):
    """Combine des multiplicateurs en espace log, poids renormalisés sur les
    présents.

    parts: liste de (multiplicateur, poids). Ignore les entrées None.
    Renvoie un multiplicateur (1.0 si rien d'utilisable → moyenne).
    """
    active = [(m, w) for m, w in parts if m is not None and w > 0]
    if not active:
        return 1.0
    tot_w = sum(w for _, w in active)
    log_m = sum(_safe_log(m) * (w / tot_w) for m, w in active)
    return math.exp(log_m)


# ════════════════════════════════════════════════════════════
#  Corde (draw position) — avantage des faibles numéros (inside)
# ════════════════════════════════════════════════════════════

def draw_multiplier(place_corde, n_field, coef=None):
    """Multiplicateur lié à la position à la corde (draw).
    Inside (placeCorde = 1) = référence (m=1), outside = pénalité exponentielle.
    Pénalité croît avec le nombre de chevaux tirés à l'intérieur (placeCorde-1),
    ce qui reflète le trafic à remonter."""
    if not place_corde or place_corde < 1 or n_field <= 1:
        return 1.0
    if coef is None:
        coef = DRAW_COEF
    return math.exp(-coef * (place_corde - 1))


# ════════════════════════════════════════════════════════════
#  G. Score d'affichage 0-100 (50 = moyenne)
# ════════════════════════════════════════════════════════════

def power_score(m):
    """Mappe un multiplicateur vers un score 0-100 centré sur 50 (m=1).
    Monotone ⇒ préserve le classement par probabilité."""
    return round(_clamp(50.0 + POWER_SCALE * _safe_log(m), 0.0, 100.0), 1)


# ════════════════════════════════════════════════════════════
#  Musique — forme récente (signal de momentum)
# ════════════════════════════════════════════════════════════

# performance normalisée par rang (1=gagnant). Approx sans taille de champ.
_PLACE_PERF = {1: 1.00, 2: 0.82, 3: 0.68, 4: 0.55, 5: 0.45,
               6: 0.37, 7: 0.30, 8: 0.24, 9: 0.18, 0: 0.10}
_INCIDENT_PERF = 0.05   # D=disqualifié, R=retiré, T=tombé, A=arrêté, I=interrompu


def musique_form_multiplier(musique: Optional[str]):
    """Parse une musique PMU et renvoie un multiplicateur de forme récente.

    Format : "1a 3a Da 0a 4a" (gauche = plus récent).
    Renvoie un multiplicateur dans ~[0.5, 1.7], 1.0 si non exploitable.
    """
    if not musique or not isinstance(musique, str):
        return 1.0
    tokens = musique.strip().split()
    if not tokens:
        return 1.0

    perf_sum, w_sum = 0.0, 0.0
    for i, tok in enumerate(tokens[:10]):           # 10 dernières courses max
        tok = tok.strip()
        if not tok:
            continue
        # chiffre de tête → place ; sinon incident
        digits = ""
        for ch in tok:
            if ch.isdigit():
                digits += ch
            else:
                break
        if digits:
            place = int(digits)
            perf = _PLACE_PERF.get(place, _INCIDENT_PERF if place == 0 else 0.08)
        elif tok[0].upper() in ("D", "R", "T", "A", "I", "P"):
            perf = _INCIDENT_PERF
        else:
            continue
        w = MUSIQUE_DECAY ** i
        perf_sum += perf * w
        w_sum += w

    if w_sum <= 0:
        return 1.0
    perf_avg = perf_sum / w_sum                      # ∈ [0, 1]
    # mappe perf_avg → multiplicateur : 0.33 (médiocre) ≈ 1.0
    return round(_clamp(0.45 + 1.30 * perf_avg, 0.45, 1.75), 3)


# ════════════════════════════════════════════════════════════
#  E. Déoverround du marché — méthode de Shin (1993)
# ════════════════════════════════════════════════════════════

def shin_probabilities(odds, z_max=0.30, iters=60):
    """Probabilités "vraies" du marché, marge bookmaker retirée (méthode de Shin).

    odds: liste de cotes (>0) ou None (cote manquante → proba 0).
    Renvoie une liste de probabilités en % (même longueur), Σ = 100 %.
    """
    n = len(odds)
    out = [0.0] * n
    # probas implicites brutes b_i = 1/cote
    b = []
    idx = []
    for i, o in enumerate(odds):
        if o and float(o) > 1.0:
            b.append(1.0 / float(o))
            idx.append(i)
    if not b:
        return out
    S = sum(b)
    r = [bi / S for bi in b]                         # implicites normalisées
    if abs(S - 1.0) < 1e-6:                          # pas de marge → simple normalisation
        for k, i in enumerate(idx):
            out[i] = r[k] * 100.0
        return out

    # bissection sur z (taux d'insider trading) pour que Σ p_i(z) = 1
    def probs_sum(z):
        tot = 0.0
        for ri in r:
            disc = z * z + 4.0 * (1.0 - z) * ri * ri
            tot += (math.sqrt(max(disc, 0.0)) - z) / (2.0 * (1.0 - z))
        return tot

    lo, hi = 0.0, z_max
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if probs_sum(mid) > 1.0:
            lo = mid
        else:
            hi = mid
    z = 0.5 * (lo + hi)
    den = 2.0 * (1.0 - z)
    for k, i in enumerate(idx):
        disc = z * z + 4.0 * (1.0 - z) * r[k] * r[k]
        out[i] = (math.sqrt(max(disc, 0.0)) - z) / den * 100.0
    # sécurité arrondi → somme exacte 100
    s = sum(out)
    if s > 0:
        out = [v * 100.0 / s for v in out]
    return out


# ════════════════════════════════════════════════════════════
#  Multiplicateurs par entité
# ════════════════════════════════════════════════════════════

def _norm(name):
    return " ".join(str(name or "").upper().split())


def team_multiplier(name, kind, team_stats, discipline=None, hippodrome=None):
    """Multiplicateur driver ou entraîneur, combinant plusieurs contextes."""
    if not team_stats or not name:
        return 1.0, 1.0
    name = _norm(name)
    base = team_stats.get(kind, {})
    short = team_stats.get(f"{kind}_short", {})
    disc = team_stats.get(f"{kind}_disc", {})

    gb = base.get(name)
    sb = short.get(name)
    db = disc.get(name, {}).get(discipline) if discipline else None
    hb = (team_stats.get(f"{kind}_hippo", {}).get(name, {}).get(hippodrome)
          if (kind == "drivers" and hippodrome) else None)

    m_g = bucket_win_mult(gb)
    m_s = bucket_win_mult(sb, min_courses=1)
    m_d = bucket_win_mult(db, min_courses=1)
    m_h = bucket_win_mult(hb, min_courses=1)

    ctx = TEAM_CTX if kind == "drivers" else TRAINER_CTX
    parts = [(m_g, ctx["global"]), (m_s, ctx["short"]),
             (m_d, ctx["disc"]), (m_h, ctx.get("hippo", 0.0))]
    m_win = blend_multipliers(parts)

    # multiplicateur place (pour proba top 3) — mêmes contextes
    mp_g = bucket_place_mult(gb)
    mp_s = bucket_place_mult(sb)
    mp_d = bucket_place_mult(db)
    mp_h = bucket_place_mult(hb)
    parts_p = [(mp_g, ctx["global"]), (mp_s, ctx["short"]),
               (mp_d, ctx["disc"]), (mp_h, ctx.get("hippo", 0.0))]
    m_place = blend_multipliers(parts_p)
    return m_win, m_place


def horse_multiplier(cheval, driver, hippodrome, discipline, horse_stats,
                     nb_courses=0, nb_victoires=0, nb_places=0, musique=""):
    """Multiplicateur cheval combinant carrière PMU, historique 180j,
    contexte driver/hippo/discipline et forme récente (musique)."""
    if not horse_stats or not cheval:
        # sans stats calculées → on se rabat sur la carrière PMU seule
        if nb_courses and nb_courses >= 2:
            dw = nb_courses / DEFAULT_FIELD
            m_c = win_multiplier(nb_courses, nb_victoires, dw)
            mp_c = place_multiplier(nb_courses, nb_places, nb_courses * PLACE_TOP / DEFAULT_FIELD)
            return m_c, mp_c
        return 1.0, 1.0

    cheval = _norm(cheval)
    global_b = horse_stats.get("global", {}).get(cheval)
    driver_b = horse_stats.get("with_driver", {}).get(cheval, {}).get(_norm(driver)) if driver else None
    hippo_b = horse_stats.get("hippo", {}).get(cheval, {}).get(hippodrome) if hippodrome else None
    disc_b = horse_stats.get("disc", {}).get(cheval, {}).get(discipline) if discipline else None

    m_global = bucket_win_mult(global_b, min_courses=1)
    m_driver = bucket_win_mult(driver_b, min_courses=1)
    m_hippo = bucket_win_mult(hippo_b, min_courses=1)
    m_disc = bucket_win_mult(disc_b, min_courses=1)

    # carrière PMU (instantanée, lifetime) — dw approximé par champ moyen
    m_career = None
    if nb_courses and nb_courses >= 2:
        m_career = win_multiplier(nb_courses, nb_victoires, nb_courses / DEFAULT_FIELD)

    m_form = musique_form_multiplier(musique)

    ctx = HORSE_CTX
    parts = [
        (m_global, ctx["global"]), (m_career, ctx["career"]),
        (m_driver, ctx["with_driver"]), (m_hippo, ctx["hippo"]),
        (m_disc, ctx["disc"]), (m_form, ctx["musique"]),
    ]
    m_win = blend_multipliers(parts)

    # place
    mp_global = bucket_place_mult(global_b)
    mp_driver = bucket_place_mult(driver_b)
    mp_hippo = bucket_place_mult(hippo_b)
    mp_disc = bucket_place_mult(disc_b)
    mp_career = None
    if nb_courses and nb_courses >= 2:
        mp_career = place_multiplier(nb_courses, nb_places, nb_courses * PLACE_TOP / DEFAULT_FIELD)
    parts_p = [
        (mp_global, ctx["global"]), (mp_career, ctx["career"]),
        (mp_driver, ctx["with_driver"]), (mp_hippo, ctx["hippo"]),
        (mp_disc, ctx["disc"]), (m_form, ctx["musique"]),
    ]
    m_place = blend_multipliers(parts_p)
    return m_win, m_place


# ════════════════════════════════════════════════════════════
#  D. Marginales de rang — Harville (top 3)
# ════════════════════════════════════════════════════════════

def harville_top3(weights):
    """P(top 3) de chaque concurrent sous le modèle de Plackett-Luce / Harville.

    weights: multiplicateurs λ_i. P(gagner) = λ_i / Σλ, puis retrait séquentiel.
    Complexité O(N²) + O(N³) — fine pour N ≤ ~20 partants.
    """
    n = len(weights)
    if n == 0:
        return []
    if n == 1:
        return [1.0]
    S = sum(weights)
    if S <= 0:
        return [1.0 / n] * n

    p1 = [weights[i] / S for i in range(n)]
    p2 = [0.0] * n
    for j in range(n):
        rem = S - weights[j]
        if rem <= 0:
            continue
        for i in range(n):
            if i != j:
                p2[i] += p1[j] * weights[i] / rem

    p3 = [0.0] * n
    for j in range(n):
        rem1 = S - weights[j]
        if rem1 <= 0:
            continue
        for k in range(n):
            if k == j:
                continue
            rem2 = rem1 - weights[k]
            if rem2 <= 0:
                continue
            for i in range(n):
                if i == j or i == k:
                    continue
                p3[i] += p1[j] * (weights[k] / rem1) * (weights[i] / rem2)
    return [_clamp(p1[i] + p2[i] + p3[i], 0.0, 1.0) for i in range(n)]


# ════════════════════════════════════════════════════════════
#  Analyse d'une course — point d'entrée principal
# ════════════════════════════════════════════════════════════

def analyze_course(parts_data, team_stats, horse_stats, discipline, hippodrome):
    """Analyse tous les partants et renvoie le classement calibré.

    Contrat de sortie (compatible UI existante) :
      rang, nom, driver, entraineur, nbCourses, nbVictoires, nbPlaces,
      cote, ordreArrivee, edge,
      scores.{cheval, driver, entraineur, composite},
      + nouveaux : proba, probaTop3, fairOdds, strength, probaMarche
    """
    partants = [p for p in parts_data.get("participants", [])
                if p.get("statut") == "PARTANT"]
    if not partants:
        return []

    # ── cotes + probabilités marché (Shin) ──
    cotes = []
    for p in partants:
        rap = p.get("dernierRapportDirect") or p.get("dernierRapportReference")
        cotes.append(float(rap["rapport"]) if rap and rap.get("rapport") else None)
    mkt = shin_probabilities(cotes)

    # ── multiplicateurs par partant ──
    rows = []
    lam_model = []      # force du modèle pur (forme + draw) — pour l'edge
    for i, p in enumerate(partants):
        cheval = p.get("nom") or ""
        driver = p.get("driver") or ""
        entraineur = p.get("entraineur") or ""
        musique = p.get("musique", "") or ""

        nb_courses = p.get("nombreCourses", 0) or 0
        nb_victoires = p.get("nombreVictoires", 0) or 0
        nb_places = p.get("nombrePlaces", 0) or 0

        m_h_win, m_h_place = horse_multiplier(
            cheval, driver, hippodrome, discipline, horse_stats,
            nb_courses, nb_victoires, nb_places, musique)
        m_d_win, m_d_place = team_multiplier(driver, "drivers", team_stats, discipline, hippodrome)
        m_t_win, m_t_place = team_multiplier(entraineur, "entraineurs", team_stats, discipline)

        # pilier course (log-space) — température appliquée (calibration)
        lam_i_win = blend_multipliers([
            (m_h_win, W_HORSE), (m_d_win, W_DRIVER), (m_t_win, W_TRAINER)]) ** TEMPERATURE
        lam_i_place = blend_multipliers([
            (m_h_place, W_HORSE), (m_d_place, W_DRIVER), (m_t_place, W_TRAINER)]) ** TEMPERATURE

        # draw (corde) : facteur de modèle non capté par les stats
        place_corde = p.get("placeCorde") or 0
        m_draw = draw_multiplier(place_corde, len(partants))
        lam_i_win *= m_draw

        gains = p.get("gainsParticipant", {}) or {}
        rows.append({
            "numPmu": p.get("numPmu"),
            "nom": cheval,
            "age": p.get("age"),
            "sexe": p.get("sexe"),
            "driver": driver or "—",
            "entraineur": entraineur or "—",
            "musique": musique,
            "nbCourses": nb_courses,
            "nbVictoires": nb_victoires,
            "nbPlaces": nb_places,
            "cote": cotes[i],
            "gainsCarriere": (gains.get("gainsCarriere", 0) or 0) // 100,
            "deferre": p.get("deferre", ""),
            "oeilleres": p.get("oeilleres", ""),
            "urlCasaque": p.get("urlCasaque"),
            "ordreArrivee": p.get("ordreArrivee"),
            "_m_horse": m_h_win,
            "_m_driver": m_d_win,
            "_m_trainer": m_t_win,
            "_m_total": lam_i_win,
            "_m_place": lam_i_place,
        })
        lam_model.append(lam_i_win)

    # ── proba modèle pur (forme + draw) — pour l'edge & calibration ──
    S_model = sum(lam_model) or 1.0
    proba_model = [lam_model[i] / S_model * 100.0 for i in range(len(rows))]

    # ── blend modèle + marché (le grand levier de précision) ──
    # λ_final = λ_modèle^(1-β) · λ_marché^β ; marché = Shin (déoverround).
    # Sans cote → marché uniforme (1/N) → le modèle prédomine sur ce cheval.
    lam_mkt = []
    for i in range(len(rows)):
        pi = mkt[i] / 100.0
        lam_mkt.append(pi if pi > 0 else 1.0 / len(rows))
    lam_final = []
    for i in range(len(rows)):
        lm = max(lam_model[i], 1e-9)
        mk = max(lam_mkt[i], 1e-9)
        lam_final.append(math.pow(lm, 1.0 - MARKET_BLEND) * math.pow(mk, MARKET_BLEND))

    # ── probabilités calibrées (softmax + Harville) sur les forces blendées ──
    S_win = sum(lam_final) or 1.0
    top3 = harville_top3(lam_final)
    for i, r in enumerate(rows):
        proba_win = lam_final[i] / S_win * 100.0
        r["proba"] = round(proba_win, 2)
        r["probaTop3"] = round(top3[i] * 100.0, 2)
        r["fairOdds"] = round(100.0 / proba_win, 2) if proba_win > 0 else None
        r["strength"] = round(lam_final[i], 3)
        r["probaMarche"] = round(mkt[i], 2)
        r["scores"] = {
            "cheval": power_score(r["_m_horse"]),
            "driver": power_score(r["_m_driver"]),
            "entraineur": power_score(r["_m_trainer"]),
            "composite": power_score(r["_m_total"]),
        }
        # edge : proba modèle PUR vs marché (vrais "value bets")
        mkt_i = mkt[i]
        r["probaModel"] = round(proba_model[i], 2)
        r["edge"] = round(proba_model[i] - mkt_i, 2) if mkt_i else 0
        r["chance"] = round(proba_win, 1)

    # ── classement par probabilité de victoire ──
    rows.sort(key=lambda x: -x["proba"])
    for rank, r in enumerate(rows, 1):
        r["rang"] = rank

    return rows


# ════════════════════════════════════════════════════════════
#  Pour les tests / backtests synthétiques
# ════════════════════════════════════════════════════════════

def legacy_bucket_score(bucket, max_score=100, min_courses=5):
    """Ancienne formule (pour comparaison backtest : win*200 + place*60)."""
    if not bucket or bucket.get("c", 0) < min_courses:
        return None
    c, v, p = bucket["c"], bucket.get("v", 0), bucket.get("p", 0)
    tv, tp = v / c, p / c
    confiance = min(1.0, c / 30)
    raw = tv * 200 + tp * 60
    return min(max_score, raw * confiance + 30 * (1 - confiance))
