"""
scorer.py — Moteur de scoring multi-critères pour prédiction hippique
Calcule un score pondéré pour chaque partant.
"""

from typing import Optional

# ─── Poids des critères (total = 100) ───────────────────────────────────────
POIDS = {
    "forme_cheval":      30,   # musique récente du cheval
    "stats_jockey":      25,   # % victoires + % places jockey
    "stats_entraineur":  15,   # % victoires + % places entraîneur
    "corde_distance":    15,   # position de corde favorable
    "poids":             10,   # poids / charge
    "gains":              5,   # gains carrière (indicateur de niveau)
}


def score_forme(stats: dict, recence_boost: bool = True) -> float:
    """Score 0-100 basé sur la musique (taux victoire + taux place pondérés)."""
    if stats["courses"] == 0:
        return 30.0  # inconnu = score neutre

    base = stats["taux_victoire"] * 0.6 + stats["taux_place"] * 0.4

    # Bonus de forme récente (5 dernières courses)
    if recence_boost and stats.get("forme_recente"):
        recentes = stats["forme_recente"][:5]
        score_recente = 0
        for i, pos in enumerate(recentes):
            poids_recence = 5 - i  # plus récent = plus de poids
            if pos == 1:
                score_recente += poids_recence * 2
            elif pos <= 3:
                score_recente += poids_recence
        max_possible = sum(range(1, len(recentes) + 1)) * 2
        bonus = (score_recente / max_possible * 30) if max_possible > 0 else 0
        base = base * 0.7 + bonus * 0.3

    return min(base, 100.0)


def score_jockey(stats: dict) -> float:
    """Score jockey 0-100."""
    if stats["courses"] < 3:
        return 35.0
    return min(stats["taux_victoire"] * 0.5 + stats["taux_place"] * 0.5, 100.0)


def score_entraineur(stats: dict) -> float:
    """Score entraîneur 0-100."""
    if stats["courses"] < 3:
        return 35.0
    return min(stats["taux_victoire"] * 0.4 + stats["taux_place"] * 0.6, 100.0)


def score_corde(corde: int, nb_partants: int, distance: int) -> float:
    """
    Score corde 0-100.
    - Distances courtes (<1600m) : corde intérieure favorisée
    - Distances longues : impact plus faible
    """
    if nb_partants == 0 or corde == 0:
        return 50.0

    position_relative = corde / nb_partants  # 0=intérieur, 1=extérieur

    if distance < 1600:
        # Corde intérieure très avantageuse
        score = (1 - position_relative) * 80 + 20
    elif distance < 2200:
        # Impact modéré
        score = (1 - position_relative) * 50 + 50
    else:
        # Long parcours, impact minimal
        score = (1 - position_relative) * 20 + 60

    return min(score, 100.0)


def score_poids(poids: int, chevaux: list[dict]) -> float:
    """
    Score poids 0-100.
    Moins de poids = meilleur. Comparé à la moyenne du champ.
    """
    poids_valides = [c["poids"] for c in chevaux if c["poids"] and c["poids"] > 0]
    if not poids_valides or poids == 0:
        return 50.0

    moy = sum(poids_valides) / len(poids_valides)
    ecart = poids - moy

    # Chaque kg en moins = +5 points
    score = 50.0 - (ecart * 5)
    return max(10.0, min(score, 90.0))


def score_gains(gains: int, chevaux: list[dict]) -> float:
    """
    Score gains carrière 0-100.
    Indique le niveau de compétition du cheval.
    """
    gains_valides = [c["gains_carriere"] for c in chevaux if c["gains_carriere"] and c["gains_carriere"] > 0]
    if not gains_valides or gains == 0:
        return 30.0

    max_gains = max(gains_valides)
    if max_gains == 0:
        return 30.0

    return round((gains / max_gains) * 100, 1)


def calculer_score(cheval: dict, tous_chevaux: list[dict], nb_partants: int, distance: int) -> dict:
    """Calcule le score global d'un cheval avec détail par critère."""
    
    s_forme = score_forme(cheval["stats_cheval"])
    s_jockey = score_jockey(cheval["stats_jockey"])
    s_entraineur = score_entraineur(cheval["stats_entraineur"])
    s_corde = score_corde(cheval["corde"], nb_partants, distance)
    s_poids = score_poids(cheval["poids"], tous_chevaux)
    s_gains = score_gains(cheval["gains_carriere"], tous_chevaux)

    score_total = (
        s_forme      * POIDS["forme_cheval"]      / 100 +
        s_jockey     * POIDS["stats_jockey"]       / 100 +
        s_entraineur * POIDS["stats_entraineur"]   / 100 +
        s_corde      * POIDS["corde_distance"]     / 100 +
        s_poids      * POIDS["poids"]              / 100 +
        s_gains      * POIDS["gains"]              / 100
    )

    return {
        **cheval,
        "scores": {
            "forme_cheval":     round(s_forme, 1),
            "stats_jockey":     round(s_jockey, 1),
            "stats_entraineur": round(s_entraineur, 1),
            "corde_distance":   round(s_corde, 1),
            "poids":            round(s_poids, 1),
            "gains":            round(s_gains, 1),
        },
        "score_total": round(score_total, 2),
    }


def analyser_course(chevaux: list[dict], distance: int = 2000) -> dict:
    """
    Analyse complète d'une course.
    Retourne les chevaux classés + recommandations.
    """
    if not chevaux:
        return {"error": "Aucun partant trouvé"}

    nb = len(chevaux)

    # Calcul du score pour chaque cheval
    resultats = [calculer_score(c, chevaux, nb, distance) for c in chevaux]

    # Tri décroissant par score
    resultats.sort(key=lambda x: x["score_total"], reverse=True)

    # Attribution des rangs
    for i, r in enumerate(resultats):
        r["rang"] = i + 1

    # Normalisation des probabilités (softmax simplifié)
    scores = [r["score_total"] for r in resultats]
    score_min = min(scores)
    score_max = max(scores)
    ecart = score_max - score_min if score_max != score_min else 1

    for r in resultats:
        prob_relative = (r["score_total"] - score_min) / ecart
        r["probabilite"] = round(prob_relative * 60 + 10, 1)  # entre 10% et 70%

    # Recommandations
    gagnant = resultats[0]
    places = resultats[:3]
    outsider = next(
        (r for r in resultats[3:6] if r.get("cote", 0) and r["cote"] > 5),
        None
    )

    return {
        "classement": resultats,
        "recommandations": {
            "gagnant": gagnant,
            "places": places,
            "outsider": outsider,
        },
        "poids_criteres": POIDS,
    }
