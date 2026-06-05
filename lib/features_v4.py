"""
v4 - Nouvelles features : pedigree, corde, équipements, régimes (spécialistes).
"""
import re
from collections import defaultdict


def empty_bucket():
    return {"c": 0, "v": 0, "p": 0}


# ============================================================
#  Pedigree : taux de réussite des descendants du père/mère
# ============================================================
def build_pedigree_stats(all_horses_data):
    """
    À partir d'une liste de (cheval, pere, mere, place), construit :
      pere_stats : {nom_pere: {c, v, p}}  (stats des descendants)
      mere_stats : pareil
    """
    pere_stats = defaultdict(empty_bucket)
    mere_stats = defaultdict(empty_bucket)
    for h in all_horses_data:
        pere = h.get("pere")
        mere = h.get("mere")
        place = h.get("place", 0) or 0
        won = 1 if place == 1 else 0
        placed = 1 if 1 <= place <= 3 else 0
        if pere:
            pere_stats[pere]["c"] += 1
            pere_stats[pere]["v"] += won
            pere_stats[pere]["p"] += placed
        if mere:
            mere_stats[mere]["c"] += 1
            mere_stats[mere]["v"] += won
            mere_stats[mere]["p"] += placed
    return dict(pere_stats), dict(mere_stats)


def get_pedigree_score(pere, mere, pere_stats, mere_stats):
    """
    Score 0-100 basé sur la réussite des descendants du père (poids 70%)
    et de la mère (poids 30%).
    """
    def bucket_score(bucket, min_count=10):
        if not bucket or bucket["c"] < min_count:
            return None
        c, v, p = bucket["c"], bucket["v"], bucket["p"]
        tv = v / c
        tp = p / c
        confiance = min(1.0, c / 100)  # confiance max à 100 descendants
        raw = tv * 250 + tp * 60
        return min(100, raw * confiance + 40 * (1 - confiance))

    s_p = bucket_score(pere_stats.get(pere))
    s_m = bucket_score(mere_stats.get(mere))

    if s_p is None and s_m is None:
        return 50
    if s_p is None:
        return s_m
    if s_m is None:
        return s_p
    return s_p * 0.70 + s_m * 0.30


# ============================================================
#  Score Corde (numéro de départ)
# ============================================================
def get_corde_score(num_pmu, nb_partants, type_corde, discipline):
    """
    Le numéro de corde a un impact selon :
      - le type de piste (corde droite / gauche / aucune)
      - la discipline (plat : impact fort, attelé : impact moyen)
    Numéros bas = corde intérieure = souvent avantage en attelé/plat.
    """
    if not num_pmu or not nb_partants:
        return 50

    # Position relative (0 = corde intérieure, 1 = extérieure)
    rel = (num_pmu - 1) / max(nb_partants - 1, 1)

    # Discipline : attelé = corde modérément importante, plat = très importante
    if discipline == "ATTELE":
        # En attelé, numéros bas avantagés mais moins fort
        score = 65 - rel * 25  # de 65 (corde 1) à 40 (dernier)
    elif discipline == "PLAT":
        # En plat, gros avantage corde intérieure
        score = 70 - rel * 35  # de 70 à 35
    elif discipline in ("MONTE", "HAIES", "STEEPLE-CHASE", "CROSS"):
        # Moins critique sur obstacles
        score = 55 - rel * 10
    else:
        score = 50

    # Si pas de corde définie (course sans corde), neutre
    if type_corde == "CORDE_AUCUNE" or not type_corde:
        score = 50

    return max(0, min(100, score))


# ============================================================
#  Score équipements (œillères, déferrage)
# ============================================================
def get_equipment_score(oeilleres, deferre, prev_oeilleres=None, prev_deferre=None):
    """
    Détecte les changements d'équipement (souvent signal d'intention).
    """
    score = 50
    # Premières œillères : souvent un boost
    if oeilleres and oeilleres != "SANS_OEILLERES":
        score += 5
        if prev_oeilleres == "SANS_OEILLERES":
            score += 10  # changement = signal d'intention
    # Déferrage complet = recherche de la perf max
    if deferre == "DEFERRE_DES_4":
        score += 12
    elif deferre in ("DEFERRE_ANTERIEURS", "DEFERRE_POSTERIEURS"):
        score += 5
    if prev_deferre and prev_deferre != deferre and "DEFERRE" in (deferre or ""):
        score += 5  # changement vers du déferrage
    return min(100, max(0, score))


# ============================================================
#  Détection de régime (spécialiste) via commentaires de course
# ============================================================
# Mots-clés indiquant des profils
PROFIL_KEYWORDS = {
    "attaquant": ["s'est élancé", "a pris la tête", "en tête", "a mené", "tenu la corde",
                  "a impulsé", "à l'aise en tête", "a fait l'allure"],
    "finisseur": ["dans la ligne droite", "fini fort", "a remonté", "dans les derniers mètres",
                  "ligne d'arrivée", "a coiffé", "battu sur le poteau", "in extremis"],
    "fragile": ["s'est galopé", "fauté", "disqualifié", "a faibli", "ne s'est pas employé",
                "a perdu ses fers", "distancé", "a chuté"],
    "regulier": ["dans le peloton", "à mi-parcours", "a suivi", "régulier", "honorable"],
}


def detect_profile(perfs_detail, comment_text=None):
    """
    Analyse les commentaires des courses passées pour détecter le profil du cheval.
    Retourne un dict avec scores par profil (0-100).
    """
    counters = defaultdict(int)
    total = 0

    # Récupère tous les commentaires des courses passées
    for course in (perfs_detail or [])[:5]:
        for p in course.get("participants", []):
            if p.get("itsHim"):
                comment = (p.get("commentaire") or {}).get("texte", "")
                if comment:
                    total += 1
                    cl = comment.lower()
                    for profil, keywords in PROFIL_KEYWORDS.items():
                        for kw in keywords:
                            if kw in cl:
                                counters[profil] += 1
                                break

    # Commentaire de la course courante si dispo
    if comment_text:
        total += 1
        cl = comment_text.lower()
        for profil, keywords in PROFIL_KEYWORDS.items():
            for kw in keywords:
                if kw in cl:
                    counters[profil] += 1
                    break

    if total == 0:
        return {"attaquant": 50, "finisseur": 50, "fragile": 50, "regulier": 50}

    return {p: min(100, (counters[p] / total) * 100 * 2.5) for p in PROFIL_KEYWORDS}


def get_profile_match_score(profil, distance, nb_partants):
    """
    Match entre le profil du cheval et le profil idéal pour cette course.
    Sprint court (< 2000m) : avantage attaquants
    Endurance (> 2700m) : avantage finisseurs
    Course peu nombreuse : régulier OK
    """
    if not profil:
        return 50

    score = 50
    # Sprint -> attaquant
    if distance and distance < 2000:
        score += (profil.get("attaquant", 50) - 50) * 0.5
        score -= (profil.get("fragile", 0) * 0.3)
    # Endurance -> finisseur
    elif distance and distance > 2700:
        score += (profil.get("finisseur", 50) - 50) * 0.5
        score -= (profil.get("fragile", 0) * 0.3)
    else:
        # Distance moyenne : équilibre
        score += (profil.get("regulier", 50) - 50) * 0.3
        score -= (profil.get("fragile", 0) * 0.2)

    # Pénalité fragilité dans tous les cas
    score -= profil.get("fragile", 0) * 0.15

    return max(0, min(100, score))
