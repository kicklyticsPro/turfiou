"""
v4→v6 - Nouvelles features : pedigree, corde, équipements, régimes (spécialistes).
+ v6 : gains_trend, terrain_perf, equipment_change, days_since_last,
        nb_courses_recent, corde_avantage_historique
"""
import re
from datetime import datetime, timedelta
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


# ============================================================
#  v6 — Features avancées pour le ML
# ============================================================

def compute_gains_trend(perfs_detail):
    """
    Tendance des gains sur les 5 dernières courses.
    Compare gains récents vs anciens pour détecter progression/régression.
    Score 0-100 : >50 = progression, <50 = régression, 50 = stable.
    """
    if not perfs_detail:
        return 50.0

    gains_list = []
    for course in perfs_detail[:5]:
        for p in course.get("participants", []):
            if p.get("itsHim"):
                place = (p.get("place") or {}).get("place", 0) or 0
                allocation = course.get("allocation") or 0
                if allocation <= 0:
                    gains_list.append(-1)
                    break
                if place == 1:
                    gains_list.append(allocation * 0.45)
                elif place == 2:
                    gains_list.append(allocation * 0.20)
                elif place == 3:
                    gains_list.append(allocation * 0.12)
                elif 4 <= place <= 5:
                    gains_list.append(allocation * 0.04)
                elif place > 0:
                    gains_list.append(0)
                else:
                    gains_list.append(-1)  # non classé → exclu
                break

    valid = [g for g in gains_list if g >= 0]
    if len(valid) < 2:
        return 50.0

    # Les plus récentes sont en premier dans la liste
    mid = max(1, len(valid) // 2)
    recent = valid[:mid]
    older = valid[mid:]

    avg_recent = sum(recent) / len(recent)
    avg_older = sum(older) / len(older)

    if avg_older == 0:
        return 75.0 if avg_recent > 0 else 50.0

    ratio = avg_recent / avg_older
    score = 50 + (ratio - 1) * 25
    return max(0, min(100, score))


def compute_terrain_perf(perfs_detail):
    """
    Performance moyenne selon le type de terrain rencontré.
    Un cheval qui performe sur tous les terrains obtient un bon score.
    Un cheval spécialiste d'un terrain obtient un score selon son historique.
    Score 0-100.
    """
    if not perfs_detail:
        return 50.0

    terrain_places = {}  # terrain -> [places]
    for course in perfs_detail[:12]:
        terrain = (course.get("natureTerrain")
                   or course.get("terrain")
                   or course.get("conditionPiste")
                   or "")
        if not terrain:
            terrain = "unknown"
        terrain = str(terrain).upper().strip()
        for p in course.get("participants", []):
            if p.get("itsHim"):
                place = (p.get("place") or {}).get("place", 0) or 0
                if place > 0:
                    terrain_places.setdefault(terrain, []).append(place)
                break

    if not terrain_places:
        return 50.0

    scores = []
    for terrain, places in terrain_places.items():
        avg_place = sum(places) / len(places)
        # Score : place 1 → 100, place 8 → ~16
        score = max(0, 100 - (avg_place - 1) * 12)
        scores.append(score)

    if not scores:
        return 50.0

    return sum(scores) / len(scores)


def detect_equipment_change(perfs_detail, current_oeilleres, current_deferre):
    """
    Détecte les changements d'équipement par rapport à la course précédente.
    1ère fois œillères / déferré = signal fort d'intention.
    Score 0-100 : 50 = pas de changement, >50 = changement positif.
    """
    score = 50.0

    if not perfs_detail:
        return score

    # Extraire l'équipement de la dernière course
    prev_oeilleres = None
    prev_deferre = None
    for course in perfs_detail[:1]:
        for p in course.get("participants", []):
            if p.get("itsHim"):
                prev_oeilleres = p.get("oeilleres")
                prev_deferre = p.get("deferre")
                break

    # --- Changement d'œillères ---
    cur_has_oeil = current_oeilleres and current_oeilleres != "SANS_OEILLERES"
    prev_has_oeil = prev_oeilleres and prev_oeilleres != "SANS_OEILLERES"

    if cur_has_oeil and not prev_has_oeil:
        score += 15  # Premières œillères = signal fort
    elif not cur_has_oeil and prev_has_oeil:
        score -= 5   # Retrait d'œillères

    # --- Changement de déferrage ---
    if current_deferre != prev_deferre and prev_deferre is not None:
        cur_def = "DEFERRE" in (current_deferre or "")
        prev_def = "DEFERRE" in (prev_deferre or "")

        if cur_def and not prev_def:
            score += 12  # Nouveau déferrage = intention
        if current_deferre == "DEFERRE_DES_4" and prev_deferre != "DEFERRE_DES_4":
            score += 8   # Passage déferrage complet
        if not cur_def and prev_def:
            score -= 5   # Re-ferrage (perte de vitesse potentielle)

    return max(0, min(100, score))


def compute_days_since_last(perfs_detail):
    """
    Jours depuis la dernière course (score normalisé 0-100).
    Fraîcheur vs manque de rythme :
      0-7j  → 85 (très frais)
      8-14j → 75 (optimal)
      15-21j → 65
      22-35j → 55
      36-60j → 40 (manque de rythme)
      60j+   → 25 (trop long)
    """
    if not perfs_detail:
        return 50

    today = datetime.now()
    for course in perfs_detail[:1]:
        date_ms = course.get("date")
        if date_ms:
            d = datetime.fromtimestamp(date_ms / 1000)
            days = (today - d).days
            if days < 0:
                return 50
            if days <= 7:
                return 85
            elif days <= 14:
                return 75
            elif days <= 21:
                return 65
            elif days <= 35:
                return 55
            elif days <= 60:
                return 40
            elif days <= 90:
                return 30
            else:
                return 20
    return 50


def compute_nb_courses_recent(perfs_detail, days=30):
    """
    Nombre de courses dans les N derniers jours (score normalisé 0-100).
    Charge de travail / fatigue cumulative :
      0 courses → 40 (manque de rythme)
      1         → 65
      2         → 80 (optimal)
      3         → 75
      4         → 55 (fatigue)
      5+        → 30 (suralerté)
    """
    if not perfs_detail:
        return 50

    today = datetime.now()
    cutoff = today - timedelta(days=days)

    count = 0
    for course in perfs_detail:
        date_ms = course.get("date")
        if not date_ms:
            continue
        d = datetime.fromtimestamp(date_ms / 1000)
        if d >= cutoff:
            count += 1

    if count == 0:
        return 40
    elif count == 1:
        return 65
    elif count == 2:
        return 80
    elif count == 3:
        return 75
    elif count == 4:
        return 55
    elif count == 5:
        return 40
    else:
        return 30


def compute_corde_avantage(perfs_detail):
    """
    Avantage corde historique : ce cheval performe-t-il mieux
    quand il part en position interne (petit numéro) ?
    Score 0-100 : >50 = bon en corde interne.
    """
    if not perfs_detail:
        return 50

    corde_perfs = []  # (position_relative, place)
    for course in perfs_detail[:10]:
        nb_parts = course.get("nbParticipants") or 0
        if nb_parts < 4:
            continue
        for p in course.get("participants", []):
            if p.get("itsHim"):
                num = p.get("numPmu") or 0
                place = (p.get("place") or {}).get("place", 0) or 0
                if num > 0 and place > 0:
                    corde_perfs.append((num / nb_parts, place))
                break

    if len(corde_perfs) < 3:
        return 50

    # Séparer corde interne (≤40% du peloton) vs externe
    inside = [p for rel, p in corde_perfs if rel <= 0.4]
    outside = [p for rel, p in corde_perfs if rel > 0.4]

    if not inside or not outside:
        return 50

    avg_inside = sum(inside) / len(inside)
    avg_outside = sum(outside) / len(outside)

    # Si meilleur en interne → score élevé (avantage corde confirmé)
    if avg_inside < avg_outside:
        return min(100, 55 + (avg_outside - avg_inside) * 5)
    else:
        return max(0, 45 - (avg_inside - avg_outside) * 5)
