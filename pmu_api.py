"""
pmu_api.py — Récupération des données via l'API officieuse PMU
Base URL : https://online.turfinfo.api.pmu.fr/rest/client/61/
"""

import requests
from datetime import datetime, date
from typing import Optional

BASE_URL = "https://online.turfinfo.api.pmu.fr/rest/client/61"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

TIMEOUT = 10


def _get(url: str) -> Optional[dict]:
    try:
        r = requests.get(url, headers=HEADERS, timeout=TIMEOUT)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.HTTPError as e:
        print(f"[API] HTTP error {e.response.status_code} → {url}")
        return None
    except Exception as e:
        print(f"[API] Erreur : {e} → {url}")
        return None


def get_programme(target_date: date) -> Optional[dict]:
    """Récupère le programme complet d'une journée."""
    d = target_date.strftime("%d%m%Y")
    url = f"{BASE_URL}/programme/{d}?specialisation=INTERNET"
    return _get(url)


def get_reunion(target_date: date, r_num: int) -> Optional[dict]:
    """Récupère une réunion (R1, R2…)."""
    d = target_date.strftime("%d%m%Y")
    url = f"{BASE_URL}/programme/{d}/R{r_num}?specialisation=INTERNET"
    return _get(url)


def get_participants(target_date: date, r_num: int, c_num: int) -> Optional[dict]:
    """Récupère les partants d'une course avec toutes les stats."""
    d = target_date.strftime("%d%m%Y")
    url = (
        f"{BASE_URL}/programme/{d}/R{r_num}/C{c_num}/participants"
        f"?specialisation=INTERNET"
    )
    return _get(url)


def get_rapports(target_date: date, r_num: int, c_num: int) -> Optional[dict]:
    """Récupère les cotes probables d'une course."""
    d = target_date.strftime("%d%m%Y")
    url = (
        f"{BASE_URL}/programme/{d}/R{r_num}/C{c_num}/rapports-definitifs"
        f"?specialisation=INTERNET"
    )
    data = _get(url)
    if not data:
        # Essai avec rapports probables (avant départ)
        url2 = (
            f"{BASE_URL}/programme/{d}/R{r_num}/C{c_num}/rapports"
            f"?specialisation=INTERNET"
        )
        data = _get(url2)
    return data


# ─── Parsers ────────────────────────────────────────────────────────────────

def parse_programme(data: dict) -> list[dict]:
    """Retourne la liste des réunions avec leurs courses."""
    reunions = []
    for r in data.get("programme", {}).get("reunions", []):
        reunion = {
            "num": r.get("numOfficiel", r.get("numReunion", "?")),
            "hippodrome": r.get("hippodrome", {}).get("libelleCourt", "?"),
            "pays": r.get("pays", {}).get("libelle", ""),
            "discipline": r.get("disciplinesMeres", ["?"])[0] if r.get("disciplinesMeres") else "?",
            "courses": [],
        }
        for c in r.get("courses", []):
            reunion["courses"].append({
                "num": c.get("numOrdre", c.get("numCourse", "?")),
                "libelle": c.get("libelle", "Course"),
                "heure": c.get("heureDepart", ""),
                "distance": c.get("distance", 0),
                "nb_partants": c.get("nombreDeclaresPartants", 0),
            })
        reunions.append(reunion)
    return reunions


def parse_musique(musique_str: str) -> dict:
    """
    Analyse la musique d'un cheval / jockey / entraîneur.
    Ex: "1a2p3a0m" → extrait victoires, places, abandons.
    """
    if not musique_str:
        return {"victoires": 0, "places": 0, "courses": 0, "taux_victoire": 0.0, "taux_place": 0.0}

    import re
    tokens = re.findall(r"(\d+|[a-zA-Z])", musique_str)
    positions = []
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.isdigit():
            positions.append(int(tok))
        i += 1

    nb = len(positions)
    if nb == 0:
        return {"victoires": 0, "places": 0, "courses": 0, "taux_victoire": 0.0, "taux_place": 0.0}

    victoires = sum(1 for p in positions if p == 1)
    places = sum(1 for p in positions if 1 <= p <= 3)

    return {
        "victoires": victoires,
        "places": places,
        "courses": nb,
        "taux_victoire": round(victoires / nb * 100, 1),
        "taux_place": round(places / nb * 100, 1),
        "forme_recente": positions[:5],  # 5 dernières
    }


def parse_participants(data: dict, rapports: Optional[dict] = None) -> list[dict]:
    """Parse les partants et retourne une liste de dicts enrichis."""
    participants = data.get("participants", [])
    
    # Construire index des cotes
    cotes_map = {}
    if rapports:
        for item in rapports.get("rapportsEWinner", rapports.get("rapportsSimpleGagnant", [])):
            num = item.get("numPmu") or item.get("numero")
            cote = item.get("rapport") or item.get("dividende", 0)
            if num:
                cotes_map[num] = cote

    chevaux = []
    for p in participants:
        num = p.get("numPmu", p.get("numero", 0))

        # ── Cheval
        cheval_nom = p.get("nom", "Inconnu")
        age = p.get("age", 0)
        sexe = p.get("sexe", "")
        poids = p.get("poidsConditionMonte", p.get("poids", 0))
        corde = p.get("placeCorde", p.get("numPmu", 0))
        handicap = p.get("handicapValeur", 0)

        # ── Musique cheval
        musique_cheval = p.get("musique", "")
        stats_cheval = parse_musique(musique_cheval)

        # ── Distance / Terrain
        gains = p.get("gainsParticipant", p.get("gainsCarriere", 0)) or 0
        nb_victoires_carriere = p.get("nombreVictoires", 0) or 0
        nb_courses_carriere = p.get("nombreCourses", 0) or 0

        # ── Jockey / Driver
        jockey = p.get("jockey") or p.get("driver") or {}
        jockey_nom = jockey.get("nom", "") if isinstance(jockey, dict) else str(jockey)
        musique_jockey = jockey.get("musique", "") if isinstance(jockey, dict) else ""
        stats_jockey = parse_musique(musique_jockey)

        # ── Entraîneur
        entraineur = p.get("entraineur") or {}
        entraineur_nom = entraineur.get("nom", "") if isinstance(entraineur, dict) else str(entraineur)
        musique_entraineur = entraineur.get("musique", "") if isinstance(entraineur, dict) else ""
        stats_entraineur = parse_musique(musique_entraineur)

        # ── Pedigree
        pere = p.get("pere", "")
        mere = p.get("mere", "")

        # ── Cote
        cote = cotes_map.get(num, p.get("rapportProbable", 0) or 0)

        chevaux.append({
            "num": num,
            "nom": cheval_nom,
            "age": age,
            "sexe": sexe,
            "poids": poids,
            "corde": corde,
            "handicap": handicap,
            "gains_carriere": gains,
            "nb_victoires_carriere": nb_victoires_carriere,
            "nb_courses_carriere": nb_courses_carriere,
            "musique": musique_cheval,
            "stats_cheval": stats_cheval,
            "jockey": jockey_nom,
            "stats_jockey": stats_jockey,
            "entraineur": entraineur_nom,
            "stats_entraineur": stats_entraineur,
            "pere": pere,
            "mere": mere,
            "cote": cote,
        })

    return chevaux
