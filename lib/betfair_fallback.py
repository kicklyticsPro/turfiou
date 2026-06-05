"""
betfair_fallback.py - API alternative quand PMU fail
Utilise les endpoints publics Betfair (gratuits, sans clé)
"""
import requests
from datetime import datetime

HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; TurfAnalyzer/5.0)"}

BETFAIR_RESULTS = "https://betfair-data-supplier-prod.herokuapp.com/api/daily_racing_results"
BETFAIR_RACECARD = "https://www.betfair.com/rest/v2/raceCard"

def get_betfair_results(date_str):
    """
    date_str format PMU: ddmmyyyy -> convert to yyyy-mm-dd
    """
    try:
        dt = datetime.strptime(date_str, "%d%m%Y")
        iso = dt.strftime("%Y-%m-%d")
        r = requests.get(f"{BETFAIR_RESULTS}?date={iso}", headers=HEADERS, timeout=10)
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        print(f"Betfair fallback error: {e}")
    return None

def get_betfair_racecard(date_str):
    """Alternative via Betfair racecard (public)"""
    try:
        # Betfair utilise timestamp
        dt = datetime.strptime(date_str, "%d%m%Y")
        # Pas d'API directe sans marketId, on retourne None
        return None
    except:
        return None

def convert_betfair_to_pmu_format(bf_data, date_str):
    """
    Convertit le format Betfair en format PMU attendu par l'app
    """
    if not bf_data:
        return None
    
    # Structure minimale compatible
    reunions = []
    # Betfair retourne liste de courses
    courses_by_venue = {}
    for race in bf_data.get('results', [])[:20]:  # limite
        venue = race.get('venue', 'BETFAIR')
        if venue not in courses_by_venue:
            courses_by_venue[venue] = []
        courses_by_venue[venue].append(race)
    
    num = 1
    for venue, races in courses_by_venue.items():
        courses = []
        for i, r in enumerate(races, 1):
            courses.append({
                "numOrdre": i,
                "libelle": r.get('race_name', f'Course {i}'),
                "discipline": "PLAT",  # default
                "distance": r.get('distance', 1600),
                "heureDepart": int(datetime.now().timestamp() * 1000),
                "nombreDeclaresPartants": len(r.get('runners', [])),
                "arriveeDefinitive": True,
                "ordreArrivee": [x.get('number') for x in r.get('runners', []) if x.get('position')]
            })
        reunions.append({
            "numOfficiel": num,
            "hippodrome": {"libelleCourt": venue},
            "courses": courses
        })
        num += 1
    
    return {"programme": {"reunions": reunions}}