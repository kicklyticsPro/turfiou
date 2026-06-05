"""
v4 - Tracking des paris réels.
Stocke les paris dans un JSON, calcule le ROI réel.
"""
import json
import os
from datetime import datetime


def load_bets(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except Exception:
        return []


def save_bets(path, bets):
    try:
        with open(path, "w") as f:
            json.dump(bets, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Bets save error: {e}")


def add_bet(path, bet):
    """bet = {date, course, cheval, cote, mise, type ('simple_gagnant' etc.)}"""
    bets = load_bets(path)
    bet["id"] = int(datetime.now().timestamp() * 1000)
    bet["created_at"] = datetime.now().isoformat()
    bet["statut"] = "EN_ATTENTE"
    bet["gain"] = 0
    bets.append(bet)
    save_bets(path, bets)
    return bet


def update_bet_result(path, bet_id, gagne, place=None):
    bets = load_bets(path)
    for b in bets:
        if b["id"] == bet_id:
            b["statut"] = "GAGNE" if gagne else "PERDU"
            b["place"] = place
            b["gain"] = round(b["mise"] * b["cote"], 2) if gagne else 0
            b["resolved_at"] = datetime.now().isoformat()
            break
    save_bets(path, bets)


def delete_bet(path, bet_id):
    bets = load_bets(path)
    bets = [b for b in bets if b["id"] != bet_id]
    save_bets(path, bets)


def compute_stats(bets):
    """Calcule les statistiques globales du portefeuille."""
    if not bets:
        return {"total_bets": 0, "resolved": 0, "wins": 0, "winrate": 0,
                "total_mise": 0, "total_gain": 0, "roi": 0, "profit": 0,
                "en_attente": 0}

    resolved = [b for b in bets if b["statut"] in ("GAGNE", "PERDU")]
    wins = [b for b in resolved if b["statut"] == "GAGNE"]
    en_attente = [b for b in bets if b["statut"] == "EN_ATTENTE"]

    total_mise = sum(b["mise"] for b in resolved)
    total_gain = sum(b.get("gain", 0) for b in resolved)
    profit = total_gain - total_mise
    roi = (profit / total_mise * 100) if total_mise else 0

    return {
        "total_bets": len(bets),
        "resolved": len(resolved),
        "wins": len(wins),
        "winrate": round(len(wins) / len(resolved) * 100, 2) if resolved else 0,
        "total_mise": round(total_mise, 2),
        "total_gain": round(total_gain, 2),
        "profit": round(profit, 2),
        "roi": round(roi, 2),
        "en_attente": len(en_attente),
    }
