"""
app.py — Serveur Flask pour PMU Predictor
Lance avec : python app.py
Accès : http://localhost:5000
"""

from flask import Flask, render_template, jsonify, request
from datetime import date, datetime
import pmu_api
import scorer

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/programme")
def api_programme():
    """Retourne le programme du jour (ou date choisie)."""
    date_str = request.args.get("date", "")
    try:
        if date_str:
            target = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            target = date.today()
    except ValueError:
        return jsonify({"error": "Format de date invalide (YYYY-MM-DD)"}), 400

    data = pmu_api.get_programme(target)
    if not data:
        return jsonify({"error": "Impossible de récupérer le programme PMU. Vérifiez la connexion."}), 503

    reunions = pmu_api.parse_programme(data)
    return jsonify({"date": target.isoformat(), "reunions": reunions})


@app.route("/api/analyse")
def api_analyse():
    """Analyse une course et retourne le scoring des partants."""
    date_str = request.args.get("date", "")
    r_num = request.args.get("r", type=int)
    c_num = request.args.get("c", type=int)
    distance = request.args.get("distance", 2000, type=int)

    if not r_num or not c_num:
        return jsonify({"error": "Paramètres r (réunion) et c (course) requis"}), 400

    try:
        if date_str:
            target = datetime.strptime(date_str, "%Y-%m-%d").date()
        else:
            target = date.today()
    except ValueError:
        return jsonify({"error": "Format de date invalide"}), 400

    # Récupération des données
    data_participants = pmu_api.get_participants(target, r_num, c_num)
    if not data_participants:
        return jsonify({"error": f"Impossible de récupérer les partants (R{r_num}C{c_num})"}), 503

    data_rapports = pmu_api.get_rapports(target, r_num, c_num)

    # Parsing
    chevaux = pmu_api.parse_participants(data_participants, data_rapports)
    if not chevaux:
        return jsonify({"error": "Aucun partant trouvé dans les données"}), 404

    # Analyse
    resultat = scorer.analyser_course(chevaux, distance=distance)
    resultat["meta"] = {
        "date": target.isoformat(),
        "reunion": r_num,
        "course": c_num,
        "distance": distance,
        "nb_partants": len(chevaux),
    }

    return jsonify(resultat)


if __name__ == "__main__":
    print("=" * 50)
    print("  🐎  PMU Predictor — Démarrage")
    print("  → http://0.0.0.0:5000  (accessible via IP du VPS)")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=False)
