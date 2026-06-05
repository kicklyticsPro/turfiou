# 🐎 PMU Predictor

Outil d'analyse et de prédiction des courses hippiques PMU.
Données récupérées en temps réel via l'API officieuse PMU.

## Installation

```bash
pip install -r requirements.txt
```

## Lancement

```bash
python app.py
```
Puis ouvrir → http://localhost:5000

## Utilisation

1. **Choisir une date** (par défaut : aujourd'hui)
2. Cliquer **📋 Programme** → charge toutes les réunions du jour
3. Sélectionner une **Réunion** puis une **Course**
4. Cliquer **🔍 Analyser** → scoring complet des partants

## Critères de scoring (pondération)

| Critère              | Poids |
|----------------------|-------|
| Forme cheval (musique) | 30%  |
| Stats jockey         | 25%   |
| Stats entraîneur     | 15%   |
| Corde / Distance     | 15%   |
| Poids / Charge       | 10%   |
| Gains carrière       | 5%    |

## Architecture

```
pmu_predictor/
├── app.py          # Serveur Flask (routes API + HTML)
├── pmu_api.py      # Client API PMU + parsers
├── scorer.py       # Moteur de scoring multi-critères
├── templates/
│   └── index.html  # Interface web
└── requirements.txt
```

## ⚠️ Note

L'API PMU utilisée est officieuse (non documentée publiquement).
Elle peut changer sans préavis. En cas d'erreur, vérifiez d'abord
votre connexion internet et réessayez.
