# Turf Analyzer — Version Épurée

Analyse hippique PMU basée sur **3 piliers uniquement** :

1. **Stats carrière** du cheval (courses, victoires, places → score)
2. **Stats driver/jockey** (courses, victoires, places → score)
3. **Stats entraîneur** (courses, victoires, places → score)

Le **classement** est la **moyenne simple** des 3 scores.
Le **backtest** mesure la performance historique de ce classement.

## Ce qui a été supprimé
- ❌ Machine Learning (v4/v5/v7/v8, XGBoost, LightGBM, TabNet, etc.)
- ❌ ELO ratings et tendances ELO
- ❌ Pedigree (stats père/mère)
- ❌ Corde / terrain / équipements
- ❌ Momentum, streaks, compétitivité avancée
- ❌ Kelly criterion
- ❌ Value bets complexes
- ❌ Modèles Top3 / Top4 dédiés
- ❌ Calibration isotone / Platt
- ❌ Features d'interactions
- ❌ Betfair fallback

## Ce qui reste
- ✅ API PMU pour données courses/participants/performances
- ✅ Calcul des stats brutes cheval/driver/entraineur sur 180 jours
- ✅ Scoring simple par bucket (win_rate + place_rate + confiance)
- ✅ Classement = moyenne des 3 scores
- ✅ Backtest avec suivi du #1
- ✅ Bilan quotidien par course
- ✅ Interface web minimaliste (analyse, backtest, bilan)

## Déploiement

```bash
pip install -r requirements.txt
python app.py
```

Ou sur Render/Fly.io : le fichier `requirements.txt` contient Flask + requests uniquement.

## Structure
```
app.py                          # Application principale (tout le code)
templates/
  index.html                    # Dashboard admin
  backtest.html                 # Page backtest
  bilan.html                    # Bilan quotidien
  public.html                   # Page d'accueil
  admin_login.html              # Login admin
requirements.txt                # flask, requests
README.md                       # Ce fichier
```
