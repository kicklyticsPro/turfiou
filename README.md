# Turf Analyzer — Version Épurée

Analyse hippique PMU basée sur **3 piliers uniquement** :

1. **Stats carrière** du cheval (courses, victoires, places → score)
2. **Stats driver/jockey** (courses, victoires, places → score)
3. **Stats entraîneur** (courses, victoires, places → score)

Le **classement** est la **moyenne simple** des 3 scores.
Le **backtest** mesure la performance historique de ce classement.

## 🆕 Moteur de calcul v8 (multiplicateurs de force)

Le scoring a été refondu pour la **précision prédictive** (voir
`SCORING_DESIGN.md` et `lib/scoring.py`). Les 3 piliers restent, mais le calcul
est désormais statistiquement fondé :

1. **Multiplicateur de victoire** normalisé par la taille du champ + shrinkage
   empirique vers la moyenne : `m = (v + κ) / (Σ 1/N_i + κ)`.
   → corrige le biais « gagner 10 % dans des champs de 18 ≠ champs de 6 ».
2. **Combinaison pondérée** cheval (45 %) / driver (35 %) / entraîneur (20 %)
   + contexte (discipline, hippodrome, forme récente via la musique).
3. **Probabilités calibrées** : `P(gagnant) = λ_i / Σλ_j` (Bradley-Terry),
   `P(top 3)` via les marginales de Harville. Somme = 100 %.
4. **Déoverround du marché** (méthode de Shin) pour un **edge** calibré :
   `edge = P_modèle − P_marché` (vrais value bets).
5. **Cote juste** = `100 / P_modèle`.

Le bucket de stats passe de `{c, v, p}` à `{c, v, p, dw, dp}` (difficultés de
taille de champ) — cache bumpé vers `stats_v8.pkl` (reconstruction automatique).

```bash
python tests/test_scoring.py      # 27 tests, dont backtest synthétique 59 % vs 41 %
```

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
