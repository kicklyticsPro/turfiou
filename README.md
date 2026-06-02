# 🐎 Turf Analyzer v7 — Plateforme PMU + Backtest Walk-Forward rigoureux

Plateforme web professionnelle d'analyse hippique PMU avec **6 modèles ML** (GBM, RF, XGBoost-like, MLP Neural Network, Ensemble, Stacking), **AutoML**, **explainability**, multi-paris, Kelly, alertes, dashboard ROI, SQLite persistante et — **nouveau en v7 — un backtest temporel walk-forward sans look-ahead** pour mesurer la performance *réelle*.

## 🔬 Nouveautés v7 — Walk-Forward Analysis

### 2️⃣7️⃣ **Backtest temporel rigoureux** — `lib/walk_forward.py` + `/api/backtest/walkforward`

Le backtest classique (v1→v6) souffrait d'un **look-ahead bias** : les stats (Elo, forme…) étaient calculées sur les 180 derniers jours *à partir d'aujourd'hui*, puis appliquées à des courses passées → le modèle « voyait » le futur, gonflant les chiffres.

Le **walk-forward** corrige cela. Pour chaque fenêtre de test :
1. Les stats sont recalculées **uniquement jusqu'à la veille** de la période de test (`ref_date` paramétrable).
2. Le modèle ML est entraîné **uniquement sur les courses antérieures** à la période de test.
3. L'évaluation se fait sur la fenêtre de test, puis la fenêtre **glisse** vers le futur.

- Modes **rolling** (fenêtre fixe) et **expanding** (train cumulatif)
- `gap` configurable (purge anti-fuite entre train et test)
- Métriques par fold **et** agrégées : Top1, Top3, AUC, log-loss, Brier, ROI Kelly, winrate value bets
- **Écart-type des Top1** entre folds → mesure de **stabilité** du modèle
- Nouvelle section UI **🔬 Walk-Forward** dans `/backtest`

> ⚠️ **Effet de réalité** : les chiffres walk-forward sont nettement plus bas que le backtest classique (ex. ~25% Top1 au lieu de ~67%). C'est **normal et sain** — ce sont les vrais chiffres « comme en conditions réelles ». Le cache de stats est désactivé pendant le walk-forward (chaque fold recalcule les stats 180j), d'où une durée de ~3-5 min/fold.

## 📊 Performances ML mesurées (6007 échantillons)

| Modèle | Log-loss ↓ | AUC ↑ | Brier ↓ | Verdict |
|---|---|---|---|---|
| 🏆 **XGBoost-like** | **0.1179** | **0.978** | **0.0348** | Meilleur global |
| 🧠 Neural Network (MLP) | 0.1382 | 0.963 | 0.0420 | Excellent #2 |
| 🌳 Random Forest | 0.1508 | 0.970 | 0.0447 | Robuste |
| 📈 Gradient Boosting | 0.2323 | 0.921 | 0.0635 | Baseline |

## 🎯 Nouveautés v6

### 2️⃣1️⃣ **Multi-Layer Perceptron (MLP)** — `lib/neural_net.py`
- Réseau de neurones 2 couches cachées (32 → 16 par défaut)
- Activation ReLU + sortie sigmoïde
- **Adam optimizer** avec moments
- **Dropout** (0.2 par défaut) pour régulariser
- **Early stopping** sur validation set
- He initialization

### 2️⃣2️⃣ **AutoML** — `lib/automl.py`
- **Random search** sur hyperparamètres
- **Cross-validation** k-fold (temporel ou random)
- Métriques avancées : log-loss, AUC, Brier, calibration curve

### 2️⃣3️⃣ **Stacking Ensemble** — méta-modèle
- Combine XGBoost + Random Forest + MLP
- **Méta-modèle logistic** qui apprend les poids optimaux
- Train/val split 80/20 pour éviter l'overfitting du méta

### 2️⃣4️⃣ **Explainability (SHAP-like)**
- Bouton **🔍 Expliquer la prédiction** sur chaque cheval
- Top 10 features qui impactent le pronostic
- Méthode par perturbation (chaque feature → valeur neutre 50)

### 2️⃣5️⃣ **Métriques avancées**
- **Log-loss** : erreur logarithmique (calibration)
- **AUC** : qualité de discrimination
- **Brier score** : précision des probabilités
- **Calibration curve** : 10 bins de comparaison prédit/observé

### 2️⃣6️⃣ **Page Modèles** (`/models`)
- Comparaison automatique des 4 modèles
- Tableau de scores avec badge "🏆 BEST"
- AutoML pour MLP (8 configurations testées par CV)
- Guide explicatif de chaque modèle

## 🏗️ Architecture

```
turf-analyzer/
├── app.py                  # Backend Flask v6 (~1700 lignes)
├── lib/
│   ├── ml_models.py        # GBM, Random Forest, Ensemble
│   ├── xgb_like.py         # XGBoost-like régularisé
│   ├── neural_net.py       # NEW v6 : MLP NumPy pur
│   ├── automl.py           # NEW v6 : métriques, CV, random search, Stacking
│   ├── kelly.py            # Kelly Criterion
│   ├── features_v4.py      # Pedigree, corde, équipements, profils
│   ├── multi_paris.py       # Placé, couplé, tiercé
│   ├── walk_forward.py      # NEW v7 : fenêtres temporelles + agrégation (sans fuite)
│   ├── db.py               # SQLite (bets, watchlist, alerts)
│   ├── bets_tracker.py     # v4 (compatibilité migration)
│   └── geny_scraper.py     # Scraping terrain/météo
├── templates/
│   ├── index.html          # Analyse + Kelly + Combos + Explain
│   ├── backtest.html       # Backtest & entraînement (6 types ML)
│   ├── paris.html          # Tracking SQLite des paris
│   ├── dashboard.html      # KPIs + graphique profit
│   └── models.html         # NEW v6 : comparaison + AutoML
└── README.md, requirements.txt, render.yaml
```

## 🧮 Algorithme v6 (toutes versions combinées)

**Score intrinsèque** (17 composantes, somme=1) :
```
forme 15% + carrière 8% + gains 7%
+ driver 9% + entraîneur 6% + distance 7%
+ cheval_stats 9% + Elo 11%
+ âge_sexe 4% + repos 4% + elo_trend 5% + confrontation 3%
+ pedigree 6% + corde 3% + equipment 2% + profile_match 1%
+ bonus contextuels (driver=entraîneur, déferré, etc.)
```

**Probabilité finale** :
```
chance = 0.55 × proba_marché + 0.45 × score_intrinsèque

Si ML actif :
chance = 0.50 × heuristique + 0.50 × Modèle (XGBoost/MLP/Stacking) calibré
```

**Décisions** :
- Value bet si `chance - proba_marché > 4%` et cote ≥ 4
- Kelly = `(p×b - q)/b × 0.25` cappé à 5% capital
- Placé via Plackett-Luce
- Couplés/tiercés : top 5 combinaisons

## 🚀 Lancement

```bash
cd turf-analyzer
pip install -r requirements.txt
python app.py
```

→ <http://localhost:5000>

⚠️ Premier lancement : ~3 min (calcul stats 180j).

## 📱 Les 5 pages

| Page | URL | Description |
|---|---|---|
| **Analyse** | `/` | 17 scores par cheval, Kelly, combos, **🔍 Explain** |
| **Backtest** | `/backtest` | Entraînement (6 types ML), backtest ROI Kelly, **🔬 Walk-Forward** |
| **Mes paris** | `/paris` | Tracking SQLite des paris réels |
| **Dashboard** | `/dashboard` | KPIs, graphique SVG profit, stats par hippo |
| 🆕 **Modèles** | `/models` | Comparaison 4 modèles + AutoML MLP |

## ⚙️ Performances techniques v6

| Opération | Temps |
|---|---|
| Stats 180j (1ère fois) | ~3 min |
| Stats (cache) | <100 ms |
| Analyse course (sans ML) | ~100 ms |
| Analyse course + Stacking | ~300 ms |
| Entraînement GBM | ~10 s |
| Entraînement RF | ~15 s |
| Entraînement XGBoost-like | ~40 s |
| Entraînement MLP | ~25 s |
| Entraînement **Stacking** | ~60 s |
| Comparaison 4 modèles | ~90 s |
| AutoML 8 configs MLP | ~3-6 min |
| Explainability 1 cheval | ~500 ms |

## 🆕 Exemple : la page Modèles en action

```
🏆 Classement modèles (6007 échantillons, 9.47% victoires) :
#1 XGBoost-like        log_loss=0.1179  AUC=0.978  ★ BEST
#2 Neural Network MLP  log_loss=0.1382  AUC=0.963
#3 Random Forest       log_loss=0.1508  AUC=0.970
#4 Gradient Boosting   log_loss=0.2323  AUC=0.921
```

**AutoML** trouvera typiquement les meilleurs hyperparams MLP :
```
🏆 Best : hidden_sizes=(32,16,8), dropout=0.2, lr=0.005
  log_loss=0.135  AUC=0.965
```

**Explainability** d'un cheval pronostiqué à 31% :
```
🔍 MARQUIS DU SAPHIR (31.62%)
  elo            : 100  → impact +18.2 pts
  pedigree       :  85  → impact +6.4 pts
  forme          :  90  → impact +5.1 pts
  marche         :  9.5 → impact +3.8 pts
  ...
```

## 🔬 Utiliser le Walk-Forward

**Via l'UI** : page `/backtest` → section **🔬 Walk-Forward** → choisir folds / train / test / gap / mode / modèle → *Lancer*.

**Via l'API** :
```bash
curl -X POST http://localhost:5000/api/backtest/walkforward \
  -H "Content-Type: application/json" \
  -d '{"n_folds":3,"train_window":21,"test_window":5,"gap":1,"model_type":"xgb","mode":"rolling","use_ml":true}'
```

Réponse (extrait) :
```json
{
  "global": {"top1_rate": 25.5, "top3_rate": 55.7, "auc": 0.757,
             "log_loss": 0.273, "brier": 0.077, "top1_std": 2.9,
             "kelly_roi": -52.5, "value_bets": 41, "vb_winrate": 2.4},
  "folds": [{"fold": 1, "train": "22/05→29/05", "test": "31/05→01/06",
             "top1_rate": 25.5, "auc": 0.757, "log_loss": 0.273}],
  "params": {"n_folds": 3, "model_type": "xgb", "mode": "rolling"}
}
```

| Paramètre | Défaut | Description |
|---|---|---|
| `n_folds` | 4 | Nombre de plis temporels (1-8) |
| `train_window` | 30 | Jours d'entraînement par fold (5-120) |
| `test_window` | 7 | Jours de test par fold (1-30) |
| `gap` | 1 | Jours de purge entre train et test (0-7) |
| `mode` | `rolling` | `rolling` (fenêtre fixe) ou `expanding` (train cumulatif) |
| `model_type` | `xgb` | `xgb`, `rf`, `gbm`, `mlp`, `ensemble` |
| `use_ml` | `true` | `false` = heuristique seule (sans modèle ML) |

## ⚠️ Disclaimer

Outil éducatif/informatif. Aucun résultat garanti. Performances backtestées sur données récentes connues ≠ performances futures réelles.

Le jeu peut être addictif : [joueurs-info-service.fr](https://www.joueurs-info-service.fr/) (**09 74 75 13 13**).
