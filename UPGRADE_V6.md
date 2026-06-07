# Upgrade v6 — Dual Model Win + Top 4 + 14 Features

## Résumé

**2 changements majeurs** dans cette version :

### 1. Modèle DUAL Win + Top 4
Avant : 1 seul modèle (y=1 si gagnant → ~8% positifs)
Maintenant : **2 modèles entraînés en parallèle**

| Modèle | Label | % Positifs | Usage |
|--------|-------|------------|-------|
| **WIN** | `y=1 si place == 1` | ~8% | Classement, proba gagnant |
| **TOP4** | `y=1 si place ≤ 4` | ~40% | Probabilité placement |

**Pourquoi c'est mieux :**
- **4× plus de signal** : le modèle Top 4 apprend sur 40% des données au lieu de 8%
- **Moins de variance** : un favori 3ème n'est plus "0" pour le modèle
- **Meilleure calibration** : les probas Top 4 sont intrinsèquement plus fiables
- **Complémentaire** : Win capture le pic de performance, Top 4 capture la fiabilité

### 2. 14 nouvelles features ML (27 → 41)

Même vecteur de features pour les deux modèles.

## Architecture

```
                    ┌──────────────────┐
  41 features ─────►│   Modèle WIN     │──► chance (proba gagnant, normalisée)
                    │  (stacking 5 ML) │
                    └──────────────────┘
                    
                    ┌──────────────────┐
  41 features ─────►│  Modèle TOP4     │──► chanceTop4 (proba placement, absolu)
                    │  (stacking 5 ML) │
                    └──────────────────┘

  Chance Win   = 20% heuristique + 80% ML_win     (normalisée entre chevaux)
  Chance Top 4 = 20% heuristique + 80% ML_top4    (absolu, pas de normalisation)
```

## Fichiers modifiés

| Fichier | Changement |
|---------|------------|
| `app.py` | +375 lignes : dual training, dual prediction, backtest top4, 14 features |
| `lib/features_v4.py` | +271 lignes : 6 nouvelles fonctions de features |
| `UPGRADE_V6.md` | Ce fichier |

## Détail du modèle dual

### Training (`train_ml_model`)
```
Pour chaque course historique :
  X.append(featurize(cheval, nb_partants))
  y_win.append(1 if place == 1 else 0)      # ~8% positifs
  y_top4.append(1 if 1 <= place <= 4 else 0) # ~40% positifs

Entraîner stacking(X, y_win)  → ml_advanced_v5.pkl
Entraîner stacking(X, y_top4) → ml_top4_advanced_v6.pkl
```

### Prediction (`analyser_course`)
```python
# WIN
chance = 0.2 * heuristique + 0.8 * ML_win    # normalisé à 100%

# TOP4 (nouveau)
chanceTop4 = 0.2 * heuristique_top4 + 0.8 * ML_top4  # proba absolue
```

### Réponse API enrichie
```json
{
  "chanceTop4": 62.4,        // proba top 4 blendée (20% heur + 80% ML)
  "chanceTop4ML": 65.2,      // proba top 4 ML pure
  "chanceTop4Heur": 50.0,    // proba top 4 heuristique seule
  "chance": 18.5,            // proba gagnant (normalisée)
  "chanceML": 19.2,          // proba gagnant ML pure
  "ml_top4_active": true     // flag modèle top4 chargé
}
```

### Backtest enrichi
```json
{
  "top4_ml_accuracy": 68.5,       // accuracy globale top4
  "top1_top4_rate": 45.2,         // le #1 algo est dans le top 4
  "top4_by_confidence": {
    "high":   {"hit": 120, "total": 165, "accuracy": 72.7},  // proba ≥ 60%
    "medium": {"hit": 200, "total": 340, "accuracy": 58.8},  // 35-60%
    "low":    {"hit": 50,  "total": 150, "accuracy": 33.3}   // < 35%
  }
}
```

## Comment utiliser

### 1. Installer
```bash
pip install -r requirements.txt
```

### 2. ⚠️ Retrainer OBLIGATOIREMENT
Les deux modèles sont entraînés en un seul appel :
```bash
curl -X POST "http://localhost:5000/api/train?type=advanced&days=21"
```
→ Entraîne WIN + TOP4 en ~3-5 minutes

### 3. Vérifier
```bash
curl "http://localhost:5000/api/ml-status"
# → {"models_loaded": {"win": true, "top4": true}}
```

### 4. Utiliser
Coche "🧠 ML" → les deux modèles sont utilisés automatiquement.

## Avantages du modèle Top 4 vs Win seul

| Critère | Win seul | Win + Top 4 |
|---------|----------|-------------|
| % positifs | ~8% | ~40% pour Top4 |
| Signal/noise | Faible | 5× meilleur |
| Calibration | Difficile | Naturelle |
| Stabilité prediction | Variance élevée | Stable |
| Usage pratique | Gagnant seul | Gagnant + Placement + Couplé |
| Confiance | Binaire | Bucket haute/moyenne/basse |

## Nouveaux fichiers modèle

```
/tmp/turf_cache/
  ml_advanced_v5.pkl          # Modèle WIN (stacking)
  ml_top4_advanced_v6.pkl     # Modèle TOP4 (stacking)  ← NOUVEAU
  ml_ensemble_v4.pkl          # Fallback WIN (numpy)
  ml_top4_ensemble_v6.pkl     # Fallback TOP4 (numpy)   ← NOUVEAU
  calibration_v4.pkl          # Calibration isotone WIN
```
