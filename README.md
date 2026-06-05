# 🐎 Turf Analyzer v4 — Plateforme de pronostics PMU pro

Plateforme web complète d'analyse des courses hippiques PMU avec **% de chance**, **Kelly Criterion**, **cotes live**, **tracking des paris**, et **Ensemble Machine Learning** (Gradient Boosting + Random Forest).

## 📊 Performances mesurées (backtest 280 courses)

| Métrique | v1 | v2 | v3 | **v4 ML** |
|---|---|---|---|---|
| Taux #1 gagne | 32.5% | 42.9% | 60% | **63.2%** |
| Top 3 | 64.6% | 76.4% | 90% | **90%** |
| ROI sur #1 (mise fixe) | -37.5% | -31.5% | +76.8% | **+75.3%** |
| Value bets winrate | 1.6% | 6.8% | 32.6% | **37.9%** |
| Value bets ROI | -59% | +70.5% | +271% | **+301%** |
| **🆕 Kelly ROI** | — | — | — | **+365.8%** 💰 |
| **🆕 Kelly profit (capital 100€)** | — | — | — | **+2221€** |

> ⚠️ Backtest sur données récentes connues. Performance réelle moindre car le marché ajuste les cotes.

## 🎯 Toutes les améliorations cumulées (14 features algorithmiques)

### v1 (baseline)
1. Cotes marché normalisées
2. Musique parsée pondérée
3. Stats carrière (V/P)
4. Gains moyens (log)

### v2
5. **Stats team multi-niveaux** : driver/entraîneur × 30j × discipline × hippodrome
6. **Stats cheval enrichies** : global × avec ce driver × hippodrome × discipline
7. **Forme enrichie** : date réelle, allocation, RK, écart
8. **Système Elo dynamique** (K=16)

### v3
9. **Gradient Boosting** (50 arbres) au lieu de régression logistique
10. **Calibration isotone** (Pool-Adjacent-Violators)
11. **Features cheval avancées** : âge×sexe, repos, tendance Elo, confrontations directes
12. **Historique 180 jours**

### v4 (NOUVEAU)
13. **Ensemble GBM + Random Forest** (réduit la variance)
14. **Pedigree** : taux de réussite des descendants du père/mère
15. **Numéro de corde** : avantage selon position de départ × discipline
16. **Équipements** : œillères, déferrage (intentions du entraîneur)
17. **Détection de profils** : attaquant / finisseur / fragile via commentaires
18. **Kelly Criterion** : sizing optimal des mises (fraction 1/4)
19. **Cotes live** : refresh auto toutes les 30s
20. **Tracking des paris réels** : page "Mes paris" avec ROI réel

## 🧮 Algorithme

```
chance = 0.55 × proba_marché + 0.45 × score_intrinsèque   (heuristique)

Si ML actif :
chance = 0.5 × heuristique + 0.5 × Ensemble (60% GBM + 40% RF) calibré

score_intrinsèque (v4, 17 composantes) :
  15% forme  +  11% Elo  +  9% driver  +  9% cheval_stats
  +  8% carrière  +  7% gains  +  7% distance  +  6% entraîneur
  +  6% pedigree  +  5% elo_trend  +  4% repos  +  4% âge_sexe
  +  3% confrontation  +  3% corde  +  2% equipment  +  1% profile
```

### Kelly Criterion (mise optimale)
```
f* = (p × b - q) / b × 0.25   (quart-Kelly pour réduire la volatilité)
mise = capital × f*            (cap à 5% du capital)
```

## 🚀 Installation & lancement

```bash
cd turf-analyzer
pip install -r requirements.txt
python app.py
```

→ http://localhost:5000

⚠️ **Premier lancement** : ~3 min pour calculer les stats sur 180 jours. Tout est ensuite en cache 24h.

## 📱 Les 3 pages

### `/` — Analyse de course
- Sélecteur de date + réunions / courses
- Pour chaque cheval : 17 scores détaillés, badges, pedigree, Kelly recommandé
- Toggle **🔴 Live** : refresh auto cotes toutes les 30s
- Toggle **🧠 ML** : active l'ensemble Gradient Boosting + Random Forest
- Champ **💰 Capital** : ajuste les mises Kelly recommandées
- Bouton **📋 Enregistrer ce pari** sur chaque cheval analysé

### `/backtest` — Backtest & ML
- Entraînement du modèle (Ensemble / GBM seul / RF seul)
- Backtest sur 3-30 jours
- Stats Kelly détaillées (mise, gain, profit, ROI)
- Top 30 drivers / entraîneurs

### `/paris` — Mes paris (NEW v4)
- Liste des paris en attente
- Boutons ✅ Gagné / ❌ Perdu pour résoudre
- Statistiques globales : winrate, ROI réel, profit cumulé
- Historique complet

## ⚙️ Performances

| Opération | Temps |
|---|---|
| Construction stats v4 (180j) | ~3 min |
| Stats v4 (cache) | <100 ms |
| Analyse 1 course | ~100 ms |
| Backtest 5j (280 courses) | ~13 s |
| Entraînement Ensemble (15j) | ~42 s |
| Refresh live | ~1 s |

## 🔧 Cache (`/tmp/turf_cache/`)

- `stats_team_v4.pkl` — drivers/entraîneurs multi-niveaux
- `horse_stats_v4.pkl` — stats par cheval
- `elo_v4.pkl`, `elo_hist_v4.pkl` — Elo + historique
- `horse_races_v4.pkl` — historique courses (repos, confrontations)
- `pedigree_v4.pkl` — stats pères/mères
- `ml_ensemble_v4.pkl` — modèle Ensemble
- `calibration_v4.pkl` — table de calibration
- `bets_v4.json` — paris enregistrés (persistant !)

## 🆕 Exemple concret v4

Sur la course **PRIX DE BLOMARD (Vichy)** d'aujourd'hui :

| # | Cheval | Cote marché | Chance v4 ML | Edge | Kelly | EV |
|---|---|---|---|---|---|---|
| 1 | MARQUIS DU SAPHIR ⭐ | 9.5 | **36.93%** | +28% | **5€** | +251% |
| 2 | MARIO MASCAR | 1.5 (favori) | 26.25% | -30% | 0€ | -61% |
| 3 | MISTER ROYAL | 6.7 | 12.25% | -0.4% | 0€ | -18% |
| 5 | MOKY BERRY ⭐ | 62 | 4% | +2.6% | 0.6€ | +147% |

L'algo détecte que le **favori du marché (1.5)** est sur-coté et que **MARQUIS DU SAPHIR (9.5)** est sous-coté → recommande de miser **5€** dessus (sur capital 100€).

## ⚠️ Disclaimer

Outil **à but éducatif/informatif** uniquement. Aucun résultat garanti. Le marché ajuste constamment ses cotes : les performances backtestées ne préjugent pas du futur.

Le jeu peut être addictif : [joueurs-info-service.fr](https://www.joueurs-info-service.fr/) (**09 74 75 13 13**).
