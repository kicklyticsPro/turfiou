# Upgrade v5 → v6 — 14 nouvelles features ML

## Résumé

**27 → 41 features** dans le pipeline ML (12 features brutes + 2 interactions).

Toutes les features demandées ont été implémentées et intégrées dans les 3 couches du pipeline :
1. **Calcul** (`lib/features_v4.py`) — 6 nouvelles fonctions
2. **Scoring** (`app.py` → `analyser_course_features()`) — intégration dans le dict `scores`
3. **ML** (`app.py` → `featurize()`) — ajout au vecteur de features
4. **Heuristique** (`app.py` → `analyser_course()`) — redistribution des poids

## Features ajoutées

| # | Feature | Nom code | Impact | Source |
|---|---------|----------|--------|--------|
| 1 | Taux driver/hippodrome | `driver_hippo` | 🔴 Fort | `team_stats["drivers_hippo"]` |
| 2 | Indice de régularité | `regularite` | 🔴 Fort | `compute_regularity()` (existant) |
| 3 | Changement d'équipement | `equip_change` | 🟡 Moyen | `detect_equipment_change()` 🆕 |
| 4 | Style attaquant | `style_attaquant` | 🟡 Moyen | `detect_profile()` (existant) |
| 5 | Style finisseur | `style_finisseur` | 🟡 Moyen | `detect_profile()` (existant) |
| 6 | Penal fragile | `style_fragile` | 🟡 Moyen | `detect_profile()` (existant) |
| 7 | Tendance des gains | `gains_trend` | 🟡 Moyen | `compute_gains_trend()` 🆕 |
| 8 | Jours depuis dernière course | `jours_derniere` | 🟢 Léger | `compute_days_since_last()` 🆕 |
| 9 | Nb courses ce mois | `nb_courses_mois` | 🟢 Léger | `compute_nb_courses_recent()` 🆕 |
| 10 | Performance terrain | `perf_terrain` | 🟡 Moyen | `compute_terrain_perf()` 🆕 |
| 11 | Avantage corde historique | `corde_avantage` | 🟢 Léger | `compute_corde_avantage()` 🆕 |
| 12 | Chimie cheval/driver | `chimie_driver` | 🟡 Moyen | `horse_stats["with_driver"]` |

### Interactions ajoutées

| # | Interaction | Logique |
|---|-------------|---------|
| 13 | `regularite_x_forme` | Régularité × Forme = fiabilité globale |
| 14 | `driver_hippo_x_terrain` | Expertise locale × affinité terrain |

## Fichiers modifiés

### `lib/features_v4.py` (+220 lignes)
- `compute_gains_trend()` — Compare gains récents vs anciens sur 5 courses
- `compute_terrain_perf()` — Performance moyenne par type de terrain (BON/SOUPLE/LOURD)
- `detect_equipment_change()` — Détecte 1ères œillères, nouveau déferrage, re-ferrage
- `compute_days_since_last()` — Jours depuis dernière course (normalisé 0-100)
- `compute_nb_courses_recent()` — Nombre de courses sur 30 jours (normalisé 0-100)
- `compute_corde_avantage()` — Performance historique en corde interne vs externe

### `app.py` (+145 lignes)
- Imports mis à jour (6 nouvelles fonctions)
- `analyser_course_features()` : calcul des 12 nouvelles scores + ajout au dict `scores`
- `featurize()` : 14 nouveaux éléments dans le vecteur (27 → 41)
- `FEATURE_NAMES` : 41 noms à jour
- `analyser_course()` : poids heuristiques redistribués (29 composantes)
- `api_course()` : extraction du terrain depuis le programme PMU

## Poids heuristiques redistribués

```
Ancien total = 1.00 (16 composantes)
Nouveau total = 1.00 (28 composantes)

Anciennes (poids réduits) : 0.74
  forme 0.12, carriere 0.06, gains 0.05, driver 0.07, entraineur 0.04,
  distance 0.05, cheval_stats 0.07, elo 0.08, age_sexe 0.03, repos 0.03,
  elo_trend 0.04, confrontation 0.02, pedigree 0.04, corde 0.02,
  equipment 0.01, profile_match 0.01

Nouvelles v6 : 0.26
  driver_hippo 0.05 🔴, regularite 0.06 🔴, equip_change 0.02 🟡,
  style_attaquant 0.02 🟡, style_finisseur 0.02 🟡, gains_trend 0.02 🟡,
  jours_derniere 0.01 🟢, nb_courses_mois 0.01 🟢, perf_terrain 0.02 🟡,
  corde_avantage 0.01 🟢, chimie_driver 0.02 🟡
```

## Comment utiliser

### 1. Installer (inchangé)
```bash
cd turfiou
pip install -r requirements.txt
```

### 2. ⚠️ Retrainer le modèle OBLIGATOIRE
```bash
# Le vecteur de features a changé (27 → 41) → l'ancien modèle est incompatible
curl -X POST "http://localhost:5000/api/train?type=advanced&days=21"
```

### 3. Utiliser
Coche "🧠 ML" comme avant → le modèle v5/v6 sera utilisé automatiquement.

## Détail des nouvelles fonctions

### `compute_gains_trend(perfs_detail)`
- Compare la moyenne des gains des courses les plus récentes vs les plus anciennes
- Ratio > 1 = progression → score > 50
- Ratio < 1 = régression → score < 50
- Utilise l'allocation de la course et la place pour estimer les gains

### `compute_terrain_perf(perfs_detail)`
- Groupe les performances par type de terrain (BON/SOUPLE/LOURD/PSF)
- Calcule le score moyen par terrain
- Retourne la moyenne des scores = capacité à performer sur tous les terrains

### `detect_equipment_change(perfs_detail, current_oeilleres, current_deferre)`
- Compare l'équipement actuel avec celui de la dernière course
- Signaux détectés :
  - Premières œillères : +15
  - Nouveau déferrage : +12
  - Déferrage complet (4 fers) : +8
  - Retrait d'œillères : -5
  - Re-ferrage : -5

### `compute_days_since_last(perfs_detail)`
- Extrait la date de la course la plus récente
- Normalise en score : 0-7j → 85, 8-14j → 75, ..., 90j+ → 20

### `compute_nb_courses_recent(perfs_detail, days=30)`
- Compte les courses dans les 30 derniers jours
- Optimal : 2-3 courses (score 75-80)
- Fatigue : 5+ (score 30)
- Manque de rythme : 0 (score 40)

### `compute_corde_avantage(perfs_detail)`
- Compare la place moyenne en corde interne (≤40% du peloton) vs externe
- Si meilleur en interne → score > 50 (avantage corde confirmé)

## Compatibilité

- ✅ Rétro-compatible avec les données existantes (toutes les features ont des valeurs par défaut)
- ✅ `compute_all_stats()` inchangé — pas besoin de re-builder le cache
- ❌ Ancien modèle ML incompatible (27 features → 41) → retraining obligatoire
- ✅ Interface web inchangée
- ✅ API publique inchangée
