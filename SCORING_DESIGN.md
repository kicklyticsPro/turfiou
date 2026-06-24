# Nouveau moteur de calcul (v8) — Précision prédictive

> Objectif : améliorer le **hit-rate Top1 / Top3** via une modélisation
> statistique correcte, en restant interprétable (pas de boîte noire ML).

## Problèmes du système actuel

| # | Problème | Impact |
|---|----------|--------|
| 1 | `get_bucket_score` : `win_rate×200 + place_rate×60` → score **arbitraire**, pas une probabilité | scores non comparables, "edge" non calibré |
| 2 | `place_rate` **non normalisé** par la taille du champ | 3 placés/18 partants ≠ 3 placés/8 → biais fort |
| 3 | Régression vers constante `30` (pas la moyenne réelle) | shrinkage statistiquement faux |
| 4 | `proba_marche` = inverse-cotes normalisé, **marge bookmaker non retirée** (overround) | proba marché biaisée |
| 5 | `edge = score − proba_marche` compare une heuristique 0-100 à une proba 0-100 | "value bet" non fondé |
| 6 | Composite = **moyenne simple** des 3 piliers | aucune pondération justifiée |

## Nouveau modèle : multiplicateurs de force (Bradley-Terry)

Chaque entité (cheval / driver / entraîneur) obtient un **multiplicateur de
victoire** `m` tel que `m = 1` ⇔ entité moyenne, `m > 1` ⇔ au-dessus de la
moyenne, `m < 1` ⇔ en-dessous.

### A. Multiplicateur normalisé par la taille du champ + shrinkage Bayésien

Pour une entité ayant couru `c` courses (tailles de champ `N_i`), gagné `v`,
la "difficulté" cumulée (nombre de victoires attendues pour un concurrent
moyen) est :

```
dw = Σ_i 1 / N_i          # un cheval moyen gagne 1 fois sur N_i
```

Le **multiplicateur de victoire** (estimateur empirique-régularisé) :

```
m_win = (v + κ) / (dw + κ)
```

- Sans shrinkage (`κ→0`) : `m = v/dw`. Un cheval moyen a `v ≈ dw` → `m ≈ 1`. ✓
- Le `κ` régresse vers `m = 1` (la moyenne de population) pour les petits
  échantillons. **Plus naturel que la constante 30.**

Idem pour la **place** (top 3) avec `dp = Σ min(3, N_i)/N_i` :

```
m_place = (p + κ_p) / (dp + κ_p)
```

> ✅ Ceci corrige les problèmes **1, 2 et 3** : normalisation par la taille du
> champ, shrinkage vers la vraie moyenne de population (1.0), et produit une
> quantité interprétable (un *odds ratio*), pas un score arbitraire.

### B. Combinaison des contextes (log-space, pondérée)

On combine les multiplicateurs de plusieurs contextes (global, 30 derniers
jours, discipline, hippodrome, avec-driver, carrière PMU, musique) en espace
logarithmique (combinaison **multiplicative** des odds ratios) :

```
ln(m_total) = Σ_k  w_k · ln(m_k)        # Σ w_k = 1, contexte ignoré si absent
```

### C. Pilier cheval / driver / entraîneur → force de course

```
ln(λ_cheval) = w_H · ln(λ_cheval)        # pilier le plus prédictif
ln(λ_course) = w_H·ln(λ_cheval) + w_D·ln(λ_driver) + w_T·ln(λ_entraineur)
```

### D. Probabilités calibrées (softmax = normalisation Bradley-Terry)

```
P(win_i) = λ_i / Σ_j λ_j                 # somme = 100 %, comparable aux cotes
P(top3_i) = Harville(λ)                  # marginales de rang cohérentes
```

> ✅ Corrige **5** : les probabilités somment à 1 et sont comparables aux cotes.

### E. Déoverround du marché (méthode de Shin)

Les cotes PMU contiennent une marge (`Σ 1/cote > 1`). On applique le modèle de
**Shin (1993)** pour estimer les probabilités "vraies" `p_i` en retirant
l'overround + la prime aux favoris (trading insider) :

```
p_i(z) = (√(z² + 4(1−z)·r_i²) − z) / (2·(1−z)),   r_i = (1/cote_i) / Σ(1/cote)
z trouvé par bissection pour que Σ p_i = 1
```

### F. Edge calibré (vrai "value bet")

```
edge_i = P_modèle(win_i) − P_marché_shin(win_i)    # en points de %
cote_juste_i = 100 / P_modèle(win_i)
```

> ✅ Corrige **5 et 6** : l'edge compare deux **probabilités**, et la pondération
> des piliers est explicite et réglable.

### G. Score d'affichage 0-100 (50 = moyenne)

```
power(m) = clamp(50 + K · ln(m), 0, 100)
```

Centré sur 50 (moyenne), monotone en `m` ⇒ **le classement affiché == le
classement par probabilité de victoire**. Les seuils UI (vert ≥65, jaune ≥45)
restent cohérents.

## Constantes réglables (lib/scoring.py)

```
DEFAULT_FIELD = 12      # taille de champ si inconnue (carrière PMU)
KAPPA_WIN    = 1.0      # shrinkage victoire  (en "victoires attendues")
KAPPA_PLACE  = 3.0      # shrinkage place
POWER_SCALE  = 28.0     # points par unité de ln(m)
W_HORSE/D/TRAINER       # pondération piliers : 0.45 / 0.35 / 0.20
```

## Collecte de données modifiée (app.py)

Le bucket `{c, v, p}` devient `{c, v, p, dw, dp}`. Dans `_process_task`, on
connaît `N` (nombre de partants) → on incrémente `dw += 1/N` et
`dp += min(3,N)/N`. Cache bumpé → `stats_v8.pkl` (reconstruction auto).

## Contrat UI préservé

`scores.{cheval,driver,entraineur,composite}` (0-100), `edge`, `rang`, `cote`,
`ordreArrivee`, `nom`, `driver`, `entraineur`, `nbCourses`, `nbVictoires` —
inchangés. **Champs ajoutés** : `proba`, `probaTop3`, `fairOdds`, `strength`.
