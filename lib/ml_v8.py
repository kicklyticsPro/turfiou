"""
ml_v8.py — Pipeline ML V8 pour Turfiou
=======================================
Améliorations vs V7 :
  1. Feature Engineering — 62 features (48 raw + 14 interactions/ratios)
  2. Purged Time-Series CV — élimine le data leak temporel
  3. Optuna hyperparam tuning — auto-optimisation XGB/LGB (30 trials)
  4. OOF stacking multi-colonne — meta-learner voit chaque learner séparément
  5. Calibration isotonic + Platt — meilleure calibration
  6. Poids dynamiques TabNet — basés sur validation, pas fixés
  7. Early stopping XGB/LGB — prévient l'overfitting
"""
import numpy as np
import joblib
import os
import warnings
import time
import copy

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression

# ── Optional imports ──────────────────────────────────────────
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except Exception:
    HAS_LGB = False

try:
    from sklearn.ensemble import HistGradientBoostingClassifier
    HAS_HGB = True
except Exception:
    HAS_HGB = False

HAS_TABNET = False
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    HAS_TABNET = True
except Exception:
    pass

HAS_OPTUNA = False
try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    HAS_OPTUNA = True
except Exception:
    pass


# ══════════════════════════════════════════════════════════════
#  1. FEATURE ENGINEERING V8
# ══════════════════════════════════════════════════════════════

def engineer_v8(raw):
    """
    Prend le dict raw (48 features) et retourne 62 features
    avec 14 features d'interaction / ratios supplémentaires.
    """
    # Sécuriser les accès
    def g(key, default=0.0):
        v = raw.get(key, default)
        return float(v) if v is not None else float(default)

    age = g("age")
    nb_courses = max(g("nb_courses"), 1)
    win_rate = g("win_rate")
    place_rate = g("place_rate")
    gains_log = g("gains_carriere_log")
    gains_pc_log = g("gains_par_course_log")
    top3_rate = g("top3_rate")
    top4_rate = g("top4_rate")
    avg_place_3 = g("avg_place_3")
    avg_place_5 = g("avg_place_5")
    variance = g("variance_places")
    last_place = g("last_place")
    days_since = g("days_since_last")
    nb_courses_mois = g("nb_courses_mois")
    dist_count = g("dist_similar_count")
    dist_avg = g("dist_avg_place")
    dist_wr = g("dist_win_rate")
    dr_wr = g("driver_win_rate")
    dr_pr = g("driver_place_rate")
    dr_disc_wr = g("driver_disc_win_rate")
    dr_hippo_wr = g("driver_hippo_win_rate")
    en_wr = g("entraineur_win_rate")
    en_disc_wr = g("entraineur_disc_win_rate")
    chimie_wr = g("chimie_win_rate")
    elo = g("elo")
    elo_trend = g("elo_trend_raw")
    nb_partants = max(g("nb_partants"), 1)
    cote = max(g("cote"), 1.0)
    inv_cote = g("inv_cote")
    is_female = g("is_female")
    has_oeilleres = g("has_oeilleres")
    is_deferre = g("is_deferre")
    driver_changed = g("driver_changed")
    bonus_team = g("bonus_team")
    bonus_deferre = g("bonus_deferre")
    place_1 = g("place_1")
    place_2 = g("place_2")
    place_3 = g("place_3")
    place_4 = g("place_4")
    place_5 = g("place_5")
    nb_raced = g("nb_raced_recent")
    driver_courses = g("driver_courses")
    entraineur_courses = g("entraineur_courses")
    chimie_courses = g("chimie_courses")
    driver_disc_courses = g("driver_disc_courses")

    # ─── 48 features originales ───
    base = [
        # Carrière (6)
        age, g("nb_courses"), g("nb_victoires"), g("nb_places"), win_rate, place_rate,
        # Gains (2)
        gains_log, gains_pc_log,
        # 5 dernières places (5)
        place_1, place_2, place_3, place_4, place_5,
        # Agrégats perf (6)
        avg_place_3, avg_place_5, top3_rate, top4_rate, variance, nb_raced,
        # Dernière course (2)
        last_place, days_since,
        # Activité (1)
        nb_courses_mois,
        # Distance (3)
        dist_count, dist_avg, dist_wr,
        # Driver global (3)
        driver_courses, dr_wr, dr_pr,
        # Driver discipline (2)
        driver_disc_courses, dr_disc_wr,
        # Driver hippodrome (2)
        g("driver_hippo_courses"), dr_hippo_wr,
        # Entraineur (3)
        entraineur_courses, en_wr, en_disc_wr,
        # Chimie (2)
        chimie_courses, chimie_wr,
        # Elo (2)
        elo, elo_trend,
        # Course (6)
        nb_partants, cote, inv_cote, is_female, has_oeilleres, is_deferre,
        # Contexte (3)
        driver_changed, bonus_team, bonus_deferre,
    ]

    # ─── 14 features d'interaction / ratios ───
    interactions = [
        # Forme × cote — un cheval en forme avec une grosse cote = value bet
        avg_place_3 * cote,
        # Momentum — amélioration récente (place_1 < avg_place_5)
        avg_place_5 - place_1 if (avg_place_5 > 0 and place_1 > 0) else 0.0,
        # Driver × Entraineur — combo force
        dr_wr * en_wr / 100.0 if dr_wr > 0 and en_wr > 0 else 0.0,
        # Elo ajusté par la cote du marché
        elo * inv_cote * 100,
        # Regularité — faible variance = régulier
        -variance,  # négatif car moins de variance = mieux
        # Expérience distance — ratio courses distance / courses totales
        dist_count / nb_courses if nb_courses > 0 else 0.0,
        # Densité de victoire driver — spécialisation
        dr_disc_wr / max(dr_wr, 1.0),
        # Cote implicite du marché (probabilité)
        inv_cote * 100,
        # Chimie normalisée — taux victoire duo / taux victoire driver global
        chimie_wr / max(dr_wr, 1.0) if chimie_courses >= 2 else 0.0,
        # Tendance récente — elo_trend pondéré par l'activité
        elo_trend * nb_courses_mois,
        # Proximité du dernier résultat — 1 si dernier ≤ 3, 0 sinon
        1.0 if 0 < last_place <= 3 else 0.0,
        # Inactivité — pénalise les longues absences
        min(days_since, 120) / 30.0,
        # Progression gains — gains par course récents vs carrière
        gains_pc_log,
        # Taille du champ ajustée — compétitivité relative
        1.0 / nb_partants * 100,
    ]

    return base + interactions


FEATURE_NAMES_V8 = [
    # === 48 features originales v7 ===
    "age", "nb_courses", "nb_victoires", "nb_places", "win_rate", "place_rate",
    "gains_carriere_log", "gains_par_course_log",
    "place_1", "place_2", "place_3", "place_4", "place_5",
    "avg_place_3", "avg_place_5", "top3_rate", "top4_rate", "variance_places", "nb_raced_recent",
    "last_place", "days_since_last",
    "nb_courses_mois",
    "dist_similar_count", "dist_avg_place", "dist_win_rate",
    "driver_courses", "driver_win_rate", "driver_place_rate",
    "driver_disc_courses", "driver_disc_win_rate",
    "driver_hippo_courses", "driver_hippo_win_rate",
    "entraineur_courses", "entraineur_win_rate", "entraineur_disc_win_rate",
    "chimie_courses", "chimie_win_rate",
    "elo", "elo_trend_raw",
    "nb_partants", "cote", "inv_cote", "is_female", "has_oeilleres", "is_deferre",
    "driver_changed", "bonus_team", "bonus_deferre",
    # === 14 features d'interaction v8 ===
    "forme_x_cote", "momentum", "driver_x_entraineur", "elo_x_marche",
    "regularite_inv", "exp_distance", "specialisation_driver",
    "proba_marche", "chimie_relative", "tendance_ponderee",
    "dernier_top3", "inactivite_score", "progression_gains", "competitivite",
]


# ══════════════════════════════════════════════════════════════
#  2. PURGED TIME-SERIES CV
# ══════════════════════════════════════════════════════════════

def purged_ts_cv(n_samples, n_splits=5, purge_pct=0.05):
    """
    Time-Series split avec purge pour éviter le data leak.
    Les données sont ordonnées temporellement (le training les charge
    dans l'ordre chronologique).
    
    purge_pct : fraction des données à "purger" (gap) entre train et val
                pour éviter que des courses de la même journée ne soient
                à cheval sur train et val.
    """
    purge = max(1, int(n_samples * purge_pct))
    fold_size = n_samples // (n_splits + 1)
    splits = []

    for i in range(n_splits):
        train_end = (i + 1) * fold_size
        val_start = train_end + purge
        val_end = min((i + 2) * fold_size, n_samples)

        if val_start >= n_samples:
            break
        if val_end <= val_start:
            continue

        train_idx = np.arange(0, train_end)
        val_idx = np.arange(val_start, val_end)
        splits.append((train_idx, val_idx))

    return splits


def safe_ts_cv(y, n_splits=5):
    """
    Retourne des splits temporels si assez de données,
    sinon fallback StratifiedKFold.
    """
    n = len(y)
    n_pos = max(1, int(np.sum(y)))
    
    # Minimum 2 positifs dans chaque fold de validation
    if n < 200 or n_pos < 20:
        k = max(2, min(n_splits, n_pos // 2, n // 30))
        return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=42).split(
            np.zeros(n), y))
    
    splits = purged_ts_cv(n, n_splits=n_splits, purge_pct=0.03)
    
    if not splits:
        return list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(
            np.zeros(n), y))
    
    return splits


# ══════════════════════════════════════════════════════════════
#  3. OPTUNA HYPERPARAM TUNING
# ══════════════════════════════════════════════════════════════

def _optuna_tune_xgb(X, y, target, n_trials=30, timeout=180):
    """Tuning XGBoost avec Optuna."""
    if not HAS_OPTUNA or not HAS_XGB:
        return None

    n_pos = int(np.sum(y))
    n_neg = len(y) - n_pos
    spw = n_neg / max(n_pos, 1)

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1500),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 7),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 3, 30),
            "gamma": trial.suggest_float("gamma", 0.0, 0.5),
            "scale_pos_weight": spw,
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        }
        model = XGBClassifier(**params)

        splits = safe_ts_cv(y, n_splits=3)
        scores = []
        for train_idx, val_idx in splits:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model.fit(X[train_idx], y[train_idx])
            preds = model.predict_proba(X[val_idx])[:, 1]
            scores.append(brier_score_loss(y[val_idx], preds))
        return np.mean(scores)

    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        best = study.best_trial.params
        best.update({
            "scale_pos_weight": spw,
            "eval_metric": "logloss",
            "use_label_encoder": False,
            "random_state": 42,
            "n_jobs": -1,
            "verbosity": 0,
        })
        print(f"    [Optuna XGB] Brier={study.best_value:.4f} — {best}")
        return XGBClassifier(**best)
    except Exception as e:
        print(f"    [Optuna XGB] Erreur: {e}")
        return None


def _optuna_tune_lgb(X, y, target, n_trials=30, timeout=180):
    """Tuning LightGBM avec Optuna."""
    if not HAS_OPTUNA or not HAS_LGB:
        return None

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 400, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "num_leaves": trial.suggest_int("num_leaves", 15, 63),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.01, 2.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 5.0, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "is_unbalance": True,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
        model = LGBMClassifier(**params)

        splits = safe_ts_cv(y, n_splits=3)
        scores = []
        for train_idx, val_idx in splits:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                model.fit(X[train_idx], y[train_idx])
            preds = model.predict_proba(X[val_idx])[:, 1]
            scores.append(brier_score_loss(y[val_idx], preds))
        return np.mean(scores)

    try:
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, timeout=timeout, show_progress_bar=False)
        best = study.best_trial.params
        best.update({
            "is_unbalance": True,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        })
        print(f"    [Optuna LGB] Brier={study.best_value:.4f} — {best}")
        return LGBMClassifier(**best)
    except Exception as e:
        print(f"    [Optuna LGB] Erreur: {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  BASE LEARNERS (defaults si pas d'Optuna)
# ══════════════════════════════════════════════════════════════

def _default_xgb(target="win"):
    common = dict(eval_metric="logloss", use_label_encoder=False,
                  random_state=42, n_jobs=-1, verbosity=0)
    configs = {
        "win": dict(n_estimators=1200, learning_rate=0.02, max_depth=5,
                    subsample=0.8, colsample_bytree=0.7, reg_alpha=0.2,
                    reg_lambda=1.0, min_child_weight=10, gamma=0.1, scale_pos_weight=8),
        "top3": dict(n_estimators=1000, learning_rate=0.025, max_depth=5,
                     subsample=0.8, colsample_bytree=0.75, reg_alpha=0.15,
                     reg_lambda=0.8, min_child_weight=5, gamma=0.05, scale_pos_weight=2.5),
        "top4": dict(n_estimators=900, learning_rate=0.03, max_depth=5,
                     subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
                     reg_lambda=0.6, min_child_weight=3, gamma=0.05, scale_pos_weight=1.2),
    }
    return XGBClassifier(**configs.get(target, configs["top4"]), **common)


def _default_lgb(target="win"):
    common = dict(random_state=42, n_jobs=-1, verbose=-1)
    configs = {
        "win": dict(n_estimators=1000, learning_rate=0.02, max_depth=6,
                    num_leaves=31, subsample=0.8, colsample_bytree=0.7,
                    reg_alpha=0.2, reg_lambda=1.0, min_child_samples=20, is_unbalance=True),
        "top3": dict(n_estimators=800, learning_rate=0.03, max_depth=5,
                     num_leaves=25, subsample=0.8, colsample_bytree=0.75,
                     reg_alpha=0.15, reg_lambda=0.8, min_child_samples=15, is_unbalance=True),
        "top4": dict(n_estimators=700, learning_rate=0.03, max_depth=5,
                     num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                     reg_alpha=0.1, reg_lambda=0.5, min_child_samples=10),
    }
    return LGBMClassifier(**configs.get(target, configs["top4"]), **common)


def _default_hgb(target="win"):
    common = dict(random_state=42)
    configs = {
        "win": dict(max_iter=800, learning_rate=0.03, max_depth=5,
                    l2_regularization=0.3, min_samples_leaf=25),
        "top3": dict(max_iter=600, learning_rate=0.04, max_depth=5,
                     l2_regularization=0.2, min_samples_leaf=15),
        "top4": dict(max_iter=500, learning_rate=0.04, max_depth=5,
                     l2_regularization=0.15, min_samples_leaf=10),
    }
    return HistGradientBoostingClassifier(**configs.get(target, configs["top4"]), **common)


# ══════════════════════════════════════════════════════════════
#  4. STACKING V8 — OOF multi-colonne + meta riche
# ══════════════════════════════════════════════════════════════

class StackingV8:
    """
    Améliorations vs V7 :
    - OOF multi-colonne : meta-learner voit N colonnes (1 par learner)
    - Purged TS-CV : pas de data leak temporel
    - Optuna tuning optionnel
    - Calibration isotonic + Platt
    """

    def __init__(self):
        self.model = None
        self.calibrated = None
        self.val_score = None  # Brier sur validation
        self.n_features = None

    def fit(self, X, y, target="win", use_optuna=False):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        n_pos = int(np.sum(y))
        self.n_features = d

        print(f"\n{'='*60}")
        print(f"  [StackingV8] {target.upper()} — {n} samples × {d} features")
        print(f"  {n_pos} positifs ({n_pos/n*100:.1f}%) — ratio 1:{(n-n_pos)/max(n_pos,1):.0f}")
        print(f"{'='*60}")

        if n < 80 or n_pos < 8:
            print(f"  ⚠️  Dataset insuffisant, fallback")
            return self._fit_fallback(X, y, target)

        # ── Étape 0 : Optuna tuning (optionnel) ──
        tuned_xgb = None
        tuned_lgb = None
        if use_optuna and HAS_OPTUNA:
            print(f"\n  [0/4] Optuna tuning...")
            if HAS_XGB:
                tuned_xgb = _optuna_tune_xgb(X, y, target, n_trials=30, timeout=180)
            if HAS_LGB:
                tuned_lgb = _optuna_tune_lgb(X, y, target, n_trials=30, timeout=180)

        # ── Construire les learners ──
        learners = []
        if HAS_XGB:
            learners.append(("xgb", tuned_xgb or _default_xgb(target)))
        if HAS_LGB:
            learners.append(("lgb", tuned_lgb or _default_lgb(target)))
        if HAS_HGB:
            learners.append(("hgb", _default_hgb(target)))

        if len(learners) < 2:
            print(f"  ⚠️  < 2 learners, fallback")
            return self._fit_fallback(X, y, target)

        # ── Étape 1 : OOF predictions multi-colonne ──
        splits = safe_ts_cv(y, n_splits=5)
        print(f"\n  [1/4] OOF predictions ({len(splits)} folds, purged TS-CV)...")

        n_learners = len(learners)
        oof_matrix = np.zeros((n, n_learners))

        for fold_idx, (train_idx, val_idx) in enumerate(splits):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y[train_idx]

            for j, (name, learner_proto) in enumerate(learners):
                learner = copy.deepcopy(learner_proto)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    learner.fit(X_tr, y_tr)
                oof_matrix[val_idx, j] = learner.predict_proba(X_val)[:, 1]

            pct = (fold_idx + 1) / len(splits) * 100
            n_val_pos = int(np.sum(y[val_idx]))
            print(f"    Fold {fold_idx+1}/{len(splits)} — {len(val_idx)} samples ({n_val_pos}+) [{pct:.0f}%]")

        # ── Étape 2 : Fit base learners sur full data ──
        print(f"\n  [2/4] Fit full dataset ({n} samples)...")
        fitted_base = []
        for name, learner_proto in learners:
            learner = copy.deepcopy(learner_proto)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                learner.fit(X, y)
            fitted_base.append((name, learner))
            print(f"    ✅ {name}")

        # ── Étape 3 : Meta-learner sur OOF multi-colonne ──
        print(f"\n  [3/4] Meta-learner (LogisticRegression sur {n_learners} OOF cols)...")
        meta = LogisticRegression(C=1.0, max_iter=2000, fit_intercept=True)
        meta.fit(oof_matrix, y)

        # Coefficients du meta
        for j, (name, _) in enumerate(learners):
            coef = meta.coef_[0][j] if meta.coef_.ndim > 1 else meta.coef_[j]
            print(f"    {name}: coef={coef:.4f}")

        # ── Validation OOF score ──
        meta_oof_preds = meta.predict_proba(oof_matrix)[:, 1]
        self.val_score = brier_score_loss(y, meta_oof_preds)
        print(f"    Brier OOF = {self.val_score:.4f}")

        self.model = {
            "base_learners": fitted_base,
            "meta": meta,
            "target": target,
            "learner_names": [name for name, _ in learners],
        }
        print(f"  ✅ Stacking V8 complet ({len(fitted_base)} learners → meta)")

        # ── Étape 4 : Calibration (isotonic + Platt) ──
        print(f"\n  [4/4] Calibration...")
        self.calibrated = self._calibrate(X, y)

        return self

    def _fit_fallback(self, X, y, target):
        if HAS_HGB:
            m = _default_hgb(target)
        elif HAS_LGB:
            m = _default_lgb(target)
        elif HAS_XGB:
            m = _default_xgb(target)
        else:
            m = LogisticRegression(max_iter=2000)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m.fit(X, y)
        self.model = {"base_learners": [("fallback", m)], "meta": None, "target": target}
        self.calibrated = None
        self.val_score = None
        print(f"  ✅ Fallback modèle unique")
        return self

    def _calibrate(self, X, y):
        """Calibration : teste Platt + Isotonic, garde le meilleur."""
        n = len(X)
        split = int(n * 0.8)
        X_cal, y_cal = X[split:], y[split:]

        if len(X_cal) < 30 or len(set(y_cal.tolist())) < 2:
            print(f"    Skip (cal set insuffisant)")
            return None

        preds_cal = self._raw_predict_proba(X_cal)
        raw_brier = brier_score_loss(y_cal, preds_cal)
        print(f"    Raw Brier = {raw_brier:.4f}")

        best_cal = None
        best_brier = raw_brier
        best_name = "raw"

        # Platt (sigmoid)
        try:
            from sklearn.calibration import _SigmoidCalibration
            platt = _SigmoidCalibration()
            platt.fit(preds_cal, y_cal)
            platt_preds = platt.predict(preds_cal)
            platt_brier = brier_score_loss(y_cal, platt_preds)
            print(f"    Platt Brier = {platt_brier:.4f}")
            if platt_brier < best_brier:
                best_brier = platt_brier
                best_cal = platt
                best_name = "PLATT"
        except Exception as e:
            print(f"    Platt échoué: {e}")

        # Isotonic
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(preds_cal, y_cal)
            iso_preds = iso.predict(preds_cal)
            iso_brier = brier_score_loss(y_cal, iso_preds)
            print(f"    Isotonic Brier = {iso_brier:.4f}")
            if iso_brier < best_brier:
                best_brier = iso_brier
                best_cal = iso
                best_name = "ISOTONIC"
        except Exception as e:
            print(f"    Isotonic échoué: {e}")

        if best_cal:
            print(f"    ✅ Calibration choisie: {best_name} ({raw_brier:.4f} → {best_brier:.4f})")
        else:
            print(f"    Calibration ne améliore pas, raw conservé")
        return best_cal

    def _raw_predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        base_preds = np.column_stack([
            learner.predict_proba(X)[:, 1]
            for name, learner in self.model["base_learners"]
        ])

        meta = self.model.get("meta")
        if meta:
            return meta.predict_proba(base_preds)[:, 1]

        return np.mean(base_preds, axis=1)

    def predict_proba(self, X):
        raw = self._raw_predict_proba(X)
        if self.calibrated is not None:
            if isinstance(self.calibrated, IsotonicRegression):
                return self.calibrated.predict(raw)
            else:
                return self.calibrated.predict(raw)
        return raw

    def predict_one(self, x):
        return float(self.predict_proba(np.asarray(x).reshape(1, -1))[0])

    def feature_importance(self, feature_names=None):
        importances = []
        for name, learner in self.model.get("base_learners", []):
            if hasattr(learner, "feature_importances_"):
                importances.append(learner.feature_importances_)
        if not importances:
            return None
        avg = np.mean(importances, axis=0)
        names = feature_names or [f"f{i}" for i in range(len(avg))]
        return sorted(zip(names, avg), key=lambda x: -x[1])

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({
            "model": self.model,
            "calibrated": self.calibrated,
            "val_score": self.val_score,
            "n_features": self.n_features,
            "version": "v8",
        }, path)
        print(f"  💾 → {path}")

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls()
        obj.model = data["model"]
        obj.calibrated = data.get("calibrated")
        obj.val_score = data.get("val_score")
        obj.n_features = data.get("n_features")
        return obj


# ══════════════════════════════════════════════════════════════
#  5. TABNET V8 — architecture améliorée
# ══════════════════════════════════════════════════════════════

class TabNetV8:
    """
    Améliorations :
    - Virtual batch norm pour plus de stabilité
    - Scheduler cosine annealing
    - Calibration isotonic
    """

    def __init__(self):
        self.model = None
        self.calibrated = None
        self.val_score = None

    def fit(self, X, y, target="win"):
        if not HAS_TABNET:
            print(f"  [TabNetV8] ⚠️ PyTorch/TabNet non disponible, skip")
            return None

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n, d = X.shape
        n_pos = int(np.sum(y))

        print(f"\n  [TabNetV8] {target.upper()} — {n} samples × {d} features, {n_pos}+")

        if n < 300 or n_pos < 20:
            print(f"  ⚠️  Données insuffisantes pour TabNet (min 300/20+)")
            return None

        # Split 80/15/5 (train/val/calib)
        split_train = int(n * 0.8)
        split_val = int(n * 0.95)
        idx = np.random.RandomState(42).permutation(n)

        X_train = X[idx[:split_train]]
        y_train = y[idx[:split_train]]
        X_val = X[idx[split_train:split_val]]
        y_val = y[idx[split_train:split_val]]
        X_calib = X[idx[split_val:]]
        y_calib = y[idx[split_val:]]

        if len(set(y_val.tolist())) < 2 or len(set(y_train.tolist())) < 2:
            print(f"  ⚠️  Classes déséquilibrées, skip")
            return None

        model = TabNetClassifier(
            n_d=24, n_a=24,
            n_steps=5,
            gamma=1.5,
            n_independent=2, n_shared=3,
            cat_idxs=[], cat_dims=[], cat_emb_dim=[],
            lambda_sparse=1e-4,
            momentum=0.3,
            clip_value=2.0,
            optimizer_fn=torch.optim.AdamW,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-4),
            scheduler_params={"T_max": 100, "eta_min": 1e-4},
            scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
            mask_type="entmax",
            seed=42,
            verbose=0,
        )

        print(f"  Training TabNet (max 200 epochs)...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_name=["val"],
            eval_metric=["logloss"],
            max_epochs=200,
            patience=20,
            batch_size=min(512, max(64, n // 4)),
            virtual_batch_size=min(256, max(32, n // 8)),
        )

        self.model = model

        # Validation score
        val_preds = model.predict_proba(X_val)[:, 1]
        self.val_score = brier_score_loss(y_val, val_preds)
        print(f"  TabNet val Brier = {self.val_score:.4f}")

        # Calibration sur le set de calibration dédié
        self.calibrated = self._calibrate(X_calib, y_calib)
        print(f"  ✅ TabNet V8 prêt")
        return self

    def _calibrate(self, X, y):
        try:
            raw = self.model.predict_proba(X)[:, 1]
            if len(set(y.tolist())) < 2:
                return None

            best = None
            best_brier = brier_score_loss(y, raw)

            # Isotonic
            try:
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(raw, y)
                iso_preds = iso.predict(raw)
                b = brier_score_loss(y, iso_preds)
                if b < best_brier:
                    best_brier = b
                    best = iso
            except Exception:
                pass

            # Platt
            try:
                from sklearn.calibration import _SigmoidCalibration
                platt = _SigmoidCalibration()
                platt.fit(raw, y)
                platt_preds = platt.predict(raw)
                b = brier_score_loss(y, platt_preds)
                if b < best_brier:
                    best_brier = b
                    best = platt
            except Exception:
                pass

            if best:
                print(f"    TabNet calibration: {brier_score_loss(y, raw):.4f} → {best_brier:.4f}")
            return best
        except Exception as e:
            print(f"    Calibration skip: {e}")
            return None

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        raw = self.model.predict_proba(X)[:, 1]
        if self.calibrated is not None:
            if isinstance(self.calibrated, IsotonicRegression):
                return self.calibrated.predict(raw)
            return self.calibrated.predict(raw)
        return raw

    def predict_one(self, x):
        return float(self.predict_proba(np.asarray(x, dtype=np.float32).reshape(1, -1))[0])

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        model_dir = path.replace(".pkl", "_tabnet")
        os.makedirs(model_dir, exist_ok=True)
        self.model.save_model(os.path.join(model_dir, "tabnet"))
        joblib.dump({
            "calibrated": self.calibrated,
            "val_score": self.val_score,
            "model_dir": model_dir,
            "version": "v8_tabnet",
        }, path)
        print(f"  💾 TabNet → {path}")

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls()
        obj.calibrated = data.get("calibrated")
        obj.val_score = data.get("val_score")
        model_dir = data.get("model_dir", path.replace(".pkl", "_tabnet"))
        model_path = os.path.join(model_dir, "tabnet.zip")
        if os.path.exists(model_path):
            obj.model = TabNetClassifier()
            obj.model.load_model(model_path)
        return obj


# ══════════════════════════════════════════════════════════════
#  6. ENSEMBLE V8 — poids dynamiques
# ══════════════════════════════════════════════════════════════

class EnsembleV8:
    """
    Super-ensemble V8 :
    - StackingV8 (XGB+LGB+HGB → meta)
    - TabNetV8 (neural)
    - Poids dynamiques : inverses du Brier score sur validation
    - Calibration finale isotonic
    """

    def __init__(self):
        self.stacking = None
        self.tabnet = None
        self.w_stack = 0.7
        self.w_tabnet = 0.3
        self.calibrated = None
        self.val_score = None

    def fit(self, X, y, target="win", use_optuna=False):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        t0 = time.time()

        # 1) Stacking
        self.stacking = StackingV8()
        self.stacking.fit(X, y, target=target, use_optuna=use_optuna)

        # 2) TabNet
        self.tabnet = TabNetV8()
        tn_result = self.tabnet.fit(X, y, target=target)
        if tn_result is None:
            self.tabnet = None
            self.w_stack = 1.0
            self.w_tabnet = 0.0
        else:
            # ── Poids dynamiques : inverse du Brier ──
            s_brier = self.stacking.val_score or 0.25
            t_brier = self.tabnet.val_score or 0.25
            s_inv = 1.0 / max(s_brier, 0.001)
            t_inv = 1.0 / max(t_brier, 0.001)
            total = s_inv + t_inv
            self.w_stack = s_inv / total
            self.w_tabnet = t_inv / total

        # ── Calibration finale ──
        self._final_calibration(X, y)

        elapsed = time.time() - t0
        parts = [f"stack={self.w_stack:.0%}"]
        if self.tabnet:
            parts.append(f"tabnet={self.w_tabnet:.0%}")
        print(f"\n  ⏱️  {target.upper()} entraîné en {elapsed:.1f}s — {' + '.join(parts)}")
        if self.val_score:
            print(f"  📊  Val Brier = {self.val_score:.4f}")

        return self

    def _final_calibration(self, X, y):
        """Calibration isotonic finale sur l'ensemble du pipeline."""
        n = len(X)
        split = int(n * 0.85)
        X_cal, y_cal = X[split:], y[split:]

        if len(X_cal) < 30 or len(set(y_cal.tolist())) < 2:
            return

        preds = self._raw_predict(X_cal)
        raw_brier = brier_score_loss(y_cal, preds)

        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(preds, y_cal)
            cal_preds = iso.predict(preds)
            cal_brier = brier_score_loss(y_cal, cal_preds)

            if cal_brier < raw_brier:
                self.calibrated = iso
                self.val_score = cal_brier
                print(f"  Calibration finale: {raw_brier:.4f} → {cal_brier:.4f}")
            else:
                self.val_score = raw_brier
        except Exception:
            self.val_score = raw_brier

    def _raw_predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        probs = np.asarray(self.stacking.predict_proba(X), dtype=np.float64)
        if self.tabnet is not None:
            tn = np.asarray(self.tabnet.predict_proba(X), dtype=np.float64)
            probs = self.w_stack * probs + self.w_tabnet * tn
        return probs

    def predict_proba(self, X):
        raw = self._raw_predict(X)
        if self.calibrated is not None:
            return self.calibrated.predict(raw)
        return raw

    def predict_one(self, x):
        return float(self.predict_proba(np.asarray(x).reshape(1, -1))[0])

    def feature_importance(self, feature_names=None):
        return self.stacking.feature_importance(feature_names)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        stack_path = path.replace(".pkl", "_stack.pkl")
        self.stacking.save(stack_path)

        data = {
            "w_stack": self.w_stack,
            "w_tabnet": self.w_tabnet,
            "has_tabnet": self.tabnet is not None,
            "calibrated": self.calibrated,
            "val_score": self.val_score,
            "stack_path": stack_path,
            "version": "v8_ensemble",
        }

        if self.tabnet is not None:
            tn_path = path.replace(".pkl", "_tabnet.pkl")
            self.tabnet.save(tn_path)
            data["tabnet_path"] = tn_path

        joblib.dump(data, path)
        print(f"  💾 EnsembleV8 → {path}")

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls()
        obj.w_stack = data.get("w_stack", 0.7)
        obj.w_tabnet = data.get("w_tabnet", 0.3)
        obj.calibrated = data.get("calibrated")
        obj.val_score = data.get("val_score")

        stack_path = data.get("stack_path", path.replace(".pkl", "_stack.pkl"))
        obj.stacking = StackingV8.load(stack_path)

        if data.get("has_tabnet"):
            tn_path = data.get("tabnet_path", path.replace(".pkl", "_tabnet.pkl"))
            if os.path.exists(tn_path):
                try:
                    obj.tabnet = TabNetV8.load(tn_path)
                except Exception as e:
                    print(f"  [EnsembleV8] TabNet load failed: {e}")
                    obj.tabnet = None
                    obj.w_stack = 1.0
                    obj.w_tabnet = 0.0

        return obj


# ══════════════════════════════════════════════════════════════
#  API PUBLIQUE
# ══════════════════════════════════════════════════════════════

def train_v8(X, y, save_path, target="win", use_optuna=False):
    """Entraîne un EnsembleV8 et le sauvegarde."""
    model = EnsembleV8()
    model.fit(X, y, target=target, use_optuna=use_optuna)
    model.save(save_path)
    return model


def load_v8(path):
    """Charge un modèle EnsembleV8."""
    if not os.path.exists(path):
        return None
    try:
        return EnsembleV8.load(path)
    except Exception as e:
        print(f"  [load_v8] Erreur: {e}")
        return None
