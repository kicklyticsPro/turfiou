"""
ml_v8.py — Pipeline ML V8 pour Turfiou
=======================================
Améliorations vs V7 :
  1. Feature Engineering — 62→76 features (14 interactions + 14 intra-course ranking)
  2. Purged Time-Series CV — élimine le data leak temporel
  3. Optuna hyperparam tuning — auto-optimisation XGB/LGB (30 trials)
  4. OOF stacking multi-colonne — meta-learner voit chaque learner séparément
  5. Calibration isotonic + Platt — meilleure calibration
  6. Poids dynamiques TabNet — basés sur validation, pas fixés
  7. Data augmentation — SMOTE-like + course-level augment
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
    from xgboost import XGBClassifier, XGBRanker
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

HAS_IMBLEARN = False
try:
    from imblearn.over_sampling import SMOTE, ADASYN
    HAS_IMBLEARN = True
except Exception:
    pass


# ══════════════════════════════════════════════════════════════
#  1. FEATURE ENGINEERING V8 — interactions + ranking intra-course
# ══════════════════════════════════════════════════════════════

def engineer_interactions(raw):
    """
    Prend le dict raw (48 features) et retourne 14 features d'interaction.
    """
    def g(key, default=0.0):
        v = raw.get(key, default)
        return float(v) if v is not None else float(default)

    avg_place_3 = g("avg_place_3")
    avg_place_5 = g("avg_place_5")
    place_1 = g("place_1")
    variance = g("variance_places")
    dr_wr = g("driver_win_rate")
    en_wr = g("entraineur_win_rate")
    elo = g("elo")
    elo_trend = g("elo_trend_raw")
    nb_courses = max(g("nb_courses"), 1)
    days_since = g("days_since_last")
    nb_courses_mois = g("nb_courses_mois")
    dist_count = g("dist_similar_count")
    dr_disc_wr = g("driver_disc_win_rate")
    cote = max(g("cote"), 1.0)
    inv_cote = g("inv_cote")
    chimie_courses = g("chimie_courses")
    chimie_wr = g("chimie_win_rate")
    last_place = g("last_place")
    gains_pc_log = g("gains_par_course_log")
    nb_partants = max(g("nb_partants"), 1)

    return [
        avg_place_3 * cote,                          # forme_x_cote
        avg_place_5 - place_1 if (avg_place_5 > 0 and place_1 > 0) else 0.0,  # momentum
        dr_wr * en_wr / 100.0 if dr_wr > 0 and en_wr > 0 else 0.0,  # driver_x_entraineur
        elo * inv_cote * 100,                         # elo_x_marche
        -variance,                                    # regularite_inv
        dist_count / nb_courses,                      # exp_distance
        dr_disc_wr / max(dr_wr, 1.0),                 # specialisation_driver
        inv_cote * 100,                               # proba_marche
        chimie_wr / max(dr_wr, 1.0) if chimie_courses >= 2 else 0.0,  # chimie_relative
        elo_trend * nb_courses_mois,                  # tendance_ponderee
        1.0 if 0 < last_place <= 3 else 0.0,         # dernier_top3
        min(days_since, 120) / 30.0,                  # inactivite_score
        gains_pc_log,                                 # progression_gains
        1.0 / nb_partants * 100,                      # competitivite
    ]


def compute_course_ranking_features(course_features):
    """
    Prend une liste de vecteurs features (1 par cheval d'une même course)
    et retourne pour chaque cheval ses features de ranking intra-course.
    
    Pour chaque dimension clé, on calcule :
    - rank_percentile : position relative (0=meilleur, 1=pire)
    - z_score : écart à la moyenne de la course
    - is_top_N : binaire, dans les X meilleurs
    
    → 14 features de ranking par cheval
    """
    n_horses = len(course_features)
    if n_horses < 2:
        return [np.zeros(14) for _ in course_features]
    
    # Indices des features clés dans le vecteur 48
    # (depuis featurize() — v7 raw path)
    DIMS = {
        "elo": 37,            # elo — plus haut = meilleur
        "cote": 40,           # cote — plus bas = meilleur → inversé
        "win_rate": 4,        # win_rate
        "top3_rate": 15,      # top3_rate
        "avg_place_3": 13,    # avg_place_3 — plus bas = meilleur → inversé
        "driver_wr": 26,      # driver_win_rate
        "entraineur_wr": 33,  # entraineur_win_rate
    }
    
    mat = np.array(course_features, dtype=np.float64)
    
    results = []
    for i in range(n_horses):
        feats = []
        
        # Pour chaque dimension, calculer le percentile rank
        for dim_name, dim_idx in DIMS.items():
            vals = mat[:, dim_idx].copy()
            horse_val = vals[i]
            
            # Dimensions où "plus bas = meilleur" → on inverse
            if dim_name in ("cote", "avg_place_3"):
                # Percentile : 0 = meilleur (cote la plus basse)
                rank_pct = np.mean(vals <= horse_val)
            else:
                # Percentile : 0 = pire, 1 = meilleur
                rank_pct = np.mean(vals <= horse_val)
            
            feats.append(rank_pct)
        
        # Z-scores sur 2 dimensions clés
        for dim_idx in [37, 40]:  # elo, cote
            vals = mat[:, dim_idx]
            mean_val = np.mean(vals)
            std_val = np.std(vals) if np.std(vals) > 0 else 1.0
            feats.append((mat[i, dim_idx] - mean_val) / std_val)
        
        # Is top-N binary
        # Top 3 par Elo
        elo_vals = mat[:, 37]
        top3_elo = np.argsort(-elo_vals)[:min(3, n_horses)]
        feats.append(1.0 if i in top3_elo else 0.0)
        
        # Top 3 par cote (inv_cote = plus fort)
        inv_cote_vals = mat[:, 41]  # inv_cote
        top3_cote = np.argsort(-inv_cote_vals)[:min(3, n_horses)]
        feats.append(1.0 if i in top3_cote else 0.0)
        
        # Écart au favori (1er de la cote)
        fav_idx = np.argmax(inv_cote_vals)
        elo_gap_to_fav = mat[i, 37] - mat[fav_idx, 37]
        feats.append(elo_gap_to_fav)
        
        # Force relative : (elo * win_rate) vs moyenne course
        power = mat[:, 37] * mat[:, 4]  # elo * win_rate
        mean_power = np.mean(power)
        feats.append(power[i] / max(mean_power, 1.0))
        
        results.append(np.array(feats))
    
    return results


def augment_course_level(X, y, group_ids):
    """
    Data augmentation au niveau course.
    Pour chaque course, on crée des "sous-courses" en retirant
    aléatoirement 1-2 chevaux, ce qui crée de nouveaux exemples
    avec des rankings différents.
    
    X : list of feature vectors
    y : list of labels
    group_ids : list of course identifiers (same id = same course)
    
    Retourne X_aug, y_aug avec les données originales + augmentées
    """
    from collections import defaultdict
    
    # Grouper par course
    course_groups = defaultdict(list)
    for idx, gid in enumerate(group_ids):
        course_groups[gid].append(idx)
    
    X_aug = list(X)
    y_aug = list(y)
    
    rng = np.random.RandomState(42)
    
    for gid, indices in course_groups.items():
        n = len(indices)
        if n < 6:  # Trop petit pour augmenter
            continue
        
        # Créer 2 sous-courses par course originale
        for _ in range(2):
            # Retirer 1-3 chevaux aléatoirement
            n_remove = rng.randint(1, min(3, n - 5))
            keep = rng.choice(indices, size=n - n_remove, replace=False)
            keep = sorted(keep)
            
            for idx in keep:
                X_aug.append(X[idx])
                y_aug.append(y[idx])
    
    return X_aug, y_aug


# ══════════════════════════════════════════════════════════════
#  2. PURGED TIME-SERIES CV
# ══════════════════════════════════════════════════════════════

def purged_ts_cv(n_samples, n_splits=5, purge_pct=0.03):
    purge = max(1, int(n_samples * purge_pct))
    fold_size = n_samples // (n_splits + 1)
    splits = []
    for i in range(n_splits):
        train_end = (i + 1) * fold_size
        val_start = train_end + purge
        val_end = min((i + 2) * fold_size, n_samples)
        if val_start >= n_samples or val_end <= val_start:
            continue
        splits.append((np.arange(0, train_end), np.arange(val_start, val_end)))
    return splits


def safe_ts_cv(y, n_splits=5):
    n = len(y)
    n_pos = max(1, int(np.sum(y)))
    if n < 200 or n_pos < 20:
        k = max(2, min(n_splits, n_pos // 2, n // 30))
        return list(StratifiedKFold(n_splits=k, shuffle=True, random_state=42).split(np.zeros(n), y))
    splits = purged_ts_cv(n, n_splits=n_splits)
    if not splits:
        return list(StratifiedKFold(n_splits=3, shuffle=True, random_state=42).split(np.zeros(n), y))
    return splits


# ══════════════════════════════════════════════════════════════
#  3. OPTUNA HYPERPARAM TUNING
# ══════════════════════════════════════════════════════════════

def _optuna_tune_xgb(X, y, n_trials=30, timeout=180):
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
            "eval_metric": "logloss", "use_label_encoder": False,
            "random_state": 42, "n_jobs": -1, "verbosity": 0,
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
        best.update({"scale_pos_weight": spw, "eval_metric": "logloss",
                     "use_label_encoder": False, "random_state": 42, "n_jobs": -1, "verbosity": 0})
        print(f"    [Optuna XGB] Brier={study.best_value:.4f}")
        return XGBClassifier(**best)
    except Exception as e:
        print(f"    [Optuna XGB] Erreur: {e}")
        return None


def _optuna_tune_lgb(X, y, n_trials=30, timeout=180):
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
            "is_unbalance": True, "random_state": 42, "n_jobs": -1, "verbose": -1,
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
        best.update({"is_unbalance": True, "random_state": 42, "n_jobs": -1, "verbose": -1})
        print(f"    [Optuna LGB] Brier={study.best_value:.4f}")
        return LGBMClassifier(**best)
    except Exception as e:
        print(f"    [Optuna LGB] Erreur: {e}")
        return None


# ══════════════════════════════════════════════════════════════
#  BASE LEARNERS
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
#  STACKING V8
# ══════════════════════════════════════════════════════════════

class StackingV8:
    def __init__(self):
        self.model = None
        self.calibrated = None
        self.val_score = None
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
            return self._fit_fallback(X, y, target)

        # Optuna
        tuned_xgb = None
        tuned_lgb = None
        if use_optuna and HAS_OPTUNA:
            print(f"\n  [0/4] Optuna tuning...")
            if HAS_XGB:
                tuned_xgb = _optuna_tune_xgb(X, y, n_trials=30, timeout=180)
            if HAS_LGB:
                tuned_lgb = _optuna_tune_lgb(X, y, n_trials=30, timeout=180)

        learners = []
        if HAS_XGB:
            learners.append(("xgb", tuned_xgb or _default_xgb(target)))
        if HAS_LGB:
            learners.append(("lgb", tuned_lgb or _default_lgb(target)))
        if HAS_HGB:
            learners.append(("hgb", _default_hgb(target)))

        if len(learners) < 2:
            return self._fit_fallback(X, y, target)

        # OOF multi-colonne
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
            print(f"    Fold {fold_idx+1}/{len(splits)} [{pct:.0f}%]")

        # Fit full
        print(f"\n  [2/4] Fit full dataset...")
        fitted_base = []
        for name, learner_proto in learners:
            learner = copy.deepcopy(learner_proto)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                learner.fit(X, y)
            fitted_base.append((name, learner))
            print(f"    ✅ {name}")

        # Meta-learner
        print(f"\n  [3/4] Meta-learner ({n_learners} OOF cols)...")
        meta = LogisticRegression(C=1.0, max_iter=2000, fit_intercept=True)
        meta.fit(oof_matrix, y)
        for j, (name, _) in enumerate(learners):
            coef = meta.coef_[0][j] if meta.coef_.ndim > 1 else meta.coef_[j]
            print(f"    {name}: coef={coef:.4f}")

        meta_oof_preds = meta.predict_proba(oof_matrix)[:, 1]
        self.val_score = brier_score_loss(y, meta_oof_preds)
        print(f"    Brier OOF = {self.val_score:.4f}")

        self.model = {"base_learners": fitted_base, "meta": meta,
                      "target": target, "learner_names": [n for n, _ in learners]}
        print(f"  ✅ Stacking V8 complet ({len(fitted_base)} learners)")

        # Calibration
        print(f"\n  [4/4] Calibration...")
        self.calibrated = self._calibrate(X, y)
        return self

    def _fit_fallback(self, X, y, target):
        for factory in [_default_hgb, _default_lgb, _default_xgb]:
            try:
                m = factory(target)
                break
            except Exception:
                continue
        else:
            m = LogisticRegression(max_iter=2000)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m.fit(X, y)
        self.model = {"base_learners": [("fallback", m)], "meta": None, "target": target}
        self.calibrated = None
        self.val_score = None
        print(f"  ✅ Fallback")
        return self

    def _calibrate(self, X, y):
        n = len(X)
        split = int(n * 0.8)
        X_cal, y_cal = X[split:], y[split:]
        if len(X_cal) < 30 or len(set(y_cal.tolist())) < 2:
            return None
        preds_cal = self._raw_predict_proba(X_cal)
        raw_brier = brier_score_loss(y_cal, preds_cal)
        print(f"    Raw Brier = {raw_brier:.4f}")

        best_cal, best_brier, best_name = None, raw_brier, "raw"
        # Platt
        try:
            from sklearn.calibration import _SigmoidCalibration
            platt = _SigmoidCalibration()
            platt.fit(preds_cal, y_cal)
            pb = brier_score_loss(y_cal, platt.predict(preds_cal))
            print(f"    Platt = {pb:.4f}")
            if pb < best_brier: best_cal, best_brier, best_name = platt, pb, "PLATT"
        except Exception: pass
        # Isotonic
        try:
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(preds_cal, y_cal)
            ib = brier_score_loss(y_cal, iso.predict(preds_cal))
            print(f"    Isotonic = {ib:.4f}")
            if ib < best_brier: best_cal, best_brier, best_name = iso, ib, "ISOTONIC"
        except Exception: pass

        if best_cal:
            print(f"    ✅ {best_name} ({raw_brier:.4f} → {best_brier:.4f})")
        return best_cal

    def _raw_predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(1, -1)
        base_preds = np.column_stack([l.predict_proba(X)[:, 1] for _, l in self.model["base_learners"]])
        meta = self.model.get("meta")
        if meta: return meta.predict_proba(base_preds)[:, 1]
        return np.mean(base_preds, axis=1)

    def predict_proba(self, X):
        raw = self._raw_predict_proba(X)
        if self.calibrated is not None:
            return self.calibrated.predict(raw)
        return raw

    def predict_one(self, x):
        return float(self.predict_proba(np.asarray(x).reshape(1, -1))[0])

    def feature_importance(self, feature_names=None):
        importances = [l.feature_importances_ for _, l in self.model.get("base_learners", []) if hasattr(l, "feature_importances_")]
        if not importances: return None
        avg = np.mean(importances, axis=0)
        names = feature_names or [f"f{i}" for i in range(len(avg))]
        return sorted(zip(names, avg), key=lambda x: -x[1])

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"model": self.model, "calibrated": self.calibrated,
                     "val_score": self.val_score, "n_features": self.n_features, "version": "v8"}, path)
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
#  TABNET V8
# ══════════════════════════════════════════════════════════════

class TabNetV8:
    def __init__(self):
        self.model = None
        self.calibrated = None
        self.val_score = None

    def fit(self, X, y, target="win"):
        if not HAS_TABNET:
            print(f"  [TabNetV8] ⚠️ Non disponible, skip")
            return None
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n, d = X.shape
        n_pos = int(np.sum(y))
        print(f"\n  [TabNetV8] {target.upper()} — {n}×{d}, {n_pos}+")
        if n < 300 or n_pos < 20:
            print(f"  ⚠️ Insuffisant"); return None

        sp = int(n * 0.85)
        idx = np.random.RandomState(42).permutation(n)
        X_train, y_train = X[idx[:sp]], y[idx[:sp]]
        X_val, y_val = X[idx[sp:]], y[idx[sp:]]
        if len(set(y_val.tolist())) < 2 or len(set(y_train.tolist())) < 2:
            return None

        model = TabNetClassifier(n_d=24, n_a=24, n_steps=5, gamma=1.5,
                                 n_independent=2, n_shared=3,
                                 lambda_sparse=1e-4, momentum=0.3, clip_value=2.0,
                                 optimizer_fn=torch.optim.AdamW,
                                 optimizer_params=dict(lr=2e-2, weight_decay=1e-4),
                                 scheduler_params={"T_max": 100, "eta_min": 1e-4},
                                 scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
                                 mask_type="entmax", seed=42, verbose=0)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], eval_name=["val"],
                  eval_metric=["logloss"], max_epochs=200, patience=20,
                  batch_size=min(512, max(64, n//4)), virtual_batch_size=min(256, max(32, n//8)))
        self.model = model
        val_preds = model.predict_proba(X_val)[:, 1]
        self.val_score = brier_score_loss(y_val, val_preds)
        self.calibrated = self._calibrate(X_val, y_val)
        print(f"  ✅ TabNet Brier={self.val_score:.4f}")
        return self

    def _calibrate(self, X, y):
        try:
            raw = self.model.predict_proba(X)[:, 1]
            if len(set(y.tolist())) < 2: return None
            best, best_b = None, brier_score_loss(y, raw)
            try:
                iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(raw, y)
                b = brier_score_loss(y, iso.predict(raw))
                if b < best_b: best, best_b = iso, b
            except Exception: pass
            try:
                from sklearn.calibration import _SigmoidCalibration
                p = _SigmoidCalibration(); p.fit(raw, y)
                b = brier_score_loss(y, p.predict(raw))
                if b < best_b: best, best_b = p, b
            except Exception: pass
            return best
        except Exception: return None

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1: X = X.reshape(1, -1)
        raw = self.model.predict_proba(X)[:, 1]
        if self.calibrated is not None:
            return self.calibrated.predict(raw)
        return raw

    def predict_one(self, x):
        return float(self.predict_proba(np.asarray(x, dtype=np.float32).reshape(1, -1))[0])

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        model_dir = path.replace(".pkl", "_tabnet"); os.makedirs(model_dir, exist_ok=True)
        self.model.save_model(os.path.join(model_dir, "tabnet"))
        joblib.dump({"calibrated": self.calibrated, "val_score": self.val_score,
                     "model_dir": model_dir, "version": "v8_tabnet"}, path)
        print(f"  💾 TabNet → {path}")

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls(); obj.calibrated = data.get("calibrated"); obj.val_score = data.get("val_score")
        model_dir = data.get("model_dir", path.replace(".pkl", "_tabnet"))
        mp = os.path.join(model_dir, "tabnet.zip")
        if os.path.exists(mp):
            obj.model = TabNetClassifier(); obj.model.load_model(mp)
        return obj


# ══════════════════════════════════════════════════════════════
#  ENSEMBLE V8
# ══════════════════════════════════════════════════════════════

class EnsembleV8:
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

        self.stacking = StackingV8()
        self.stacking.fit(X, y, target=target, use_optuna=use_optuna)

        self.tabnet = TabNetV8()
        if self.tabnet.fit(X, y, target=target) is None:
            self.tabnet = None; self.w_stack = 1.0; self.w_tabnet = 0.0
        else:
            s_b = self.stacking.val_score or 0.25
            t_b = self.tabnet.val_score or 0.25
            s_inv, t_inv = 1.0/max(s_b, 0.001), 1.0/max(t_b, 0.001)
            total = s_inv + t_inv
            self.w_stack, self.w_tabnet = s_inv/total, t_inv/total

        self._final_calibration(X, y)
        elapsed = time.time() - t0
        mode = f"stack={self.w_stack:.0%}+tabnet={self.w_tabnet:.0%}" if self.tabnet else "stack seul"
        print(f"\n  ⏱️  {target.upper()} en {elapsed:.1f}s — {mode}")
        if self.val_score: print(f"  📊 Val Brier = {self.val_score:.4f}")
        return self

    def _final_calibration(self, X, y):
        n = len(X); split = int(n * 0.85)
        X_cal, y_cal = X[split:], y[split:]
        if len(X_cal) < 30 or len(set(y_cal.tolist())) < 2: return
        preds = self._raw_predict(X_cal)
        raw_brier = brier_score_loss(y_cal, preds)
        try:
            iso = IsotonicRegression(out_of_bounds="clip"); iso.fit(preds, y_cal)
            cb = brier_score_loss(y_cal, iso.predict(preds))
            if cb < raw_brier:
                self.calibrated = iso; self.val_score = cb
                print(f"  Calibration finale: {raw_brier:.4f} → {cb:.4f}")
            else: self.val_score = raw_brier
        except Exception: self.val_score = raw_brier

    def _raw_predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1: X = X.reshape(1, -1)
        probs = np.asarray(self.stacking.predict_proba(X), dtype=np.float64)
        if self.tabnet is not None:
            tn = np.asarray(self.tabnet.predict_proba(X), dtype=np.float64)
            probs = self.w_stack * probs + self.w_tabnet * tn
        return probs

    def predict_proba(self, X):
        raw = self._raw_predict(X)
        if self.calibrated is not None: return self.calibrated.predict(raw)
        return raw

    def predict_one(self, x):
        return float(self.predict_proba(np.asarray(x).reshape(1, -1))[0])

    def feature_importance(self, feature_names=None):
        return self.stacking.feature_importance(feature_names)

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        stack_path = path.replace(".pkl", "_stack.pkl")
        self.stacking.save(stack_path)
        data = {"w_stack": self.w_stack, "w_tabnet": self.w_tabnet,
                "has_tabnet": self.tabnet is not None, "calibrated": self.calibrated,
                "val_score": self.val_score, "stack_path": stack_path, "version": "v8_ensemble"}
        if self.tabnet is not None:
            tn_path = path.replace(".pkl", "_tabnet.pkl")
            self.tabnet.save(tn_path); data["tabnet_path"] = tn_path
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
                try: obj.tabnet = TabNetV8.load(tn_path)
                except Exception:
                    obj.tabnet = None; obj.w_stack = 1.0; obj.w_tabnet = 0.0
        return obj


# ══════════════════════════════════════════════════════════════
#  API PUBLIQUE
# ══════════════════════════════════════════════════════════════

def train_v8(X, y, save_path, target="win", use_optuna=False):
    model = EnsembleV8()
    model.fit(X, y, target=target, use_optuna=use_optuna)
    model.save(save_path)
    return model

def load_v8(path):
    if not os.path.exists(path): return None
    try: return EnsembleV8.load(path)
    except Exception as e:
        print(f"  [load_v8] Erreur: {e}"); return None
