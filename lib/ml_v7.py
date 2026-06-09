"""
ml_v7.py — Pipeline ML moderne pour Turfiou
============================================
Stacking Level 1 (base learners):
  - XGBoost (gradient boosting optimisé)
  - LightGBM (rapide, bon sur tabulaire)
  - HistGradientBoosting (sklearn natif, fallback GPU-free)

Stacking Level 2 (meta-learner):
  - LogisticRegression calibrée

+ Modèle TabNet (neural network tabulaire, optionnel si PyTorch dispo)

Features : 48 features brutes v7 (build_raw_features)
Targets  : WIN, TOP3, TOP4 (3 modèles indépendants)
Calibration : Platt scaling (sigmoid)
"""
import numpy as np
import joblib
import os
import warnings
import time
from pathlib import Path

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, roc_auc_score, log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

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

try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN = True
except ImportError:
    HAS_FROZEN = False

HAS_TABNET = False
try:
    from pytorch_tabnet.tab_model import TabNetClassifier
    import torch
    HAS_TABNET = True
except Exception:
    pass

try:
    from sklearn.ensemble import StackingClassifier
    HAS_STACKING = True
except Exception:
    HAS_STACKING = False


# ── Helpers ────────────────────────────────────────────────────

def _safe_cv(y, n_splits=5):
    n = len(y)
    n_pos = max(1, int(np.sum(y)))
    n_neg = n - n_pos
    max_k = min(
        n_splits,
        max(2, n_pos // 2),
        max(2, n_neg // 10),
        max(2, n // 30),
    )
    k = max(2, min(max_k, 5))
    return StratifiedKFold(n_splits=k, shuffle=True, random_state=42)


def _both_classes(y):
    return len(set(np.asarray(y).ravel().tolist())) >= 2


# ── Base Learners ──────────────────────────────────────────────

def _make_xgb(target="win"):
    """XGBoost optimisé par cible."""
    common = dict(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    if target == "win":
        return XGBClassifier(
            n_estimators=1200, learning_rate=0.02, max_depth=5,
            subsample=0.8, colsample_bytree=0.7, reg_alpha=0.2,
            reg_lambda=1.0, min_child_weight=10, gamma=0.1,
            scale_pos_weight=8,  # ~8% base rate
            **common
        )
    elif target == "top3":
        return XGBClassifier(
            n_estimators=1000, learning_rate=0.025, max_depth=5,
            subsample=0.8, colsample_bytree=0.75, reg_alpha=0.15,
            reg_lambda=0.8, min_child_weight=5, gamma=0.05,
            scale_pos_weight=2.5,  # ~25% base rate
            **common
        )
    else:  # top4
        return XGBClassifier(
            n_estimators=900, learning_rate=0.03, max_depth=5,
            subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1,
            reg_lambda=0.6, min_child_weight=3, gamma=0.05,
            scale_pos_weight=1.2,  # ~40% base rate
            **common
        )


def _make_lgb(target="win"):
    common = dict(
        random_state=42, n_jobs=-1, verbose=-1,
    )
    if target == "win":
        return LGBMClassifier(
            n_estimators=1000, learning_rate=0.02, max_depth=6,
            num_leaves=31, subsample=0.8, colsample_bytree=0.7,
            reg_alpha=0.2, reg_lambda=1.0, min_child_samples=20,
            is_unbalance=True,
            **common
        )
    elif target == "top3":
        return LGBMClassifier(
            n_estimators=800, learning_rate=0.03, max_depth=5,
            num_leaves=25, subsample=0.8, colsample_bytree=0.75,
            reg_alpha=0.15, reg_lambda=0.8, min_child_samples=15,
            is_unbalance=True,
            **common
        )
    else:
        return LGBMClassifier(
            n_estimators=700, learning_rate=0.03, max_depth=5,
            num_leaves=31, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.5, min_child_samples=10,
            **common
        )


def _make_hgb(target="win"):
    common = dict(random_state=42)
    if target == "win":
        return HistGradientBoostingClassifier(
            max_iter=800, learning_rate=0.03, max_depth=5,
            l2_regularization=0.3, min_samples_leaf=25,
            **common
        )
    elif target == "top3":
        return HistGradientBoostingClassifier(
            max_iter=600, learning_rate=0.04, max_depth=5,
            l2_regularization=0.2, min_samples_leaf=15,
            **common
        )
    else:
        return HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.04, max_depth=5,
            l2_regularization=0.15, min_samples_leaf=10,
            **common
        )


def _build_base_learners(target):
    """Construit les base learners adaptés à la cible."""
    learners = []
    if HAS_XGB:
        learners.append(("xgb", _make_xgb(target)))
    if HAS_LGB:
        learners.append(("lgb", _make_lgb(target)))
    if HAS_HGB:
        learners.append(("hgb", _make_hgb(target)))
    return learners


# ── StackingV7 : le cœur du modèle ────────────────────────────

class StackingV7:
    """
    Stacking Level 1 : XGBoost + LightGBM + HistGradientBoosting
    Stacking Level 2 : LogisticRegression
    Calibration      : Platt (sigmoid) sur split 80/20
    """

    def __init__(self):
        self.model = None
        self.calibrated = None
        self.feature_names = None
        self.meta_features_train = None

    def fit(self, X, y, target="win"):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, d = X.shape
        n_pos = int(np.sum(y))

        print(f"\n[StackingV7] === Target: {target.upper()} ===")
        print(f"  Dataset : {n} samples × {d} features, {n_pos} positifs ({n_pos/n*100:.1f}%)")

        if n < 80 or n_pos < 8:
            print(f"  ⚠️  Dataset insuffisant, fallback HGB seul")
            return self._fit_fallback(X, y, target)

        learners = _build_base_learners(target)
        if len(learners) < 2:
            print(f"  ⚠️  < 2 learners disponibles, fallback simple")
            return self._fit_fallback(X, y, target)

        cv = _safe_cv(y, n_splits=5)

        # --- Étape 1 : Out-of-fold predictions pour le meta-learner ---
        print(f"  [1/3] Génération OOF predictions (CV={cv.n_splits})...")
        oof_preds = np.zeros(n)
        fitted_base = []

        for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y)):
            X_tr, X_val = X[train_idx], X[val_idx]
            y_tr = y[train_idx]

            # Moyenne des probas de chaque base learner
            fold_preds = np.zeros(len(val_idx))
            for name, learner_cls in learners:
                import copy
                learner = copy.deepcopy(learner_cls)
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore")
                    learner.fit(X_tr, y_tr)
                fold_preds += learner.predict_proba(X_val)[:, 1]
            fold_preds /= len(learners)
            oof_preds[val_idx] = fold_preds

        # --- Étape 2 : Fit tous les base learners sur 100% ---
        print(f"  [2/3] Fit base learners sur full dataset...")
        fitted_base = []
        for name, learner_cls in learners:
            import copy
            learner = copy.deepcopy(learner_cls)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                learner.fit(X, y)
            fitted_base.append((name, learner))
            print(f"    ✅ {name} fitté")

        # --- Meta-learner : LogisticRegression sur OOF ---
        meta_X = oof_preds.reshape(-1, 1)
        meta = LogisticRegression(C=1.0, max_iter=2000)
        meta.fit(meta_X, y)

        self.model = {
            "base_learners": fitted_base,
            "meta": meta,
            "target": target,
        }
        print(f"  ✅ Stacking complet ({len(fitted_base)} learners + meta)")

        # --- Étape 3 : Calibration Platt ---
        print(f"  [3/3] Calibration Platt...")
        self.calibrated = self._calibrate(X, y)

        return self

    def _fit_fallback(self, X, y, target):
        if HAS_HGB:
            m = _make_hgb(target)
        elif HAS_LGB:
            m = _make_lgb(target)
        elif HAS_XGB:
            m = _make_xgb(target)
        else:
            from sklearn.linear_model import LogisticRegression
            m = LogisticRegression(max_iter=2000)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            m.fit(X, y)
        # Wrap comme un seul base learner
        self.model = {
            "base_learners": [("fallback", m)],
            "meta": None,
            "target": target,
        }
        self.calibrated = None
        print(f"  ✅ Fallback modèle unique fitté")
        return self

    def _calibrate(self, X, y):
        """Calibration Platt (sigmoid) sur split 80/20."""
        n = len(X)
        split = int(n * 0.8)
        X_cal, y_cal = X[split:], y[split:]

        if len(X_cal) < 30 or not _both_classes(y_cal):
            print(f"    Skip (cal set trop petit ou 1 classe)")
            return None

        preds_cal = self._raw_predict_proba(X_cal)

        # Fit Platt scaling avec CalibratedClassifierCV
        try:
            from sklearn.calibration import _SigmoidCalibration
            calibrator = _SigmoidCalibration()
            calibrator.fit(preds_cal, y_cal)
            # Evaluate
            calibrated_preds = calibrator.predict(preds_cal)
            brier = brier_score_loss(y_cal, calibrated_preds)
            raw_brier = brier_score_loss(y_cal, preds_cal)
            print(f"    Brier raw={raw_brier:.4f} → calibrated={brier:.4f}")
            if brier < raw_brier:
                return calibrator
            else:
                print(f"    Calibration ne améliore pas, skip")
                return None
        except Exception as e:
            print(f"    Calibration échouée: {e}")
            return None

    def _raw_predict_proba(self, X):
        """Prédiction brute (avant calibration)."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        base_preds = []
        for name, learner in self.model["base_learners"]:
            base_preds.append(learner.predict_proba(X)[:, 1])

        avg_preds = np.mean(base_preds, axis=0)

        meta = self.model.get("meta")
        if meta:
            meta_X = avg_preds.reshape(-1, 1)
            return meta.predict_proba(meta_X)[:, 1]

        return avg_preds

    def predict_proba(self, X):
        """Prédiction calibrée si possible, sinon brute."""
        raw = self._raw_predict_proba(X)
        if self.calibrated is not None:
            return self.calibrated.predict(raw)
        return raw

    def predict_one(self, x):
        return float(self.predict_proba(np.asarray(x).reshape(1, -1))[0])

    def feature_importance(self, feature_names=None):
        """Retourne l'importance moyenne des features (si XGB ou LGB dispo)."""
        importances = []
        for name, learner in self.model.get("base_learners", []):
            if hasattr(learner, "feature_importances_"):
                importances.append(learner.feature_importances_)
        if not importances:
            return None
        avg = np.mean(importances, axis=0)
        names = feature_names or [f"f{i}" for i in range(len(avg))]
        ranked = sorted(zip(names, avg), key=lambda x: -x[1])
        return ranked

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "calibrated": self.calibrated,
            "version": "v7",
        }, path)
        print(f"  💾 Sauvegardé → {path}")

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls()
        obj.model = data["model"]
        obj.calibrated = data.get("calibrated")
        return obj


# ── TabNetV7 : neural tabulaire ────────────────────────────────

class TabNetV7:
    """
    Modèle TabNet léger pour données tabulaires hippiques.
    - Architecture compacte (n_d=n_a=16, 2 shared layers)
    - Entraînement avec early stopping
    - Calibration Platt en post-processing
    """

    def __init__(self):
        self.model = None
        self.calibrated = None

    def fit(self, X, y, target="win"):
        if not HAS_TABNET:
            print(f"  [TabNet] ⚠️ PyTorch/TabNet non disponible, skip")
            return None

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        n, d = X.shape
        n_pos = int(np.sum(y))

        print(f"\n[TabNetV7] === Target: {target.upper()} ===")
        print(f"  Dataset : {n} samples × {d} features, {n_pos} positifs")

        if n < 200 or n_pos < 15:
            print(f"  ⚠️  Dataset insuffisant pour TabNet (min 200/15+)")
            return None

        # Split 80/20 pour validation + early stopping
        split = int(n * 0.8)
        indices = np.random.RandomState(42).permutation(n)
        train_idx, val_idx = indices[:split], indices[split:]

        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        if not _both_classes(y_val) or not _both_classes(y_train):
            print(f"  ⚠️  Classes déséquilibrées, skip")
            return None

        max_epochs = 150
        patience = 15

        model = TabNetClassifier(
            n_d=16, n_a=16,
            n_steps=4,
            gamma=1.5,
            n_independent=2, n_shared=2,
            cat_idxs=[],
            cat_dims=[],
            cat_emb_dim=[],
            lambda_sparse=1e-4,
            momentum=0.3,
            clip_value=2.0,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            scheduler_params={"step_size": 30, "gamma": 0.9},
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            mask_type="entmax",
            seed=42,
            verbose=0,
        )

        print(f"  Entraînement TabNet (max {max_epochs} epochs, patience {patience})...")
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            eval_name=["val"],
            eval_metric=["logloss"],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=min(256, n // 4),
            virtual_batch_size=min(128, n // 8),
        )

        self.model = model
        self.calibrated = self._calibrate(X_val, y_val)
        print(f"  ✅ TabNet fitté")
        return self

    def _calibrate(self, X, y):
        try:
            raw = self.model.predict_proba(X)[:, 1]
            from sklearn.calibration import _SigmoidCalibration
            cal = _SigmoidCalibration()
            cal.fit(raw, y)
            cal_preds = cal.predict(raw)
            brier = brier_score_loss(y, cal_preds)
            print(f"    TabNet calibration Brier={brier:.4f}")
            return cal
        except Exception as e:
            print(f"    TabNet calibration skip: {e}")
            return None

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        raw = self.model.predict_proba(X)[:, 1]
        if self.calibrated:
            return self.calibrated.predict(raw)
        return raw

    def predict_one(self, x):
        return float(self.predict_proba(np.asarray(x, dtype=np.float32).reshape(1, -1))[0])

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # TabNet utilise son propre format de sauvegarde
        model_dir = path.replace(".pkl", "_tabnet")
        os.makedirs(model_dir, exist_ok=True)
        self.model.save_model(os.path.join(model_dir, "tabnet"))
        joblib.dump({
            "calibrated": self.calibrated,
            "model_dir": model_dir,
            "version": "v7_tabnet",
        }, path)
        print(f"  💾 TabNet sauvegardé → {path}")

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls()
        obj.calibrated = data.get("calibrated")
        model_dir = data.get("model_dir", path.replace(".pkl", "_tabnet"))
        model_path = os.path.join(model_dir, "tabnet.zip")
        if os.path.exists(model_path):
            obj.model = TabNetClassifier()
            obj.model.load_model(model_path)
        return obj


# ── EnsembleV7 : combine stacking + tabnet ─────────────────────

class EnsembleV7:
    """
    Super-ensemble :
    - Stacking XGB+LGB+HGB (poids 0.7)
    - TabNet neural (poids 0.3)
    - Calibration finale Platt
    Compatible predict_one() avec le reste de l'app.
    """

    def __init__(self):
        self.stacking = None
        self.tabnet = None
        self.w_stack = 0.7
        self.w_tabnet = 0.3
        self.calibrated = None

    def fit(self, X, y, target="win"):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)

        t0 = time.time()

        # 1) Stacking
        self.stacking = StackingV7()
        self.stacking.fit(X, y, target=target)

        # 2) TabNet (optionnel)
        self.tabnet = TabNetV7()
        tn_result = self.tabnet.fit(X, y, target=target)
        if tn_result is None:
            self.tabnet = None
            self.w_stack = 1.0
            self.w_tabnet = 0.0

        elapsed = time.time() - t0
        mode = f"stack({self.w_stack:.0%}) + tabnet({self.w_tabnet:.0%})" if self.tabnet else "stack seul"
        print(f"\n  ⏱️  {target.upper()} entraîné en {elapsed:.1f}s — mode: {mode}")

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        probs = self.stacking.predict_proba(X)

        if self.tabnet is not None:
            tn_probs = self.tabnet.predict_proba(X)
            # Harmoniser les dtypes
            probs = np.asarray(probs, dtype=np.float64)
            tn_probs = np.asarray(tn_probs, dtype=np.float64)
            probs = self.w_stack * probs + self.w_tabnet * tn_probs

        return probs

    def predict_one(self, x):
        return float(self.predict_proba(np.asarray(x).reshape(1, -1))[0])

    def feature_importance(self, feature_names=None):
        return self.stacking.feature_importance(feature_names)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {
            "w_stack": self.w_stack,
            "w_tabnet": self.w_tabnet,
            "has_tabnet": self.tabnet is not None,
            "version": "v7_ensemble",
        }
        # Sauvegarder le stacking
        stack_path = path.replace(".pkl", "_stack.pkl")
        self.stacking.save(stack_path)
        data["stack_path"] = stack_path

        # Sauvegarder tabnet si dispo
        if self.tabnet is not None:
            tn_path = path.replace(".pkl", "_tabnet.pkl")
            self.tabnet.save(tn_path)
            data["tabnet_path"] = tn_path

        joblib.dump(data, path)
        print(f"  💾 EnsembleV7 sauvegardé → {path}")

    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls()
        obj.w_stack = data.get("w_stack", 0.7)
        obj.w_tabnet = data.get("w_tabnet", 0.3)

        stack_path = data.get("stack_path", path.replace(".pkl", "_stack.pkl"))
        obj.stacking = StackingV7.load(stack_path)

        if data.get("has_tabnet"):
            tn_path = data.get("tabnet_path", path.replace(".pkl", "_tabnet.pkl"))
            if os.path.exists(tn_path):
                try:
                    obj.tabnet = TabNetV7.load(tn_path)
                except Exception as e:
                    print(f"  [EnsembleV7] TabNet load failed: {e}")
                    obj.tabnet = None
                    obj.w_stack = 1.0
                    obj.w_tabnet = 0.0

        return obj


# ── API publique (compatible app.py) ──────────────────────────

def train_v7(X, y, save_path, target="win"):
    """Entraîne un EnsembleV7 et le sauvegarde."""
    model = EnsembleV7()
    model.fit(X, y, target=target)
    model.save(save_path)
    return model


def load_v7(path):
    """Charge un modèle EnsembleV7 sauvegardé."""
    if not os.path.exists(path):
        return None
    try:
        return EnsembleV7.load(path)
    except Exception as e:
        print(f"  [load_v7] Erreur: {e}")
        return None
