"""
ml_advanced.py - Stacking v6 pour Turfiou
Améliorations vs v5 :
- CV : StratifiedKFold préserve l'équilibre des classes
  (sklearn 1.6+ refuse TimeSeriesSplit dans cross_val_predict)
- Calibration sécurisée : split manuel (80/20), FrozenEstimator, pas de cross_val_predict
- Fallback automatique si trop peu de données ou classes déséquilibrées
- Compatible avec l'interface predict_one() existante
"""
import numpy as np
import joblib
import os
import warnings
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

try:
    from lightgbm import LGBMClassifier
    HAS_LGB = True
except:
    HAS_LGB = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except:
    HAS_CAT = False

try:
    from sklearn.frozen import FrozenEstimator
    HAS_FROZEN = True
except ImportError:
    HAS_FROZEN = False


def _safe_cv(X, y, n_splits=5):
    """Retourne un StratifiedKFold adapté à la taille du dataset.
    
    StratifiedKFold garantit que chaque fold contient les 2 classes,
    ce qui évite le crash cross_val_predict avec des données déséquilibrées.
    """
    n = len(X)
    n_pos = int(np.sum(y))
    n_neg = n - n_pos
    
    # Chaque fold doit avoir au moins 2 positifs et 10 négatifs
    max_by_pos = max(1, n_pos // 2)
    max_by_neg = max(1, n_neg // 10)
    max_by_size = max(2, n // 30)
    actual = min(n_splits, max_by_pos, max_by_neg, max_by_size)
    actual = max(actual, 2)
    actual = min(actual, 5)
    
    print(f"  [CV] {n} samples ({n_pos}+/{n_neg}-) → StratifiedKFold(n_splits={actual})")
    return StratifiedKFold(n_splits=actual, shuffle=True, random_state=42)


def _has_both_classes(y):
    """Vérifie qu'un array contient au moins 2 classes."""
    return len(set(np.asarray(y).tolist())) >= 2


class AdvancedEnsemble:
    """Wrapper compatible avec l'ancien code (predict_one)"""
    def __init__(self):
        self.model = None
        self.calibrated = None
        self.feature_names = None
        self.scaler = None
        
    def fit(self, X, y, dates=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        
        n = len(X)
        n_pos = int(np.sum(y))
        
        # Sécurité : minimum absolu
        if n < 50 or n_pos < 5:
            print(f"[Advanced] ⚠️ Dataset trop petit ({n} samples, {n_pos} positifs), fallback simple")
            return self._fit_simple(X, y)
        
        # CV adaptatif (stratifié pour équilibre des classes)
        cv = _safe_cv(X, y, n_splits=5)
        
        # Base learners
        estimators = self._build_estimators()
        
        print(f"[Advanced] Entraînement stacking avec {len(estimators)} modèles sur {n} samples ({n_pos}+)...")
        
        # --- Étape 1 : fit du StackingClassifier ---
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*Number of classes.*")
            warnings.filterwarnings("ignore", message=".*L-BFGS-B.*")
            
            stack = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(C=1.0, max_iter=2000),
                cv=cv,
                n_jobs=1,
                passthrough=False,
                stack_method='predict_proba'
            )
            stack.fit(X, y)
        
        self.model = stack
        print("[Advanced] ✅ Stacking fitté")
        
        # --- Étape 2 : Calibration sécurisée ---
        print("[Advanced] Calibration...")
        try:
            self.calibrated = self._calibrate_split(X, y)
        except Exception as e:
            print(f"  [Calibration] Échec ({e}), modèle brut")
            self.calibrated = None
        
        return self
    
    def _fit_simple(self, X, y):
        """Fallback pour petits datasets : un seul HGB + Logistic."""
        print("[Advanced] Fallback : HistGradientBoosting seul")
        model = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.05, max_depth=4, random_state=42
        )
        model.fit(X, y)
        self.model = model
        self.calibrated = None
        return self
    
    def _build_estimators(self):
        """Construit la liste des base learners."""
        estimators = []
        
        if HAS_LGB:
            estimators.append(('lgb', LGBMClassifier(
                n_estimators=800, learning_rate=0.03, max_depth=5,
                num_leaves=31, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1, random_state=42,
                n_jobs=-1, verbose=-1
            )))
        
        if HAS_CAT:
            estimators.append(('cat', CatBoostClassifier(
                iterations=600, depth=6, learning_rate=0.04,
                l2_leaf_reg=3, random_seed=42, verbose=False,
                allow_writing_files=False
            )))
        
        estimators.append(('hgb', HistGradientBoostingClassifier(
            max_iter=500, learning_rate=0.05, max_depth=6,
            l2_regularization=0.1, random_state=42
        )))
        
        estimators.append(('rf', RandomForestClassifier(
            n_estimators=300, max_depth=10, min_samples_leaf=20,
            max_features='sqrt', n_jobs=-1, random_state=42
        )))
        
        estimators.append(('lr', Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(C=0.5, max_iter=1000, n_jobs=-1))
        ])))
        
        return estimators
    
    def _calibrate_split(self, X, y):
        """Calibration par split 80/20 avec cv='prefit'.
        
        Pas de cross_val_predict : on splitte manuellement,
        on re-fit un stacking sur 80%, puis on calibre sur 20%.
        """
        n = len(X)
        split_idx = int(n * 0.8)
        
        X_train, X_cal = X[:split_idx], X[split_idx:]
        y_train, y_cal = y[:split_idx], y[split_idx:]
        
        if len(X_cal) < 20:
            print(f"  [Calibration] Calibration set trop petit ({len(X_cal)}), skip")
            return None
        if not _has_both_classes(y_cal):
            print(f"  [Calibration] 1 seule classe dans calibration set, skip")
            return None
        if not _has_both_classes(y_train):
            print(f"  [Calibration] 1 seule classe dans train set, skip")
            return None
        
        # Re-fit stacking sur 80%
        cv_sub = _safe_cv(X_train, y_train, n_splits=3)
        estimators = self._build_estimators()
        
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            stack_sub = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(C=1.0, max_iter=2000),
                cv=cv_sub,
                n_jobs=1,
                passthrough=False,
                stack_method='predict_proba'
            )
            stack_sub.fit(X_train, y_train)
        
        # Calibrer avec FrozenEstimator ou cv='prefit'
        if HAS_FROZEN:
            frozen = FrozenEstimator(stack_sub)
            calib_sigmoid = CalibratedClassifierCV(frozen, method='sigmoid')
            calib_isotonic = CalibratedClassifierCV(frozen, method='isotonic')
        else:
            calib_sigmoid = CalibratedClassifierCV(stack_sub, method='sigmoid', cv='prefit')
            calib_isotonic = CalibratedClassifierCV(stack_sub, method='isotonic', cv='prefit')
        
        try:
            calib_sigmoid.fit(X_cal, y_cal)
        except Exception as e:
            print(f"  [Calibration] Sigmoid failed: {e}")
            calib_sigmoid = None
        
        try:
            calib_isotonic.fit(X_cal, y_cal)
        except Exception as e:
            print(f"  [Calibration] Isotonic failed: {e}")
            calib_isotonic = None
        
        # Choix par Brier score
        best = None
        best_brier = 999.0
        best_name = "none"
        
        for name, calib in [("PLATT", calib_sigmoid), ("ISOTONIC", calib_isotonic)]:
            if calib is None:
                continue
            try:
                pred = calib.predict_proba(X_cal)[:, 1]
                brier = brier_score_loss(y_cal, pred)
                print(f"  [Calibration] {name}: Brier = {brier:.4f}")
                if brier < best_brier:
                    best_brier = brier
                    best = calib
                    best_name = name
            except Exception as e:
                print(f"  [Calibration] {name} eval failed: {e}")
        
        if best is not None:
            print(f"  [Calibration] ✅ Choisie: {best_name} (Brier {best_brier:.4f})")
        else:
            print(f"  [Calibration] ⚠️ Aucune calibration valide")
        
        return best
    
    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        if self.calibrated:
            return self.calibrated.predict_proba(X)[:,1]
        return self.model.predict_proba(X)[:,1]
    
    def predict_one(self, x):
        return float(self.predict_proba(x)[0])
    
    def save(self, path):
        joblib.dump({
            'model': self.model,
            'calibrated': self.calibrated,
            'version': 'v6'
        }, path)
    
    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls()
        obj.model = data['model']
        obj.calibrated = data.get('calibrated')
        return obj


def train_advanced(X, y, save_path):
    """Entraîne et sauvegarde le modèle avancé"""
    model = AdvancedEnsemble()
    model.fit(X, y)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    return model


def load_advanced(path):
    if os.path.exists(path):
        try:
            return AdvancedEnsemble.load(path)
        except:
            return None
    return None
