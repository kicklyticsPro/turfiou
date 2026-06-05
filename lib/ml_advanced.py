"""
ml_advanced.py - Stacking v5 pour Turfiou
Améliorations vs v4 :
- Diversité : LightGBM + CatBoost + HistGB + RandomForest + Logistic
- Validation temporelle (TimeSeriesSplit) au lieu de fit sur tout
- Calibration double : Platt (sigmoid) + Isotone, choisie par Brier
- Meta-learner Ridge Logistic
- Compatible avec l'interface predict_one() existante
"""
import numpy as np
import joblib
import os
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import brier_score_loss, log_loss
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
        
        # TimeSeriesSplit - crucial pour turf
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Base learners diversifiés
        estimators = []
        
        # 1. LightGBM - capture non-linéarités rapides
        if HAS_LGB:
            estimators.append(('lgb', LGBMClassifier(
                n_estimators=800,
                learning_rate=0.03,
                max_depth=5,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )))
        
        # 2. CatBoost - gère interactions catégorielles
        if HAS_CAT:
            estimators.append(('cat', CatBoostClassifier(
                iterations=600,
                depth=6,
                learning_rate=0.04,
                l2_leaf_reg=3,
                random_seed=42,
                verbose=False,
                allow_writing_files=False
            )))
        
        # 3. HistGradientBoosting - stable, sklearn natif
        estimators.append(('hgb', HistGradientBoostingClassifier(
            max_iter=500,
            learning_rate=0.05,
            max_depth=6,
            l2_regularization=0.1,
            random_state=42
        )))
        
        # 4. RandomForest - réduit variance
        estimators.append(('rf', RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=20,
            max_features='sqrt',
            n_jobs=-1,
            random_state=42
        )))
        
        # 5. Logistic lissé - ancre linéaire
        estimators.append(('lr', Pipeline([
            ('scaler', StandardScaler()),
            ('lr', LogisticRegression(C=0.5, max_iter=1000, n_jobs=-1))
        ])))
        
        # Meta-learner
        final_estimator = LogisticRegression(C=1.0, max_iter=2000)
        
        stack = StackingClassifier(
            estimators=estimators,
            final_estimator=final_estimator,
            cv=tscv,
            n_jobs=-1,
            passthrough=False,  # on ne passe pas les features brutes
            stack_method='predict_proba'
        )
        
        print(f"[Advanced] Entraînement stacking avec {len(estimators)} modèles...")
        stack.fit(X, y)
        self.model = stack
        
        # Calibration : on teste Platt vs Isotone
        print("[Advanced] Calibration temporelle...")
        # On calibre sur les prédictions out-of-fold
        calibrated_sigmoid = CalibratedClassifierCV(
            stack, method='sigmoid', cv=tscv
        )
        calibrated_isotonic = CalibratedClassifierCV(
            stack, method='isotonic', cv=tscv
        )
        
        calibrated_sigmoid.fit(X, y)
        calibrated_isotonic.fit(X, y)
        
        # Choix par Brier score en CV
        pred_sig = calibrated_sigmoid.predict_proba(X)[:,1]
        pred_iso = calibrated_isotonic.predict_proba(X)[:,1]
        
        brier_sig = brier_score_loss(y, pred_sig)
        brier_iso = brier_score_loss(y, pred_sig)
        
        # Isotone souvent meilleur mais surfit si <5k samples
        if len(X) > 5000 and brier_iso < brier_sig:
            self.calibrated = calibrated_isotonic
            print(f"[Advanced] Calibration choisie: ISOTONIC (Brier {brier_iso:.4f} vs {brier_sig:.4f})")
        else:
            self.calibrated = calibrated_sigmoid
            print(f"[Advanced] Calibration choisie: PLATT (Brier {brier_sig:.4f})")
        
        return self
    
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
            'version': 'v5'
        }, path)
    
    @classmethod
    def load(cls, path):
        data = joblib.load(path)
        obj = cls()
        obj.model = data['model']
        obj.calibrated = data['calibrated']
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