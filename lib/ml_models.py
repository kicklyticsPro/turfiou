"""
v4 - Ensemble Random Forest + Gradient Boosting (NumPy pur).
"""
import math
import random


class DecisionStump:
    """Arbre CART régression sur résidus (utilisé par GBM)."""
    def __init__(self, max_depth=3, min_samples=20):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None

    def fit(self, X, y, sample_weight=None):
        import numpy as np
        self.tree = self._build(np.asarray(X, dtype=float), np.asarray(y, dtype=float), depth=0)

    def _build(self, X, y, depth):
        import numpy as np
        n = len(y)
        if n < self.min_samples or depth >= self.max_depth:
            return {"leaf": True, "value": float(np.mean(y)) if n else 0.0}
        best_split = None
        best_loss = float('inf')
        n_features = X.shape[1]
        for f in range(n_features):
            vals = X[:, f]
            quantiles = np.percentile(vals, [10, 20, 30, 40, 50, 60, 70, 80, 90])
            for q in np.unique(quantiles):
                left_mask = vals <= q
                right_mask = ~left_mask
                nl, nr = left_mask.sum(), right_mask.sum()
                if nl < self.min_samples or nr < self.min_samples:
                    continue
                yl, yr = y[left_mask], y[right_mask]
                loss = nl * np.var(yl) + nr * np.var(yr)
                if loss < best_loss:
                    best_loss = loss
                    best_split = (f, float(q), left_mask, right_mask)
        if best_split is None:
            return {"leaf": True, "value": float(np.mean(y))}
        f, q, lm, rm = best_split
        return {"leaf": False, "feature": f, "threshold": q,
                "left": self._build(X[lm], y[lm], depth + 1),
                "right": self._build(X[rm], y[rm], depth + 1)}

    def predict_one(self, x):
        node = self.tree
        while not node["leaf"]:
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["value"]

    def predict(self, X):
        import numpy as np
        return np.array([self.predict_one(x) for x in X])


class GradientBoosting:
    """GBM binaire log-loss."""
    def __init__(self, n_trees=50, max_depth=3, learning_rate=0.1):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.lr = learning_rate
        self.trees = []
        self.init_pred = 0.0

    def fit(self, X, y):
        import numpy as np
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        mean_y = max(min(y.mean(), 0.99), 0.01)
        self.init_pred = math.log(mean_y / (1 - mean_y))
        F = np.full(len(y), self.init_pred)
        for _ in range(self.n_trees):
            p = 1 / (1 + np.exp(-F))
            residuals = y - p
            tree = DecisionStump(max_depth=self.max_depth)
            tree.fit(X, residuals)
            pred = tree.predict(X)
            F = F + self.lr * pred
            self.trees.append(tree)
        return self

    def predict_one(self, x):
        F = self.init_pred
        for tree in self.trees:
            F += self.lr * tree.predict_one(x)
        return 1 / (1 + math.exp(-F))

    def predict_proba(self, X):
        import numpy as np
        return np.array([self.predict_one(x) for x in X])

    def to_dict(self):
        return {"type": "gbm", "n_trees": self.n_trees, "max_depth": self.max_depth,
                "lr": self.lr, "init_pred": self.init_pred,
                "trees": [t.tree for t in self.trees]}

    @classmethod
    def from_dict(cls, d):
        m = cls(n_trees=d["n_trees"], max_depth=d["max_depth"], learning_rate=d["lr"])
        m.init_pred = d["init_pred"]
        for tree_data in d["trees"]:
            ds = DecisionStump(max_depth=d["max_depth"])
            ds.tree = tree_data
            m.trees.append(ds)
        return m


class DecisionTreeClassifier:
    """Arbre de classification avec Gini, pour Random Forest."""
    def __init__(self, max_depth=10, min_samples=10, max_features=None):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.max_features = max_features  # nb features tirées au hasard à chaque split
        self.tree = None

    def fit(self, X, y):
        import numpy as np
        self.tree = self._build(np.asarray(X, dtype=float), np.asarray(y, dtype=float), 0)

    def _build(self, X, y, depth):
        import numpy as np
        n = len(y)
        if n < self.min_samples or depth >= self.max_depth or len(np.unique(y)) == 1:
            return {"leaf": True, "value": float(np.mean(y)) if n else 0.5}
        n_feat = X.shape[1]
        max_f = self.max_features or n_feat
        features_to_try = random.sample(range(n_feat), min(max_f, n_feat))

        best_split = None
        best_gini = float('inf')
        for f in features_to_try:
            vals = X[:, f]
            quantiles = np.percentile(vals, [20, 35, 50, 65, 80])
            for q in np.unique(quantiles):
                lm = vals <= q
                rm = ~lm
                nl, nr = lm.sum(), rm.sum()
                if nl < self.min_samples or nr < self.min_samples:
                    continue
                # Gini pondéré
                pl = y[lm].mean() if nl > 0 else 0
                pr = y[rm].mean() if nr > 0 else 0
                gini = nl * (2 * pl * (1 - pl)) + nr * (2 * pr * (1 - pr))
                if gini < best_gini:
                    best_gini = gini
                    best_split = (f, float(q), lm, rm)
        if best_split is None:
            return {"leaf": True, "value": float(np.mean(y))}
        f, q, lm, rm = best_split
        return {"leaf": False, "feature": f, "threshold": q,
                "left": self._build(X[lm], y[lm], depth + 1),
                "right": self._build(X[rm], y[rm], depth + 1)}

    def predict_one(self, x):
        node = self.tree
        while not node["leaf"]:
            if x[node["feature"]] <= node["threshold"]:
                node = node["left"]
            else:
                node = node["right"]
        return node["value"]


class RandomForest:
    """Random Forest : bagging d'arbres avec sous-échantillonnage."""
    def __init__(self, n_trees=30, max_depth=8, min_samples=15):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.trees = []
        self.n_features_used = 0

    def fit(self, X, y, seed=42):
        random.seed(seed)
        import numpy as np
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(y)
        n_feat = X.shape[1]
        # sqrt(n_features) features par split (heuristique standard)
        max_f = max(2, int(math.sqrt(n_feat)))
        self.n_features_used = max_f
        for i in range(self.n_trees):
            # Bagging : échantillonnage avec remise
            idx = [random.randint(0, n - 1) for _ in range(n)]
            X_boot = X[idx]
            y_boot = y[idx]
            tree = DecisionTreeClassifier(max_depth=self.max_depth,
                                          min_samples=self.min_samples,
                                          max_features=max_f)
            tree.fit(X_boot, y_boot)
            self.trees.append(tree)
        return self

    def predict_one(self, x):
        if not self.trees:
            return 0.5
        return sum(t.predict_one(x) for t in self.trees) / len(self.trees)

    def to_dict(self):
        return {"type": "rf", "n_trees": self.n_trees, "max_depth": self.max_depth,
                "min_samples": self.min_samples,
                "n_features_used": self.n_features_used,
                "trees": [t.tree for t in self.trees]}

    @classmethod
    def from_dict(cls, d):
        m = cls(n_trees=d["n_trees"], max_depth=d["max_depth"],
                min_samples=d["min_samples"])
        for tree_data in d["trees"]:
            t = DecisionTreeClassifier(max_depth=d["max_depth"],
                                       min_samples=d["min_samples"])
            t.tree = tree_data
            m.trees.append(t)
        m.n_features_used = d.get("n_features_used", 0)
        return m


class Ensemble:
    """Combine GBM + RandomForest avec pondération."""
    def __init__(self, gbm=None, rf=None, w_gbm=0.5, w_rf=0.5):
        self.gbm = gbm
        self.rf = rf
        self.w_gbm = w_gbm
        self.w_rf = w_rf

    def predict_one(self, x):
        if self.gbm and self.rf:
            return self.w_gbm * self.gbm.predict_one(x) + self.w_rf * self.rf.predict_one(x)
        if self.gbm:
            return self.gbm.predict_one(x)
        if self.rf:
            return self.rf.predict_one(x)
        return 0.5

    def to_dict(self):
        return {"type": "ensemble",
                "gbm": self.gbm.to_dict() if self.gbm else None,
                "rf": self.rf.to_dict() if self.rf else None,
                "w_gbm": self.w_gbm, "w_rf": self.w_rf}

    @classmethod
    def from_dict(cls, d):
        gbm = GradientBoosting.from_dict(d["gbm"]) if d.get("gbm") else None
        rf = RandomForest.from_dict(d["rf"]) if d.get("rf") else None
        return cls(gbm=gbm, rf=rf, w_gbm=d.get("w_gbm", 0.5), w_rf=d.get("w_rf", 0.5))


def load_model_from_dict(d):
    """Charge n'importe quel type de modèle."""
    if not d:
        return None
    t = d.get("type", "gbm")
    if t == "ensemble":
        return Ensemble.from_dict(d)
    if t == "rf":
        return RandomForest.from_dict(d)
    return GradientBoosting.from_dict(d)


# ============================================================
#  Calibration isotone (Pool-Adjacent-Violators)
# ============================================================
def fit_isotonic(predictions, actuals, n_bins=20):
    pairs = sorted(zip(predictions, actuals))
    if not pairs:
        return [(0.0, 0.0), (1.0, 1.0)]
    bin_size = max(1, len(pairs) // n_bins)
    bins = []
    for i in range(0, len(pairs), bin_size):
        batch = pairs[i:i+bin_size]
        if not batch:
            continue
        avg_pred = sum(p for p, _ in batch) / len(batch)
        avg_actual = sum(a for _, a in batch) / len(batch)
        bins.append((avg_pred, avg_actual))
    # PAV simplifié
    calibrated = []
    for pred, actual in bins:
        while calibrated and calibrated[-1][1] > actual:
            prev_pred, prev_actual = calibrated.pop()
            pred = (prev_pred + pred) / 2
            actual = (prev_actual + actual) / 2
        calibrated.append((pred, actual))
    return calibrated


def apply_calibration(p, calib_table):
    if not calib_table:
        return p
    if p <= calib_table[0][0]:
        return calib_table[0][1]
    if p >= calib_table[-1][0]:
        return calib_table[-1][1]
    for i in range(len(calib_table) - 1):
        x0, y0 = calib_table[i]
        x1, y1 = calib_table[i+1]
        if x0 <= p <= x1:
            if x1 == x0:
                return y0
            return y0 + (y1 - y0) * (p - x0) / (x1 - x0)
    return p
