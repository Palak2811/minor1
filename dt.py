import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import math

# -----------------------------
# === (A) Your Tree Classes ===
# -----------------------------
# KL tree (same logic as you provided)
def kl_divergence(p, q):
    p = np.array(p, dtype=float)
    q = np.array(q, dtype=float)
    mask = (p > 0) & (q > 0)
    return np.sum(p[mask] * np.log(p[mask] / q[mask]))

def kl_impurity(y_left, y_right):
    n_total = len(y_left) + len(y_right)
    classes = np.unique(np.concatenate([y_left, y_right]))
    def class_dist(y):
        counts = np.array([np.sum(y == c) for c in classes], dtype=float)
        if counts.sum() == 0:
            return np.ones_like(counts) / len(counts)
        return counts / counts.sum()
    p_parent = class_dist(np.concatenate([y_left, y_right]))
    p_left = class_dist(y_left)
    p_right = class_dist(y_right)
    return (len(y_left) / n_total) * kl_divergence(p_left, p_parent) + \
           (len(y_right) / n_total) * kl_divergence(p_right, p_parent)

class KLDecisionTree:
    def __init__(self, max_depth=4, min_samples_split=10, n_random_directions=20, random_state=None):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_random_directions = n_random_directions
        self.random_state = np.random.RandomState(random_state)
        self.tree_ = None

    def best_split(self, X, y):
        best_score = -np.inf
        best_params = None
        n_samples, n_features = X.shape
        for _ in range(self.n_random_directions):
            f = self.random_state.randint(0, n_features)
            thresholds = np.unique(X[:, f])
            for t in thresholds:
                left_mask = X[:, f] <= t
                right_mask = X[:, f] > t
                if len(y[left_mask]) < self.min_samples_split or len(y[right_mask]) < self.min_samples_split:
                    continue
                score = kl_impurity(y[left_mask], y[right_mask])
                if score > best_score:
                    best_score = score
                    best_params = (f, t)
        return best_params

    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y, minlength=np.max(y)+1).argmax()
        split = self.best_split(X, y)
        if split is None:
            return np.bincount(y, minlength=np.max(y)+1).argmax()
        f, t = split
        left_mask = X[:, f] <= t
        right_mask = X[:, f] > t
        left_child = self.build_tree(X[left_mask], y[left_mask], depth+1)
        right_child = self.build_tree(X[right_mask], y[right_mask], depth+1)
        return {'feature': f, 'threshold': t, 'left': left_child, 'right': right_child}

    def fit(self, X, y):
        self.tree_ = self.build_tree(X, y)
        return self

    def _predict_one(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        else:
            return self._predict_one(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in X])

# Tsallis tree (same logic as you provided)
def tsallis_impurity(y, q=1.5):
    classes, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    if q == 1:
        # Shannon
        return -np.sum(probs * np.log2(probs))
    else:
        return (1 - np.sum(probs ** q)) / (q - 1)

class TsallisDecisionTree:
    def __init__(self, max_depth=5, min_samples_split=2, q=1.5):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.q = q
        self.tree_ = None

    def best_split(self, X, y):
        best_score = -np.inf
        best_split = None
        n_features = X.shape[1]
        for feature in range(n_features):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                left_mask = X[:, feature] <= threshold
                right_mask = X[:, feature] > threshold
                if len(y[left_mask]) < self.min_samples_split or len(y[right_mask]) < self.min_samples_split:
                    continue
                parent_impurity = tsallis_impurity(y, q=self.q)
                left_impurity = tsallis_impurity(y[left_mask], q=self.q)
                right_impurity = tsallis_impurity(y[right_mask], q=self.q)
                n = len(y)
                score = parent_impurity - (len(y[left_mask]) / n) * left_impurity - (len(y[right_mask]) / n) * right_impurity
                if score > best_score:
                    best_score = score
                    best_split = (feature, threshold)
        return best_split

    def build_tree(self, X, y, depth=0):
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            return np.bincount(y, minlength=np.max(y)+1).argmax()
        split = self.best_split(X, y)
        if split is None:
            return np.bincount(y, minlength=np.max(y)+1).argmax()
        feature, threshold = split
        left_mask = X[:, feature] <= threshold
        right_mask = X[:, feature] > threshold
        left_child = self.build_tree(X[left_mask], y[left_mask], depth+1)
        right_child = self.build_tree(X[right_mask], y[right_mask], depth+1)
        return {"feature": feature, "threshold": threshold, "left": left_child, "right": right_child}

    def fit(self, X, y):
        self.tree_ = self.build_tree(X, y)
        return self

    def _predict_one(self, x, node):
        if not isinstance(node, dict):
            return node
        if x[node["feature"]] <= node["threshold"]:
            return self._predict_one(x, node["left"])
        else:
            return self._predict_one(x, node["right"])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree_) for x in X])


# -----------------------------
# === (B) Boosting Procedure ===
# -----------------------------
def boosting_train(X_train, y_train, n_rounds=20, sample_size=None,
                   max_depth=3, tsallis_q=1.3, min_samples_split=2, random_state=42):
    """
    Train boosted ensemble alternating KL and Tsallis trees.
    We use importance resampling with sample weights to train each weak learner.
    Returns: list of (model, alpha)
    """
    rng = np.random.RandomState(random_state)
    n_samples = X_train.shape[0]
    if sample_size is None:
        sample_size = n_samples

    n_classes = len(np.unique(y_train))
    weights = np.ones(n_samples) / n_samples
    ensemble = []

    for m in range(n_rounds):
        # pick learner type: alternate to get ~50% each
        if m % 2 == 0:
            learner_type = 'kl'
            model = KLDecisionTree(max_depth=max_depth, min_samples_split=min_samples_split, random_state=random_state)
        else:
            learner_type = 'tsallis'
            model = TsallisDecisionTree(max_depth=max_depth, min_samples_split=min_samples_split, q=tsallis_q)

        # importance resampling according to weights
        idx = rng.choice(np.arange(n_samples), size=sample_size, replace=True, p=weights/weights.sum())
        X_sample = X_train[idx]
        y_sample = y_train[idx]

        # fit weak learner
        model.fit(X_sample, y_sample)

        # predictions on training set (full)
        preds = model.predict(X_train)

        # weighted error
        incorrect = (preds != y_train).astype(float)
        err = np.sum(weights * incorrect) / np.sum(weights)

        # handle edge cases
        # If error >= 1 - 1/n_classes, break (weak learner no better than random)
        if err >= 1.0 - 1.0 / n_classes:
            # skip adding this weak learner
            # optionally: continue or break
            # break to stop boosting early
            print(f"Round {m}: weak learner too weak (err={err:.4f}), stopping early.")
            break

        # SAMME alpha for discrete multiclass boosting:
        # alpha = ln((1 - err)/err) + ln(K - 1)
        # reference: Zhu et al., 2009 (SAMME)
        # guard against zero error
        err = max(err, 1e-12)
        alpha = math.log((1.0 - err) / err) + math.log(n_classes - 1.0)

        # store model and its weight
        ensemble.append((model, alpha))

        # update weights
        # increase weights of misclassified samples
        # w_i <- w_i * exp(alpha * I(y_i != h(x_i)))
        weights = weights * np.exp(alpha * incorrect)

        # normalize
        weights = weights / weights.sum()

        # debug print
        print(f"Round {m+1}/{n_rounds} | Learner: {learner_type} | err: {err:.4f} | alpha: {alpha:.4f}")

    return ensemble

def boosting_predict(ensemble, X, n_classes):
    """
    Predict using weighted vote (alpha-weighted) for discrete boosting (SAMME).
    ensemble: list of (model, alpha)
    """
    n_samples = X.shape[0]
    # accumulate score per class
    scores = np.zeros((n_samples, n_classes))
    for model, alpha in ensemble:
        preds = model.predict(X)
        for i, p in enumerate(preds):
            scores[i, int(p)] += alpha
    # choose class with highest score
    return np.argmax(scores, axis=1)


# -----------------------------
# === (C) Load data & run ===
# -----------------------------
# Load dataset (change path if needed)
df = pd.read_csv("E:/fortransferee/mlproject6-p/Maternal Health Risk Data Set.csv")
X = df.drop(columns=['RiskLevel']).values
y_raw = df['RiskLevel'].values

le = LabelEncoder()
y = le.fit_transform(y_raw)
n_classes = len(le.classes_)

X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train boosted ensemble
ensemble = boosting_train(
    X_tr, y_tr,
    n_rounds=20,            # number of weak learners (try 20-100)
    sample_size=len(X_tr),  # sample size per round (same as training size)
    max_depth=3,
    tsallis_q=1.3,
    min_samples_split=5,
    random_state=42
)

# Predict & evaluate
y_pred_train = boosting_predict(ensemble, X_tr, n_classes=n_classes)
y_pred_test = boosting_predict(ensemble, X_te, n_classes=n_classes)

print("\n=== Results ===")
print("Train Accuracy:", accuracy_score(y_tr, y_pred_train))
print("Test Accuracy :", accuracy_score(y_te, y_pred_test))
print("Test F1 (macro):", f1_score(y_te, y_pred_test, average='macro'))
