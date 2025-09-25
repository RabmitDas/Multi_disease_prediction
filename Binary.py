# A_binary_t2d_cleaned.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import shap
from sklearn.pipeline import Pipeline, FunctionTransformer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator, ClassifierMixin
from xgboost import XGBClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor


# --- RF-based imputer via IterativeImputer ---
rf_imputer = IterativeImputer(
    estimator=RandomForestRegressor(
        n_estimators=50,
        random_state=42,
        n_jobs=-1
    ),
    max_iter=10,
    random_state=42
)

# --------- DeLong implementation (adapted) ---------
def compute_midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1)
        i = j
    T2 = np.empty(N, dtype=float)
    T2[J] = T + 1
    return T2

def fastDeLong(preds_sorted_transposed, label_1_count):
    m = label_1_count
    n = preds_sorted_transposed.shape[1] - m
    positive = preds_sorted_transposed[:, :m]
    negative = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float)
    ty = np.empty([k, n], dtype=float)
    tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive[r, :])
        ty[r, :] = compute_midrank(negative[r, :])
        tz[r, :] = compute_midrank(preds_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.cov(v01)
    sy = np.cov(v10)
    s = sx / m + sy / n
    return aucs, s

def delong_roc_test(y_true, prob1, prob2):
    y_true = np.asarray(y_true)
    order = np.argsort(-prob1)
    label_1_count = int(np.sum(y_true))
    if label_1_count == 0 or label_1_count == len(y_true):
        return np.nan, (np.nan, np.nan)
    preds = np.vstack((prob1, prob2))
    try:
        aucs, cov = fastDeLong(preds[:, order], label_1_count)
        diff = aucs[0] - aucs[1]
        var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
        if var <= 0:
            return np.nan, (aucs[0], aucs[1])
        z = diff / np.sqrt(var)
        p = stats.norm.sf(abs(z)) * 2
        return p, (aucs[0], aucs[1])
    except Exception:
        return np.nan, (np.nan, np.nan)

# --------- Pareto scaler transformer ---------
def pareto_scale(X):
    X = np.asarray(X)
    mu = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    denom = np.sqrt(std)
    denom[denom == 0] = 1.0
    return (X - mu) / denom

pareto_transformer = FunctionTransformer(lambda X: pareto_scale(X), validate=False)

# --------- PLS-DA Class ---------
class PLS_DA(BaseEstimator, ClassifierMixin):
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pls = PLSRegression(n_components=n_components)

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.pls.fit(X, y)
        return self

    def predict(self, X):
        y_pred = self.pls.predict(X).ravel()
        return np.where(y_pred > 0.5, self.classes_[1], self.classes_[0])

    def predict_proba(self, X):
        y_pred = self.pls.predict(X).ravel()
        proba = (y_pred - y_pred.min()) / (y_pred.max() - y_pred.min() + 1e-9)
        return np.vstack([1 - proba, proba]).T

# --------- Load data ---------
df = pd.read_csv(r"D:/Downloads/T2D101.csv", header=None)
features = df.iloc[2:].set_index(0).T
X = features.apply(pd.to_numeric, errors='coerce').values
y_raw = df.iloc[1, 1:].values
le = LabelEncoder()
le.fit(["Control","T2D"])
y = le.transform(y_raw)

# --------- Preprocessing & Pipelines ---------
base_preproc = Pipeline([
    ("imputer", rf_imputer),
    ("log1p", FunctionTransformer(lambda Z: np.log1p(Z), validate=False)),
    ("pareto", pareto_transformer),
])

pipelines = {
    "Logistic Regression": Pipeline([("clf", LogisticRegression(max_iter=2000, random_state=42))]),
    "SVM (Linear)": Pipeline([("clf", SVC(kernel="linear", probability=True, random_state=42))]),
    "SVM (RBF)": Pipeline([("clf", SVC(kernel="rbf", probability=True, random_state=42))]),
    "Random Forest": Pipeline([("clf", RandomForestClassifier(n_estimators=200, random_state=42))]),
    "XGBoost": Pipeline([("clf", XGBClassifier(use_label_encoder=False, eval_metric="logloss", random_state=42))]),
    "kNN": Pipeline([("clf", KNeighborsClassifier(n_neighbors=5))]),
    "MLP": Pipeline([("clf", MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42))]),
    "PLS-DA": Pipeline([("clf", PLS_DA(n_components=5))])
}

# --------- Preprocessing QC Plots ---------
X_proc = base_preproc.fit_transform(features)
pca = PCA(n_components=2)
pcs = pca.fit_transform(X_proc)
plt.figure(figsize=(5,4))
for label in np.unique(y):
    idx = y == label
    plt.scatter(pcs[idx,0], pcs[idx,1], alpha=0.7, label=le.classes_[label])
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
plt.title("Panel B: PCA of preprocessed features")
plt.legend(); plt.tight_layout()
plt.savefig(r"D:/Downloads/T2D_Fig2B_PCA.png", dpi=1200); plt.close()

X_proc = base_preproc.fit_transform(X)

# --------- Cross-validated OOF probabilities & metrics ---------
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {}
oof_probs = {name: np.zeros(len(y)) for name in pipelines}

for name, pipe in pipelines.items():
    print("\n---", name, "---")
    probs = cross_val_predict(pipe, X_proc, y, cv=cv, method="predict_proba", n_jobs=1)
    pos_prob = probs[:, 1]
    oof_probs[name] = pos_prob

    preds = (pos_prob >= 0.5).astype(int)
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds)
    auc_global = roc_auc_score(y, pos_prob)
    ci_low, ci_high = np.percentile([roc_auc_score(
        y[np.random.RandomState(i).choice(len(y), len(y), replace=True)],
        oof_probs[name][np.random.RandomState(i).choice(len(y), len(y), replace=True)]
    ) for i in range(500)], [2.5, 97.5])
    results[name] = {"accuracy": acc, "f1": f1, "auc": auc_global, "ci": (ci_low, ci_high)}
    print(f"Acc {acc:.3f} F1 {f1:.3f} AUC {auc_global:.3f} (CI {ci_low:.3f}-{ci_high:.3f})")

# --------- Combined ROC Curve ---------
plt.figure(figsize=(8,6))
mean_fpr = np.linspace(0,1,200)
for name, pipe in pipelines.items():
    tprs, aucs = [], []
    for train_idx, test_idx in cv.split(X, y):
        pipe.fit(X_proc[train_idx], y[train_idx])
        prob_test = pipe.predict_proba(X_proc[test_idx])[:,1]
        fpr, tpr, _ = roc_curve(y[test_idx], prob_test)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))
    mean_tpr = np.mean(tprs, axis=0); mean_tpr[-1] = 1.0
    mean_auc, std_auc = np.mean(aucs), np.std(aucs)
    plt.plot(mean_fpr, mean_tpr, lw=2, label=f"{name} (AUC={mean_auc:.3f} ± {std_auc:.3f})")
plt.plot([0,1],[0,1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
plt.title("Combined ROC Curve (T2D)")
plt.legend(loc="lower right"); plt.grid(True); plt.tight_layout()
plt.savefig(r"D:/Downloads/T2D_Combined_ROC_all_models.png", dpi=1200)
plt.show()


# --------- Save metrics to CSV ---------
metrics_df = pd.DataFrame([{
    "model": name,
    "accuracy": v["accuracy"],
    "f1": v["f1"],
    "auc": v["auc"],
    "ci_low": v["ci"][0],
    "ci_high": v["ci"][1]
} for name, v in results.items()])
metrics_df.to_csv(r"D:/Downloads/T2D_model_metrics_oof.csv", index=False)
