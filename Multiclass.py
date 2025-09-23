import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, auc, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cross_decomposition import PLSRegression
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer
from scipy import stats

# --- Pareto scaling ---
def pareto_scale(X):
    X = np.asarray(X)
    mu = np.nanmean(X, axis=0)
    std = np.nanstd(X, axis=0)
    denom = np.sqrt(std)
    denom[denom == 0] = 1.0
    return (X - mu) / denom

pareto_transformer = FunctionTransformer(lambda X: pareto_scale(X), validate=False)

# --- Load pooled dataset ---
def load_dataset(path):
    df = pd.read_csv(path, index_col=0)
    labels = df.iloc[0, 1:].values
    features = df.iloc[1:, 1:].copy()
    features.index = df.iloc[1:, 0]
    features = features.apply(pd.to_numeric, errors='coerce')
    features = features.groupby(features.index).mean()
    return features, labels

paths = {
    "IBD": r"D:/Downloads/IBD101_Y.csv",
    "T2D": r"D:/Downloads/T2D101_Y.csv",
    "CRC": r"D:/Downloads/CRC101_Y.csv"
}
datasets = {k: load_dataset(v) for k, v in paths.items()}

all_features = sorted(set().union(*[df.index for df, _ in datasets.values()]))
X_list, y_list = [], []
for df, y in datasets.values():
    df_aligned = df.reindex(all_features, fill_value=np.nan)
    df_final = df_aligned.T
    X_list.append(df_final)
    y_list.append(pd.Series(y, index=df_final.index))

X_uni = pd.concat(X_list, axis=0).reset_index(drop=True)
y_uni = pd.concat(y_list, axis=0).reset_index(drop=True)

# Encode labels
le = LabelEncoder()
le.fit(["Control","T2D","CRC","IBD"])
y = le.transform(y_uni)
y = le.fit_transform(y_uni)

# Preproc pipeline
base_preproc = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("log1p", FunctionTransformer(lambda Z: np.log1p(Z), validate=False)),
    ("pareto", pareto_transformer),
    ("scaler", StandardScaler())
])

# --- Models ---
class PLS_DA:
    def __init__(self, n_components=5):
        self.pls = PLSRegression(n_components=n_components)
        self.clf = LogisticRegression(max_iter=1000)

    def fit(self, X, y):
        Y_bin = pd.get_dummies(y)
        X_scores = self.pls.fit_transform(X, Y_bin)[0]
        self.clf.fit(X_scores, y)
        return self

    def predict(self, X):
        X_scores = self.pls.transform(X)
        return self.clf.predict(X_scores)

    def predict_proba(self, X):
        X_scores = self.pls.transform(X)
        return self.clf.predict_proba(X_scores)

models = {
    "Logistic Regression": Pipeline([("pre", base_preproc), ("clf", LogisticRegression(max_iter=2000, multi_class="ovr"))]),
    "SVM (Linear)": Pipeline([("pre", base_preproc), ("clf", SVC(kernel="linear", probability=True))]),
    "SVM (RBF)": Pipeline([("pre", base_preproc), ("clf", SVC(kernel="rbf", probability=True))]),
    "Random Forest": Pipeline([("pre", base_preproc), ("clf", RandomForestClassifier(n_estimators=200, random_state=42))]),
    "XGBoost": Pipeline([("pre", base_preproc), ("clf", XGBClassifier(use_label_encoder=False, eval_metric="mlogloss", random_state=42))]),
    "kNN": Pipeline([("pre", base_preproc), ("clf", KNeighborsClassifier(n_neighbors=5))]),
    "MLP": Pipeline([("pre", base_preproc), ("clf", MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42))]),
    "PLS-DA": Pipeline([("pre", base_preproc), ("clf", PLS_DA(n_components=5))])
}

# --- CV evaluation ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
results = {name: {"accuracy": [], "f1": [], "auc": [], "ci": []} for name in models}

plt.figure(figsize=(8, 6))

for name, model in models.items():
    print(f"\n--- {name} ---")
    mean_fpr = np.linspace(0, 1, 100)
    tprs, aucs, all_probs, all_labels = [], [], [], []

    for train_idx, test_idx in cv.split(X_uni, y):
        X_train, X_test = X_uni.iloc[train_idx].values, X_uni.iloc[test_idx].values
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)
        else:
            raise ValueError(f"{name} has no predict_proba")

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        auc_score = roc_auc_score(pd.get_dummies(y_test), y_prob, average="macro", multi_class="ovr")

        results[name]["accuracy"].append(acc)
        results[name]["f1"].append(f1)
        results[name]["auc"].append(auc_score)

        all_probs.extend(y_prob)
        all_labels.extend(y_test)

        # One-vs-rest ROC (macro)
        y_test_bin = pd.get_dummies(y_test).values
        fpr, tpr, _ = roc_curve(y_test_bin.ravel(), y_prob.ravel())
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        interp_tpr[-1] = 1.0
        tprs.append(interp_tpr)
        aucs.append(auc(fpr, tpr))

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    plt.plot(mean_fpr, mean_tpr, label=f"{name} (AUC={mean_auc:.2f} ± {std_auc:.2f})")

    # Bootstrap CI
    all_probs = np.array(all_probs).reshape(-1, len(le.classes_))
    all_labels = np.array(all_labels)
    aucs_boot = []
    rng = np.random.RandomState(42)
    for i in range(1000):
        idx = rng.randint(0, len(all_labels), len(all_labels))
        if len(np.unique(all_labels[idx])) < len(le.classes_):
            continue
        try:
            aucs_boot.append(roc_auc_score(pd.get_dummies(all_labels[idx]), all_probs[idx], 
                                           average="macro", multi_class="ovr"))
        except:
            continue
    ci_low, ci_high = np.percentile(aucs_boot, [2.5, 97.5])
    results[name]["ci"] = (ci_low, ci_high)

    print("Accuracy:", np.mean(results[name]["accuracy"]))
    print("F1-score:", np.mean(results[name]["f1"]))
    print(f"ROC-AUC: {np.mean(results[name]['auc']):.3f} (95% CI: {ci_low:.3f}–{ci_high:.3f})")

plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig("D:/Downloads/Pooled_Multiclass_ROC.png", dpi=1200)
plt.show()

# --- Save results ---
metrics_df = pd.DataFrame({
    "Model": list(results.keys()),
    "Accuracy": [np.mean(results[m]["accuracy"]) for m in results],
    "F1": [np.mean(results[m]["f1"]) for m in results],
    "ROC-AUC": [np.mean(results[m]["auc"]) for m in results],
    "CI_low": [results[m]["ci"][0] for m in results],
    "CI_high": [results[m]["ci"][1] for m in results]
})
metrics_df.to_csv("D:/Downloads/Pooled_Multiclass_Model_Metrics.csv", index=False)
