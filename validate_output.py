import argparse, pathlib, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score, make_scorer
from sklearn.linear_model import LogisticRegression

# ------------------------------------------------------------------
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        sys.exit(f"[ERR] {path} is empty.")
    return df

def label_checks(df: pd.DataFrame, name: str) -> pd.Series:
    labels = df.iloc[:, -1]
    assert labels.notna().all(), f"{name}: NaNs in label column."
    u = sorted(labels.unique())
    assert set(u).issubset({0, 1}), f"{name}: labels must be 0/1, got {u}."
    return labels.astype(int)

# ------------------------------------------------------------------
def plot_class_balance(orig_lbls, aug_lbls):
    counts = {
        "Original\nmajority": (orig_lbls == 0).sum(),
        "Original\nminority": (orig_lbls == 1).sum(),
        "Augmented\nmajority": (aug_lbls == 0).sum(),
        "Augmented\nminority": (aug_lbls == 1).sum(),
    }
    plt.figure()
    plt.bar(counts.keys(), counts.values())
    plt.title("Class balance before and after augmentation")
    plt.ylabel("Samples")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------
def plot_pca(original_df, augmented_df):
    X_orig = original_df.iloc[:, :-1].values
    X_aug  = augmented_df.iloc[:, :-1].values
    X      = np.vstack([X_orig, X_aug])
    y_vis  = np.hstack([np.zeros(len(X_orig)), np.ones(len(X_aug))])

    pca = PCA(n_components=2, random_state=0)
    XY  = pca.fit_transform(X)

    plt.figure()
    plt.scatter(*XY[y_vis == 0].T, s=8, alpha=0.4, label="Original")
    plt.scatter(*XY[y_vis == 1].T, s=8, alpha=0.4, label="Augmented")
    plt.title("PCA projection – original vs augmented")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.legend(); plt.tight_layout(); plt.show()

# ------------------------------------------------------------------
def model_score(df, clf):
    X = df.iloc[:, :-1].values
    y = df.iloc[:,  -1].values
    scorer = make_scorer(f1_score)   # swap to neg_mean_absolute_error for regression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scores = cross_val_score(clf, X, y, scoring=scorer, cv=5)
    return scores.mean()

def plot_model_performance(orig_df, aug_df):
    clf = LogisticRegression(max_iter=200, solver="liblinear")
    orig_f1 = model_score(orig_df, clf)
    aug_f1  = model_score(aug_df , clf)

    plt.figure()
    plt.bar(["Original data", "Augmented data"], [orig_f1, aug_f1])
    plt.ylabel("Mean 5‑fold F1")
    plt.title("Does augmentation improve model performance?")
    plt.ylim(0, 1)
    for x,y in zip([0,1],[orig_f1, aug_f1]):
        plt.text(x, y+0.02, f"{y:.3f}", ha="center")
    plt.tight_layout(); plt.show()

# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original", required=True, help="CSV before augmentation")
    ap.add_argument("--aug",      required=True, help="CSV after augmentation")
    args = ap.parse_args()

    orig_df = load_df(args.original)
    aug_df  = load_df(args.aug)

    orig_lbls = label_checks(orig_df, "original")
    aug_lbls  = label_checks(aug_df , "augmented")

    print(f"[OK] Original balance: 0={ (orig_lbls==0).sum() }  1={ (orig_lbls==1).sum() }")
    print(f"[OK] Augmented balance: 0={ (aug_lbls==0).sum() }  1={ (aug_lbls==1).sum() }")
    assert abs((aug_lbls==0).sum() - (aug_lbls==1).sum()) <= 1, "Augmented set is imbalanced."

    # 1️⃣ balance chart
    plot_class_balance(orig_lbls, aug_lbls)

    # 2️⃣ PCA scatter
    plot_pca(orig_df, aug_df)

    # 3️⃣ model‑performance bar chart
    plot_model_performance(orig_df, aug_df)

if __name__ == "__main__":
    main()