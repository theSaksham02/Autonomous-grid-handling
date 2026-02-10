"""
Day 5 — Validation & Sanity Checks
Histograms, distribution comparisons, logistic-regression smoke test.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def validate_dataset(cfg):
    """Run all Day-5 checks."""
    tr = np.load(cfg["paths"]["train_data"])
    va = np.load(cfg["paths"]["val_data"])
    te = np.load(cfg["paths"]["test_data"])
    X_tr, y_tr = tr["X"], tr["y"]
    X_va, y_va = va["X"], va["y"]
    X_te, y_te = te["X"], te["y"]

    fig_dir = cfg["paths"]["figures_dir"]
    os.makedirs(fig_dir, exist_ok=True)

    # ── 1. Feature histograms ───────────────────────────────────────────
    n_features = X_tr.shape[1]
    sample_feats = np.linspace(0, n_features - 1, min(16, n_features), dtype=int)

    fig, axes = plt.subplots(4, 4, figsize=(16, 12))
    for ax, fi in zip(axes.flat, sample_feats):
        ax.hist(X_tr[:, fi], bins=40, alpha=0.7)
        ax.set_title(f"Feature {fi}", fontsize=8)
    plt.suptitle("Feature Histograms (training set)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "feature_histograms.png"), dpi=150)
    plt.close()
    print("[validate] Feature histograms saved.")

    # ── 2. Cascade vs non-cascade distributions ─────────────────────────
    cascade_mask = y_tr > 0
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, fi in zip(axes.flat, sample_feats[:8]):
        ax.hist(X_tr[~cascade_mask, fi], bins=30, alpha=0.5, label="σ=0")
        ax.hist(X_tr[cascade_mask, fi], bins=30, alpha=0.5, label="σ>0")
        ax.set_title(f"Feature {fi}", fontsize=8)
        ax.legend(fontsize=6)
    plt.suptitle("Cascade vs Non-cascade Feature Distributions")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "cascade_distributions.png"), dpi=150)
    plt.close()
    print("[validate] Distribution comparison saved.")

    # ── 3. Class balance across splits ──────────────────────────────────
    for name, y in [("train", y_tr), ("val", y_va), ("test", y_te)]:
        unique, counts = np.unique(y, return_counts=True)
        print(f"  {name}: " + ", ".join(f"σ={u}: {c}" for u, c in zip(unique, counts)))

    # ── 4. Check for NaN / Inf ──────────────────────────────────────────
    for name, X in [("train", X_tr), ("val", X_va), ("test", X_te)]:
        assert not np.isnan(X).any(), f"NaN in {name}!"
        assert not np.isinf(X).any(), f"Inf in {name}!"
    print("[validate] No NaN / Inf found.")

    # ── 5. Logistic Regression smoke test ───────────────────────────────
    # Binary classification: cascade (σ>0) vs. no-cascade (σ=0)
    y_tr_bin = (y_tr > 0).astype(int)
    y_va_bin = (y_va > 0).astype(int)
    y_te_bin = (y_te > 0).astype(int)

    lr = LogisticRegression(max_iter=500, solver="lbfgs")
    lr.fit(X_tr, y_tr_bin)

    pred_tr = lr.predict(X_tr)
    pred_va = lr.predict(X_va)
    pred_te = lr.predict(X_te)

    acc_tr = accuracy_score(y_tr_bin, pred_tr)
    acc_va = accuracy_score(y_va_bin, pred_va)
    acc_te = accuracy_score(y_te_bin, pred_te)

    print(f"\n[validate] Logistic Regression (binary cascade detection):")
    print(f"  Train acc: {acc_tr:.3f}")
    print(f"  Val acc:   {acc_va:.3f}")
    print(f"  Test acc:  {acc_te:.3f}")

    if acc_va < 0.55:
        print("  ⚠️  Val accuracy < 55% — features may not be discriminative!")
    else:
        print("  ✅  Features look discriminative enough to proceed.")

    print(f"\n  Test classification report:")
    print(classification_report(y_te_bin, pred_te, target_names=["No cascade", "Cascade"]))

    return dict(lr_train=acc_tr, lr_val=acc_va, lr_test=acc_te)


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config()
    stats = validate_dataset(cfg)
    print("[validate] ✅ Day 5 complete.")
