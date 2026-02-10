"""
Day 11 — Generate All Figures
Produces publication-quality matplotlib figures for the paper.
"""

import os, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
from sklearn.metrics import roc_curve, auc


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _savefig(fig, name, cfg, tight=True):
    d = cfg["paths"]["figures_dir"]
    os.makedirs(d, exist_ok=True)
    if tight:
        fig.tight_layout()
    path = os.path.join(d, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7 — Metrics Comparison Bar Chart
# ═══════════════════════════════════════════════════════════════════════════

def fig_metrics_comparison(cfg):
    with open(os.path.join(cfg["paths"]["tables_dir"], "all_results.json")) as f:
        results = json.load(f)

    methods = []
    metrics_names = ["accuracy", "precision", "recall", "f1", "auc"]
    data = {m: [] for m in metrics_names}
    errors = {m: [] for m in metrics_names}

    for name, m in results.items():
        methods.append(name)
        for metric in metrics_names:
            if f"{metric}_mean" in m:
                data[metric].append(m[f"{metric}_mean"])
                errors[metric].append(m[f"{metric}_std"])
            else:
                data[metric].append(m.get(metric, 0))
                errors[metric].append(0)

    x = np.arange(len(methods))
    width = 0.15
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = sns.color_palette("Set2", len(metrics_names))
    for i, metric in enumerate(metrics_names):
        offset = (i - len(metrics_names)/2 + 0.5) * width
        bars = ax.bar(x + offset, data[metric], width, yerr=errors[metric],
                      label=metric.capitalize(), color=colors[i],
                      capsize=3, edgecolor="black", linewidth=0.5)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Performance Comparison Across Methods", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=10)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    _savefig(fig, "fig7_metrics_comparison.png", cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8 — ROC Curves
# ═══════════════════════════════════════════════════════════════════════════

def fig_roc_curves(cfg):
    """Generate ROC curves for methods that produce probabilities."""
    te = np.load(cfg["paths"]["test_data"])
    y_te = (te["y"] > 0).astype(int)

    fig, ax = plt.subplots(figsize=(8, 8))

    # MLP probabilities (re-train quickly or load from results)
    from src.baselines import train_supervised_mlp
    _, mlp_probs, mlp_y, _, _ = train_supervised_mlp(cfg)
    fpr_mlp, tpr_mlp, _ = roc_curve(mlp_y, mlp_probs)
    auc_mlp = auc(fpr_mlp, tpr_mlp)
    ax.plot(fpr_mlp, tpr_mlp, label=f"Supervised MLP (AUC={auc_mlp:.3f})",
            linewidth=2)

    # For other methods, plot using confusion matrix points
    with open(os.path.join(cfg["paths"]["tables_dir"], "all_results.json")) as f:
        results = json.load(f)

    for name in ["Rule-based", "OPF"]:
        if name in results:
            m = results[name]
            fpr_pt = m.get("fpr", 0)
            rec_pt = m.get("recall", 0)
            ax.plot([0, fpr_pt, 1], [0, rec_pt, 1], "o--",
                    label=f"{name} (point)", markersize=8)

    # DDPG point
    if "DDPG (ours)" in results:
        m = results["DDPG (ours)"]
        fpr_pt = m.get("fpr_mean", m.get("fpr", 0))
        rec_pt = m.get("recall_mean", m.get("recall", 0))
        ax.plot(fpr_pt, rec_pt, "r*", markersize=15,
                label=f"DDPG (FPR={fpr_pt:.3f}, TPR={rec_pt:.3f})")

    ax.plot([0, 1], [0, 1], "k--", alpha=0.3, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)

    _savefig(fig, "fig8_roc_curves.png", cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 9 — Cascade Timeline (pick a representative case)
# ═══════════════════════════════════════════════════════════════════════════

def fig_cascade_timeline(cfg):
    """Line loadings over cascade propagation stages for a sample scenario."""
    cascade_raw = np.load(cfg["paths"]["cascade_results"], allow_pickle=True)
    severities = cascade_raw["severity"]
    n_iters = cascade_raw["n_iterations"]
    lines_tripped = cascade_raw["lines_tripped"]

    # Pick a scenario with moderate cascade (σ=2 or 3, >2 iterations)
    candidates = np.where((severities >= 2) & (n_iters > 2))[0]
    if len(candidates) == 0:
        candidates = np.where(severities >= 1)[0]
    if len(candidates) == 0:
        print("[fig] No cascade scenarios found for timeline figure.")
        return

    idx = candidates[0]
    tripped = lines_tripped[idx]
    tripped = tripped[tripped >= 0]  # remove padding

    fig, ax = plt.subplots(figsize=(10, 5))
    stages = np.arange(len(tripped))
    ax.bar(stages, tripped, color=sns.color_palette("Reds_d", len(tripped)),
           edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Cascade Stage", fontsize=12)
    ax.set_ylabel("Line Index Tripped", fontsize=12)
    ax.set_title(f"Cascade Propagation Timeline — Scenario {idx} "
                 f"(σ={severities[idx]}, {n_iters[idx]} iterations)", fontsize=13)
    ax.grid(axis="y", alpha=0.3)

    _savefig(fig, "fig9_cascade_timeline.png", cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 10 — Bus Vulnerability Heatmap
# ═══════════════════════════════════════════════════════════════════════════

def fig_bus_heatmap(cfg):
    """Count cascade participation per bus/line, plot as heatmap."""
    cascade_raw = np.load(cfg["paths"]["cascade_results"], allow_pickle=True)
    lines_tripped = cascade_raw["lines_tripped"]
    n_lines = cfg["grid"]["n_lines"]

    line_counts = np.zeros(n_lines, dtype=int)
    for row in lines_tripped:
        for li in row:
            if 0 <= li < n_lines:
                line_counts[li] += 1

    # Reshape into a grid for heatmap (approximate: 14 × ~14)
    side = int(np.ceil(np.sqrt(n_lines)))
    padded = np.zeros(side * side)
    padded[:n_lines] = line_counts
    grid = padded.reshape(side, side)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(grid, cmap="YlOrRd", ax=ax, cbar_kws={"label": "Trip count"})
    ax.set_title("Line Vulnerability Heatmap (trip frequency)", fontsize=14)
    ax.set_xlabel("Line index (row-major)", fontsize=11)
    ax.set_ylabel("")

    _savefig(fig, "fig10_bus_vulnerability_heatmap.png", cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 11 — Training Convergence Curve
# ═══════════════════════════════════════════════════════════════════════════

def fig_training_convergence(cfg):
    """Episode reward over training + validation accuracy."""
    log_dir = cfg["paths"]["logs_dir"]
    logs = []
    for fn in sorted(os.listdir(log_dir)):
        if fn.startswith("train_log_ddpg_per_seed") and fn.endswith(".json"):
            with open(os.path.join(log_dir, fn)) as f:
                logs.append(json.load(f))

    if not logs:
        print("[fig] No training logs found for convergence figure.")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for i, lg in enumerate(logs):
        rewards = lg["episode_rewards"]
        # Smooth with moving average
        window = 20
        smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
        ax1.plot(smoothed, alpha=0.7, label=f"Seed {i}")

    ax1.set_ylabel("Episode Reward (smoothed)", fontsize=11)
    ax1.set_title("Training Convergence", fontsize=14)
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # Validation accuracy
    eval_interval = cfg["rl"]["eval_interval"]
    for i, lg in enumerate(logs):
        val_acc = lg.get("val_accuracy", [])
        x_val = [(j+1) * eval_interval for j in range(len(val_acc))]
        ax2.plot(x_val, val_acc, "o-", alpha=0.7, label=f"Seed {i}")

    ax2.set_ylabel("Validation Accuracy", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3)

    # OU noise annealing
    for i, lg in enumerate(logs):
        sigma = lg.get("sigma_history", [])
        ax3.plot(sigma, alpha=0.7, label=f"Seed {i}")

    ax3.set_ylabel("OU σ (exploration noise)", fontsize=11)
    ax3.set_xlabel("Episode", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.grid(alpha=0.3)

    _savefig(fig, "fig11_training_convergence.png", cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 12 — Sensitivity Analysis (forecast degradation)
# ═══════════════════════════════════════════════════════════════════════════

def fig_sensitivity(cfg):
    """Placeholder: will be populated after sensitivity experiments."""
    rmse_levels = [0, 10, 20, 30]
    # These will be real values after running sensitivity experiments
    accuracies = [0.0, 0.0, 0.0, 0.0]  # placeholder

    # Check if sensitivity results exist
    sens_path = os.path.join(cfg["paths"]["tables_dir"], "sensitivity_results.json")
    if os.path.exists(sens_path):
        with open(sens_path) as f:
            sens = json.load(f)
        rmse_levels = sens.get("rmse_levels", rmse_levels)
        accuracies = sens.get("accuracies", accuracies)

    if max(accuracies) == 0:
        print("[fig] Sensitivity results not yet computed. Skipping fig12.")
        return

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(rmse_levels, accuracies, "bo-", linewidth=2, markersize=8)
    ax.set_xlabel("Weather Forecast RMSE (%)", fontsize=12)
    ax.set_ylabel("Cascade Detection Accuracy", fontsize=12)
    ax.set_title("Sensitivity to Weather Forecast Quality", fontsize=14)
    ax.grid(alpha=0.3)

    _savefig(fig, "fig12_sensitivity_forecast.png", cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Generate ALL figures
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_figures(cfg=None):
    if cfg is None:
        cfg = load_config()

    print("\n[figures] Generating all figures …\n")
    fig_metrics_comparison(cfg)
    fig_roc_curves(cfg)
    fig_cascade_timeline(cfg)
    fig_bus_heatmap(cfg)
    fig_training_convergence(cfg)
    fig_sensitivity(cfg)
    print("\n[figures] ✅ Day 11 complete.")


if __name__ == "__main__":
    generate_all_figures()
