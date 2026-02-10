"""
Stage 7 — Ablation Study, Sensitivity Analysis, Figure Generation, LaTeX Tables

Uses cascade-based evaluation (single-step paradigm) consistent with Stage 6.
All agents are loaded from trained weights — no re-training needed for ablation
of *evaluation-time* variants; only the feature-ablation variants need training.
"""
import warnings
warnings.filterwarnings("ignore")
import os, logging
logging.disable(logging.WARNING)

# Monkey-patch pandapower for numba=False
import pandapower as pp
_orig_runpp = pp.runpp
def _quiet_runpp(net, **kwargs):
    if 'numba' not in kwargs:
        kwargs['numba'] = False
    return _orig_runpp(net, **kwargs)
pp.runpp = _quiet_runpp

import json, copy, time, sys
import numpy as np
import torch
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.grid_setup import load_base_case
from src.grid_env import GridCascadeEnv
from src.ddpg import DDPGAgent
from src.train import train_ddpg
from src.baselines import RuleBasedAgent
from run_stage6 import evaluate_with_cascade, evaluate_no_agent, _get_splits

# ── Figures (inline to avoid import issues) ───────────────────────────────
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)


def _savefig(fig, name, cfg):
    d = cfg["paths"]["figures_dir"]
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, name)
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  [fig] Saved → {path}")


# ═══════════════════════════════════════════════════════════════════════════
# Figure 1 — Training Convergence (reward + max line loading)
# ═══════════════════════════════════════════════════════════════════════════
def fig_training_curves(cfg):
    """Episode reward and exploration noise over training for each seed."""
    log_dir = cfg["paths"]["logs_dir"]
    logs = []
    for fn in sorted(os.listdir(log_dir)):
        if fn.startswith("train_log_ddpg_per_seed") and fn.endswith(".json"):
            with open(os.path.join(log_dir, fn)) as f:
                logs.append(json.load(f))
    if not logs:
        print("  [fig] No training logs found. Skipping.")
        return

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    colors = sns.color_palette("tab10", len(logs))
    window = 10

    for i, lg in enumerate(logs):
        rewards = np.array(lg["episode_rewards"])
        smoothed = np.convolve(rewards, np.ones(window)/window, mode="valid")
        axes[0].plot(range(window-1, len(rewards)), smoothed,
                     color=colors[i], alpha=0.8, label=f"Seed {i}")
        axes[0].fill_between(range(window-1, len(rewards)),
                             smoothed - 2, smoothed + 2,
                             color=colors[i], alpha=0.1)

    axes[0].set_ylabel("Episode Reward (smoothed)")
    axes[0].set_title("DDPG Training Convergence")
    axes[0].legend()

    for i, lg in enumerate(logs):
        sigma = lg.get("sigma_history", [])
        axes[1].plot(sigma, color=colors[i], alpha=0.8, label=f"Seed {i}")

    axes[1].set_ylabel("OU Noise σ")
    axes[1].set_xlabel("Episode")
    axes[1].legend()

    _savefig(fig, "fig1_training_convergence.png", cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 2 — Cascade Rate Comparison Bar Chart
# ═══════════════════════════════════════════════════════════════════════════
def fig_cascade_comparison(cfg):
    """Bar chart: cascade rate and mean load shed for each method."""
    with open(os.path.join(cfg["paths"]["tables_dir"], "all_results.json")) as f:
        results = json.load(f)

    methods = []
    cascade_rates = []
    load_sheds = []
    for name, m in results.items():
        if name == "Supervised MLP":
            continue  # predictor, not preventor
        methods.append(name.replace(" (baseline)", "\n(baseline)").replace(" (ours)", "\n(ours)"))
        cr = m.get("cascade_rate_mean", m.get("cascade_rate", 0))
        ls = m.get("mean_load_shed_mean", m.get("mean_load_shed", 0))
        cascade_rates.append(cr)
        load_sheds.append(ls)

    x = np.arange(len(methods))
    fig, ax1 = plt.subplots(figsize=(10, 6))

    bar1 = ax1.bar(x - 0.18, cascade_rates, 0.35, label="Cascade Rate",
                   color=sns.color_palette("Set2")[0], edgecolor="black", linewidth=0.5)
    ax1.set_ylabel("Cascade Rate", fontsize=12)
    ax1.set_ylim(0, max(cascade_rates) * 1.3 + 0.05)

    ax2 = ax1.twinx()
    bar2 = ax2.bar(x + 0.18, load_sheds, 0.35, label="Mean Load Shed",
                   color=sns.color_palette("Set2")[1], edgecolor="black", linewidth=0.5)
    ax2.set_ylabel("Mean Load Shed Fraction", fontsize=12)
    ax2.set_ylim(0, max(load_sheds) * 1.3 + 0.02)

    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, fontsize=10)
    ax1.set_title("Cascade Prevention Performance Comparison", fontsize=14)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=10)

    _savefig(fig, "fig2_cascade_comparison.png", cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 3 — Severity Distribution
# ═══════════════════════════════════════════════════════════════════════════
def fig_severity_distribution(cfg):
    """Stacked bar: severity σ distribution for each method."""
    with open(os.path.join(cfg["paths"]["tables_dir"], "all_results.json")) as f:
        results = json.load(f)

    methods = []
    sev_data = {0: [], 1: [], 2: [], 3: []}
    for name, m in results.items():
        if "severity_dist" not in m:
            continue
        methods.append(name)
        sd = m["severity_dist"]
        for s in range(4):
            sev_data[s].append(sd.get(str(s), 0))

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    labels = ["σ₀ (none)", "σ₁ (minor)", "σ₂ (moderate)", "σ₃ (severe)"]

    bottom = np.zeros(len(methods))
    for s in range(4):
        vals = np.array(sev_data[s], dtype=float)
        ax.bar(x, vals, bottom=bottom, color=colors[s], label=labels[s],
               edgecolor="black", linewidth=0.3)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Number of Scenarios (n=25)", fontsize=12)
    ax.set_title("Cascade Severity Distribution by Method", fontsize=14)
    ax.legend(fontsize=10)

    _savefig(fig, "fig3_severity_distribution.png", cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 4 — Cascade Event Histogram from Ground Truth
# ═══════════════════════════════════════════════════════════════════════════
def fig_cascade_histogram(cfg):
    """Histogram of ground-truth severity across all 1000 scenarios."""
    cascade_raw = np.load(cfg["paths"]["cascade_results"], allow_pickle=True)
    sev = cascade_raw["severity"]
    shed = cascade_raw["load_shed_frac"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    counts = [int((sev == s).sum()) for s in range(4)]
    colors = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c"]
    ax1.bar(range(4), counts, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_xticks(range(4))
    ax1.set_xticklabels(["σ₀\n(none)", "σ₁\n(minor)", "σ₂\n(moderate)", "σ₃\n(severe)"])
    ax1.set_ylabel("Count")
    ax1.set_title(f"Severity Distribution (N={len(sev)})")
    for i, c in enumerate(counts):
        ax1.text(i, c + 5, str(c), ha="center", fontsize=11)

    ax2.hist(shed[shed > 0], bins=30, color="#3498db", edgecolor="black",
             linewidth=0.5, alpha=0.8)
    ax2.set_xlabel("Load Shed Fraction")
    ax2.set_ylabel("Count")
    ax2.set_title("Load Shed Distribution (cascade scenarios only)")
    ax2.axvline(0.20, color="orange", linestyle="--", label="σ₁ threshold")
    ax2.axvline(0.50, color="red", linestyle="--", label="σ₂ threshold")
    ax2.legend()

    _savefig(fig, "fig4_cascade_histogram.png", cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 5 — Line Vulnerability Heatmap
# ═══════════════════════════════════════════════════════════════════════════
def fig_line_vulnerability(cfg):
    """Top-N most frequently tripped lines."""
    cascade_raw = np.load(cfg["paths"]["cascade_results"], allow_pickle=True)
    lines_tripped = cascade_raw["lines_tripped"]
    n_lines = cfg["grid"]["n_lines"]

    line_counts = np.zeros(n_lines, dtype=int)
    for row in lines_tripped:
        for li in row:
            if 0 <= li < n_lines:
                line_counts[li] += 1

    top_n = 25
    top_idx = np.argsort(line_counts)[::-1][:top_n]
    top_counts = line_counts[top_idx]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.barh(range(top_n), top_counts[::-1], color=sns.color_palette("YlOrRd_r", top_n),
            edgecolor="black", linewidth=0.3)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([f"Line {i}" for i in top_idx[::-1]], fontsize=9)
    ax.set_xlabel("Trip Frequency (across 1000 scenarios)")
    ax.set_title(f"Top {top_n} Most Vulnerable Lines", fontsize=14)

    _savefig(fig, "fig5_line_vulnerability.png", cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 6 — Ablation Study Bar Chart
# ═══════════════════════════════════════════════════════════════════════════
def fig_ablation(cfg):
    """Bar chart comparing ablation variants."""
    abl_path = os.path.join(cfg["paths"]["tables_dir"], "ablation_results.json")
    if not os.path.exists(abl_path):
        print("  [fig] No ablation results. Skipping.")
        return

    with open(abl_path) as f:
        abl = json.load(f)

    variants = list(abl.keys())
    cascade_rates = [abl[v].get("cascade_rate", 0) for v in variants]
    load_sheds = [abl[v].get("mean_load_shed", 0) for v in variants]

    x = np.arange(len(variants))
    fig, ax = plt.subplots(figsize=(9, 5))
    bar1 = ax.bar(x - 0.18, cascade_rates, 0.35, label="Cascade Rate",
                  color="#3498db", edgecolor="black", linewidth=0.5)
    bar2 = ax.bar(x + 0.18, load_sheds, 0.35, label="Mean Load Shed",
                  color="#e74c3c", edgecolor="black", linewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(variants, rotation=15, ha="right", fontsize=10)
    ax.set_ylabel("Rate / Fraction")
    ax.set_title("Ablation Study — Feature Contribution", fontsize=14)
    ax.legend()

    _savefig(fig, "fig6_ablation.png", cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 7 — Sensitivity to Forecast Noise
# ═══════════════════════════════════════════════════════════════════════════
def fig_sensitivity(cfg):
    """Line plot: cascade rate vs. forecast noise."""
    sens_path = os.path.join(cfg["paths"]["tables_dir"], "sensitivity_results.json")
    if not os.path.exists(sens_path):
        print("  [fig] No sensitivity results. Skipping.")
        return

    with open(sens_path) as f:
        sens = json.load(f)

    rmse_levels = sens["rmse_levels"]
    cascade_rates = sens["cascade_rates"]
    load_sheds = sens["load_sheds"]

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(rmse_levels, cascade_rates, "bo-", linewidth=2, markersize=8,
             label="Cascade Rate")
    ax1.set_xlabel("Weather Forecast RMSE (%)", fontsize=12)
    ax1.set_ylabel("Cascade Rate", color="blue", fontsize=12)
    ax1.tick_params(axis='y', labelcolor="blue")

    ax2 = ax1.twinx()
    ax2.plot(rmse_levels, load_sheds, "rs--", linewidth=2, markersize=8,
             label="Mean Load Shed")
    ax2.set_ylabel("Mean Load Shed", color="red", fontsize=12)
    ax2.tick_params(axis='y', labelcolor="red")

    ax1.set_title("Sensitivity to Weather Forecast Degradation", fontsize=14)
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    _savefig(fig, "fig7_sensitivity_forecast.png", cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Figure 8 — Inference Latency Comparison
# ═══════════════════════════════════════════════════════════════════════════
def fig_latency(cfg):
    """Bar chart of inference latency per method."""
    with open(os.path.join(cfg["paths"]["tables_dir"], "all_results.json")) as f:
        results = json.load(f)

    methods = []
    latencies = []
    for name, m in results.items():
        if name in ("Supervised MLP", "No Agent (baseline)"):
            continue
        methods.append(name)
        ms = m.get("mean_time_ms_mean", m.get("mean_time_ms", 0))
        latencies.append(ms)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette("Set2", len(methods))
    bars = ax.bar(methods, latencies, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Inference Latency (ms)", fontsize=12)
    ax.set_title("Inference Speed Comparison", fontsize=14)
    for bar, val in zip(bars, latencies):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                f"{val:.0f}ms", ha="center", fontsize=10)
    ax.set_xticklabels(methods, rotation=15, ha="right")

    _savefig(fig, "fig8_latency.png", cfg)


# ═══════════════════════════════════════════════════════════════════════════
# Ablation Study (cascade-based)
# ═══════════════════════════════════════════════════════════════════════════
def run_ablation(cfg, net, weather, test_idx, na_preds, na_shed, n_test):
    """
    Ablation variants — all use cascade-based single-step evaluation:
    1. Full model with PER (from saved results)
    2. Without weather features (zero weather in scenarios)
    3. Without PER (from saved model)
    """
    print("\n[ablation] Running ablation study …")
    results = {}
    obs_dim = 489
    act_dim = 4

    # ── Full model (load from all_results.json) ─────────────────────────
    res_path = os.path.join(cfg["paths"]["tables_dir"], "all_results.json")
    with open(res_path) as f:
        all_res = json.load(f)

    # Use seed=0 results for consistency
    agent_full = DDPGAgent(obs_dim, act_dim, cfg)
    ckpt = torch.load("models/trained_weights/best_ddpg_ddpg_per_seed0.pt",
                       map_location="cpu")
    agent_full.actor.load_state_dict(ckpt["actor"])
    agent_full.critic.load_state_dict(ckpt["critic"])
    agent_full.actor.eval()

    preds_full, sev_full, shed_full, times_full = evaluate_with_cascade(
        lambda obs, net, a=agent_full: a.select_action(obs, add_noise=False),
        net, weather, cfg, test_idx, n_test=n_test
    )
    results["Full model (PER)"] = {
        "cascade_rate": float(preds_full.mean()),
        "mean_load_shed": float(shed_full.mean()),
        "shed_reduction": float(na_shed.mean() - shed_full.mean()),
        "cascades_prevented": int(np.sum((na_preds == 1) & (preds_full == 0))),
    }
    print(f"  Full model: cascade_rate={preds_full.mean():.3f}, shed={shed_full.mean():.3f}")

    # ── Without weather features ────────────────────────────────────────
    print("  Training DDPG without weather features …")
    weather_zeroed = copy.deepcopy(weather)
    for key in ["wind_speed", "ghi", "temperature", "wind_power", "solar_power"]:
        if key in weather_zeroed:
            weather_zeroed[key] = np.zeros_like(weather_zeroed[key])

    cfg_abl = copy.deepcopy(cfg)
    cfg_abl["rl"]["n_episodes"] = 100
    cfg_abl["rl"]["episode_length"] = 8
    cfg_abl["rl"]["warmup_steps"] = 50
    cfg_abl["rl"]["eval_interval"] = 20

    agent_nw, _ = train_ddpg(cfg_abl, net, weather_zeroed, seed=0,
                              use_per=True, tag="ddpg_no_weather",
                              n_episodes=100)
    preds_nw, sev_nw, shed_nw, _ = evaluate_with_cascade(
        lambda obs, net, a=agent_nw: a.select_action(obs, add_noise=False),
        net, weather_zeroed, cfg, test_idx, n_test=n_test
    )
    results["Without weather"] = {
        "cascade_rate": float(preds_nw.mean()),
        "mean_load_shed": float(shed_nw.mean()),
        "shed_reduction": float(na_shed.mean() - shed_nw.mean()),
        "cascades_prevented": int(np.sum((na_preds == 1) & (preds_nw == 0))),
    }
    print(f"  Without weather: cascade_rate={preds_nw.mean():.3f}, shed={shed_nw.mean():.3f}")

    # ── Without PER ─────────────────────────────────────────────────────
    agent_noper = DDPGAgent(obs_dim, act_dim, cfg)
    ckpt = torch.load("models/trained_weights/best_ddpg_ddpg_noper_seed0.pt",
                       map_location="cpu")
    agent_noper.actor.load_state_dict(ckpt["actor"])
    agent_noper.critic.load_state_dict(ckpt["critic"])
    agent_noper.actor.eval()

    preds_np, sev_np, shed_np, _ = evaluate_with_cascade(
        lambda obs, net, a=agent_noper: a.select_action(obs, add_noise=False),
        net, weather, cfg, test_idx, n_test=n_test
    )
    results["Without PER"] = {
        "cascade_rate": float(preds_np.mean()),
        "mean_load_shed": float(shed_np.mean()),
        "shed_reduction": float(na_shed.mean() - shed_np.mean()),
        "cascades_prevented": int(np.sum((na_preds == 1) & (preds_np == 0))),
    }
    print(f"  Without PER: cascade_rate={preds_np.mean():.3f}, shed={shed_np.mean():.3f}")

    out_path = os.path.join(cfg["paths"]["tables_dir"], "ablation_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  [ablation] Saved → {out_path}")
    return results


# ═══════════════════════════════════════════════════════════════════════════
# Sensitivity Analysis (cascade-based)
# ═══════════════════════════════════════════════════════════════════════════
def run_sensitivity(cfg, net, weather, test_idx, na_preds, na_shed, n_test):
    """
    Forecast degradation: add noise to weather, re-evaluate trained DDPG.
    """
    print("\n[sensitivity] Running forecast degradation analysis …")
    obs_dim = 489
    act_dim = 4

    agent = DDPGAgent(obs_dim, act_dim, cfg)
    ckpt = torch.load("models/trained_weights/best_ddpg_ddpg_per_seed0.pt",
                       map_location="cpu")
    agent.actor.load_state_dict(ckpt["actor"])
    agent.critic.load_state_dict(ckpt["critic"])
    agent.actor.eval()

    rmse_levels = [0, 10, 20, 30]
    cascade_rates = []
    load_sheds = []

    for rmse in rmse_levels:
        weather_noisy = copy.deepcopy(weather)
        rng = np.random.default_rng(42 + rmse)
        noise_scale = rmse / 100.0

        for key in ["wind_speed", "ghi", "temperature"]:
            if key in weather_noisy:
                std = max(np.std(weather_noisy[key]), 1e-6)
                noise = rng.normal(0, noise_scale * std,
                                   size=weather_noisy[key].shape)
                weather_noisy[key] = (weather_noisy[key] + noise).astype(np.float32)

        preds, sev, shed, _ = evaluate_with_cascade(
            lambda obs, net, a=agent: a.select_action(obs, add_noise=False),
            net, weather_noisy, cfg, test_idx, n_test=n_test
        )
        cascade_rates.append(float(preds.mean()))
        load_sheds.append(float(shed.mean()))
        print(f"  RMSE {rmse}%: cascade_rate={preds.mean():.3f}, shed={shed.mean():.3f}")

    sens = {
        "rmse_levels": rmse_levels,
        "cascade_rates": cascade_rates,
        "load_sheds": load_sheds,
    }
    out_path = os.path.join(cfg["paths"]["tables_dir"], "sensitivity_results.json")
    with open(out_path, "w") as f:
        json.dump(sens, f, indent=2)
    print(f"  [sensitivity] Saved → {out_path}")
    return sens


# ═══════════════════════════════════════════════════════════════════════════
# LaTeX Table Generation
# ═══════════════════════════════════════════════════════════════════════════
def generate_latex_tables(cfg):
    """Generate publication-ready LaTeX tables from results JSONs."""
    tables_dir = cfg["paths"]["tables_dir"]

    # ── Table I: Dataset Summary ────────────────────────────────────────
    cascade_raw = np.load(cfg["paths"]["cascade_results"], allow_pickle=True)
    severities = cascade_raw["severity"]
    train_idx, val_idx, test_idx = _get_splits(cfg)
    y_tr, y_va, y_te = severities[train_idx], severities[val_idx], severities[test_idx]

    table1 = r"""\begin{table}[t]
\centering
\caption{Dataset Split and Cascade Distribution}
\label{tab:dataset}
\begin{tabular}{lccc}
\toprule
 & Train & Validation & Test \\
\midrule
Total scenarios & """ + f"{len(y_tr)} & {len(y_va)} & {len(y_te)}" + r""" \\
Cascade ($\sigma > 0$) & """ + f"{int((y_tr>0).sum())} & {int((y_va>0).sum())} & {int((y_te>0).sum())}" + r""" \\
Non-cascade ($\sigma = 0$) & """ + f"{int((y_tr==0).sum())} & {int((y_va==0).sum())} & {int((y_te==0).sum())}" + r""" \\
\bottomrule
\end{tabular}
\end{table}"""

    # ── Table II: Main Results ──────────────────────────────────────────
    with open(os.path.join(tables_dir, "all_results.json")) as f:
        results = json.load(f)

    rows = []
    order = ["No Agent (baseline)", "Rule-based", "OPF", "DDPG (ours)", "DDPG (no PER)"]
    for name in order:
        m = results.get(name, {})
        cr = m.get("cascade_rate_mean", m.get("cascade_rate", "---"))
        ls = m.get("mean_load_shed_mean", m.get("mean_load_shed", "---"))
        sr = m.get("shed_reduction_mean", m.get("shed_reduction", "---"))
        prev = m.get("cascades_prevented_mean", m.get("cascades_prevented", "---"))
        ms = m.get("mean_time_ms_mean", m.get("mean_time_ms", "---"))

        cr_s = f"{cr:.3f}" if isinstance(cr, float) else str(cr)
        ls_s = f"{ls:.3f}" if isinstance(ls, float) else str(ls)
        sr_s = f"{sr:+.3f}" if isinstance(sr, float) else str(sr)
        prev_s = f"{prev:.0f}" if isinstance(prev, (int, float)) else str(prev)
        ms_s = f"{ms:.1f}" if isinstance(ms, (int, float)) else str(ms)

        rows.append(f"{name} & {cr_s} & {ls_s} & {sr_s} & {prev_s} & {ms_s} " + r"\\")

    table2 = r"""\begin{table}[t]
\centering
\caption{Cascade Prevention Performance on Test Set (25 scenarios)}
\label{tab:results}
\begin{tabular}{lccccc}
\toprule
Method & Cascade Rate & Load Shed & Shed $\downarrow$ & Prevented & Latency (ms) \\
\midrule
""" + "\n".join(rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""

    # ── Table III: MLP Classification ───────────────────────────────────
    mlp = results.get("Supervised MLP", {})
    table3 = r"""\begin{table}[t]
\centering
\caption{Supervised MLP Cascade Prediction (classification-only)}
\label{tab:mlp}
\begin{tabular}{lcccccc}
\toprule
Method & Accuracy & Precision & Recall & F1 & AUC & FPR \\
\midrule
""" + f"Supervised MLP & {mlp.get('accuracy',0):.3f} & {mlp.get('precision',0):.3f} & {mlp.get('recall',0):.3f} & {mlp.get('f1',0):.3f} & {mlp.get('auc',0):.3f} & {mlp.get('fpr',0):.3f}" + r""" \\
\bottomrule
\end{tabular}
\end{table}"""

    # ── Table IV: Ablation ──────────────────────────────────────────────
    abl_path = os.path.join(tables_dir, "ablation_results.json")
    if os.path.exists(abl_path):
        with open(abl_path) as f:
            abl = json.load(f)
        abl_rows = []
        for name, m in abl.items():
            abl_rows.append(
                f"{name} & {m.get('cascade_rate',0):.3f} & "
                f"{m.get('mean_load_shed',0):.3f} & "
                f"{m.get('shed_reduction',0):+.3f} & "
                f"{m.get('cascades_prevented',0)} " + r"\\")
        table4 = r"""\begin{table}[t]
\centering
\caption{Ablation Study}
\label{tab:ablation}
\begin{tabular}{lcccc}
\toprule
Variant & Cascade Rate & Load Shed & Shed $\downarrow$ & Prevented \\
\midrule
""" + "\n".join(abl_rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""
    else:
        table4 = "% Ablation: run ablation first\n"

    # ── Table V: Sensitivity ────────────────────────────────────────────
    sens_path = os.path.join(tables_dir, "sensitivity_results.json")
    if os.path.exists(sens_path):
        with open(sens_path) as f:
            sens = json.load(f)
        sens_rows = []
        for rmse, cr, ls in zip(sens["rmse_levels"], sens["cascade_rates"],
                                sens["load_sheds"]):
            sens_rows.append(f"{rmse}\\% & {cr:.3f} & {ls:.3f} " + r"\\")
        table5 = r"""\begin{table}[t]
\centering
\caption{Sensitivity to Weather Forecast Degradation}
\label{tab:sensitivity}
\begin{tabular}{lcc}
\toprule
Forecast RMSE & Cascade Rate & Mean Load Shed \\
\midrule
""" + "\n".join(sens_rows) + r"""
\bottomrule
\end{tabular}
\end{table}"""
    else:
        table5 = "% Sensitivity: run sensitivity first\n"

    all_tables = "\n\n".join([table1, table2, table3, table4, table5])
    out_path = os.path.join(tables_dir, "latex_tables.tex")
    with open(out_path, "w") as f:
        f.write(all_tables)
    print(f"  [tables] LaTeX tables saved → {out_path}")
    return all_tables


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)

    n_test = 25

    print("=" * 60)
    print("  STAGE 7: Analysis, Figures & Tables")
    print("=" * 60)

    net = load_base_case(cfg["paths"]["base_case"])
    weather = dict(np.load(cfg["paths"]["weather_scenarios"], allow_pickle=True))
    _, _, test_idx = _get_splits(cfg)

    # No-agent baseline (needed for shed_reduction and prevented calculations)
    print("\n[baseline] Computing no-agent baseline …")
    na_preds, na_sev, na_shed = evaluate_no_agent(
        net, weather, cfg, test_idx, n_test=n_test)
    print(f"  No-agent: cascade_rate={na_preds.mean():.3f}, shed={na_shed.mean():.3f}")

    # ── Ablation ────────────────────────────────────────────────────────
    abl = run_ablation(cfg, net, weather, test_idx, na_preds, na_shed, n_test)

    # ── Sensitivity ─────────────────────────────────────────────────────
    sens = run_sensitivity(cfg, net, weather, test_idx, na_preds, na_shed, n_test)

    # ── LaTeX tables ────────────────────────────────────────────────────
    print("\n[tables] Generating LaTeX tables …")
    generate_latex_tables(cfg)

    # ── Figures ─────────────────────────────────────────────────────────
    print("\n[figures] Generating all figures …")
    fig_training_curves(cfg)
    fig_cascade_comparison(cfg)
    fig_severity_distribution(cfg)
    fig_cascade_histogram(cfg)
    fig_line_vulnerability(cfg)
    fig_ablation(cfg)
    fig_sensitivity(cfg)
    fig_latency(cfg)

    print("\n" + "=" * 60)
    print("  ✅ Stage 7 complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
