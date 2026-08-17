"""Build extra paper figures/tables from the n=150 evaluation."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import binomtest, wilcoxon

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures"
TAB = ROOT / "results" / "tables"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "figure.dpi": 160,
    "savefig.dpi": 220,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

COLORS = {
    "none": "#94a3b8",
    "rule": "#2563eb",
    "opf": "#64748b",
    "ddpg0": "#0f766e",
    "ddpg1": "#14b8a6",
    "noper": "#a16207",
}


def paired(na_p, na_s, ot_p, ot_s):
    b = int(np.sum((na_p == 1) & (ot_p == 0)))
    c = int(np.sum((na_p == 0) & (ot_p == 1)))
    n_disc = b + c
    p_mc = float(binomtest(b, n=n_disc, p=0.5).pvalue) if n_disc else 1.0
    try:
        w = wilcoxon(na_s - ot_s, zero_method="wilcox", alternative="greater")
        p_w, stat = float(w.pvalue), float(w.statistic)
    except ValueError:
        p_w, stat = 1.0, 0.0
    return dict(
        prevented=b, induced=c, unchanged_cascade=int(np.sum((na_p == 1) & (ot_p == 1))),
        both_safe=int(np.sum((na_p == 0) & (ot_p == 0))),
        cascade_rate=float(ot_p.mean()), mean_shed=float(ot_s.mean()),
        shed_drop=float(na_s.mean() - ot_s.mean()),
        mcnemar_p=p_mc, wilcoxon_p=p_w, wilcoxon_stat=stat,
    )


def smooth(y, k=7):
    y = np.asarray(y, dtype=float)
    if len(y) < k:
        return y
    k += k % 2 == 0
    pad = k // 2
    yp = np.pad(y, (pad, pad), mode="edge")
    ker = np.ones(k) / k
    return np.convolve(yp, ker, mode="valid")


def main():
    d = np.load(TAB / "per_scenario_n150.npz")
    na_p, na_s, na_v = d["no_agent_preds"], d["no_agent_shed"], d["no_agent_sev"]
    rule_p, rule_s = d["rule_preds"], d["rule_shed"]
    opf_p = d["opf_preds"]
    d0_p, d0_s = d["ddpg_per_seed0_preds"], d["ddpg_per_seed0_shed"]
    d1_p, d1_s = d["ddpg_per_seed1_preds"], d["ddpg_per_seed1_shed"]
    np_p, np_s = d["ddpg_noper_preds"], d["ddpg_noper_shed"]

    stats = {
        "Do nothing": dict(prevented=0, induced=0, cascade_rate=float(na_p.mean()),
                           mean_shed=float(na_s.mean()), shed_drop=0.0,
                           mcnemar_p=None, wilcoxon_p=None),
        "If-then rule": paired(na_p, na_s, rule_p, rule_s),
        "DDPG seed 0": paired(na_p, na_s, d0_p, d0_s),
        "DDPG seed 1": paired(na_p, na_s, d1_p, d1_s),
        "DDPG no PER": paired(na_p, na_s, np_p, np_s),
    }
    stats["OPF"] = dict(
        prevented=int(np.sum((na_p == 1) & (opf_p == 0))),
        induced=int(np.sum((na_p == 0) & (opf_p == 1))),
        cascade_rate=float(opf_p.mean()),
        mean_shed=None, shed_drop=None, mcnemar_p=None, wilcoxon_p=None,
    )
    with open(TAB / "extra_stats_n150.json", "w") as f:
        json.dump(stats, f, indent=2)

    # 1) pipeline schematic
    fig, ax = plt.subplots(figsize=(7.2, 2.15))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 2.2)
    ax.axis("off")
    boxes = [
        (0.2, "1. Weather\n1000 days"),
        (2.15, "2. IEEE 118\nstressed grid"),
        (4.1, "3. One action\n4 small knobs"),
        (6.05, "4. Same accident\nseed i+1000"),
        (8.0, "5. Score\ncascade / shed"),
    ]
    for x, t in boxes:
        ax.add_patch(plt.Rectangle((x, 0.45), 1.75, 1.35, fc="#ecfeff", ec="#0f766e", lw=1.4, zorder=2))
        ax.text(x + 0.875, 1.12, t, ha="center", va="center", fontsize=8, zorder=3)
    for x in (1.95, 3.9, 5.85, 7.8):
        ax.annotate("", xy=(x + 0.18, 1.12), xytext=(x - 0.02, 1.12),
                    arrowprops=dict(arrowstyle="->", color="#334155", lw=1.4))
    ax.set_title("How one test day is scored (same accident for every method)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_pipeline.png", bbox_inches="tight")
    plt.close()

    # 2) severity of the no-agent baseline
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    labels = ["None", "Minor", "Moderate", "Severe"]
    counts = [int((na_v == i).sum()) for i in range(4)]
    ax.bar(labels, counts, color=["#86efac", "#fde047", "#fb923c", "#f87171"])
    for i, c in enumerate(counts):
        ax.text(i, c + 1.2, str(c), ha="center", fontsize=8)
    ax.set_ylabel("Number of test days")
    ax.set_title("How bad is a cascade if we do nothing? (n=150)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_severity_none.png", bbox_inches="tight")
    plt.close()

    # 3) load-shed boxplot
    fig, ax = plt.subplots(figsize=(6.4, 2.8))
    data = [na_s, rule_s, d0_s, d1_s, np_s]
    names = ["Do nothing", "If-then", "DDPG s0", "DDPG s1", "DDPG no PER"]
    bp = ax.boxplot(data, tick_labels=names, showfliers=False, patch_artist=True,
                    medianprops=dict(color="#0f172a", lw=1.4))
    cols = [COLORS["none"], COLORS["rule"], COLORS["ddpg0"], COLORS["ddpg1"], COLORS["noper"]]
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.55)
    ax.set_ylabel("Dropped demand (fraction)")
    ax.set_title("Load shed on each of 150 test days")
    ax.tick_params(axis="x", rotation=12)
    fig.tight_layout()
    fig.savefig(FIG / "fig_shed_box.png", bbox_inches="tight")
    plt.close()

    # 4) prevented vs induced
    fig, ax = plt.subplots(figsize=(5.6, 2.7))
    methods = ["If-then", "OPF", "DDPG s0", "DDPG s1", "DDPG no PER"]
    prev = [stats["If-then rule"]["prevented"], stats["OPF"]["prevented"],
            stats["DDPG seed 0"]["prevented"], stats["DDPG seed 1"]["prevented"],
            stats["DDPG no PER"]["prevented"]]
    indu = [stats["If-then rule"]["induced"], stats["OPF"]["induced"],
            stats["DDPG seed 0"]["induced"], stats["DDPG seed 1"]["induced"],
            stats["DDPG no PER"]["induced"]]
    x = np.arange(len(methods))
    w = 0.36
    ax.bar(x - w / 2, prev, w, label="Saved a cascade day", color="#0f766e")
    ax.bar(x + w / 2, indu, w, label="Created a new cascade day", color="#b91c1c")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Number of days (out of 150)")
    ax.set_title("Helped vs hurt, versus doing nothing")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_prevented_induced.png", bbox_inches="tight")
    plt.close()

    # 5) paired shed scatter for best DDPG seed
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    ax.scatter(na_s, d1_s, s=14, c=COLORS["ddpg1"], alpha=0.7, edgecolors="none")
    ax.plot([0, 1], [0, 1], "--", c="#94a3b8", lw=1)
    ax.set_xlabel("Do-nothing load shed")
    ax.set_ylabel("DDPG seed-1 load shed")
    ax.set_title("Below the line = agent shed less")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(FIG / "fig_shed_scatter.png", bbox_inches="tight")
    plt.close()

    # 6) training curves
    fig, ax = plt.subplots(figsize=(6.4, 2.7))
    for name, path, c in [
        ("DDPG seed 0", ROOT / "logs/train_log_ddpg_per_seed0.json", COLORS["ddpg0"]),
        ("DDPG seed 1", ROOT / "logs/train_log_ddpg_per_seed1.json", COLORS["ddpg1"]),
        ("DDPG no PER", ROOT / "logs/train_log_ddpg_noper_seed0.json", COLORS["noper"]),
    ]:
        log = json.loads(path.read_text())
        y = np.array(log["episode_rewards"], dtype=float)
        ax.plot(y, color=c, alpha=0.25, lw=0.8)
        ax.plot(smooth(y), color=c, lw=1.6, label=name)
    ax.set_xlabel("Training episode")
    ax.set_ylabel("Episode reward (proxy)")
    ax.set_title("Training: the agent learns a healthier-looking grid")
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "fig_training.png", bbox_inches="tight")
    plt.close()

    # 7) cascade rate + latency already exists; refresh a 3-panel summary
    fig, axes = plt.subplots(1, 3, figsize=(8.6, 2.75))
    names = ["Do nothing", "If-then", "OPF", "DDPG\n(mean)"]
    cr = [na_p.mean(), rule_p.mean(), opf_p.mean(), 0.5 * (d0_p.mean() + d1_p.mean())]
    cols = [COLORS["none"], COLORS["rule"], COLORS["opf"], COLORS["ddpg0"]]
    axes[0].bar(names, [x * 100 for x in cr], color=cols)
    axes[0].set_ylabel("Cascade rate (%)")
    axes[0].set_title("Cascade frequency")
    axes[0].set_ylim(0, 45)
    for i, v in enumerate(cr):
        axes[0].text(i, v * 100 + 0.8, f"{100*v:.1f}", ha="center", fontsize=7)

    shed_n = ["Do nothing", "If-then", "DDPG\nmean", "DDPG\nno PER"]
    shed_v = [na_s.mean(), rule_s.mean(), 0.5 * (d0_s.mean() + d1_s.mean()), np_s.mean()]
    axes[1].bar(shed_n, shed_v, color=[COLORS["none"], COLORS["rule"], COLORS["ddpg0"], COLORS["noper"]])
    axes[1].set_ylabel("Mean load shed")
    axes[1].set_title("How much demand is dropped")
    axes[1].set_ylim(0, 0.2)
    for i, v in enumerate(shed_v):
        axes[1].text(i, v + 0.004, f"{v:.3f}", ha="center", fontsize=7)

    lat_n = ["If-then", "DDPG", "OPF"]
    lat_v = [293.5, 299.4, 567.8]
    axes[2].bar(lat_n, lat_v, color=[COLORS["rule"], COLORS["ddpg0"], COLORS["opf"]])
    axes[2].set_ylabel("ms per decision")
    axes[2].set_title("Speed")
    for i, v in enumerate(lat_v):
        axes[2].text(i, v + 8, f"{v:.0f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_summary_three.png", bbox_inches="tight")
    plt.close()

    print("wrote figures in", FIG)
    for k, v in stats.items():
        print(k, {kk: v[kk] for kk in v if kk in ("prevented", "induced", "cascade_rate", "mean_shed", "mcnemar_p")})


if __name__ == "__main__":
    main()
