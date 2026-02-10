"""
Day 9 — Baselines
Rule-based, Supervised MLP, OPF, and DDPG-no-PER / DDPG+PER comparison.
"""

import os, time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import pandapower as pp
import yaml
import copy
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, roc_curve,
                             confusion_matrix)
from tqdm import trange


def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


# ═══════════════════════════════════════════════════════════════════════════
# Baseline 1 — Rule-based
# ═══════════════════════════════════════════════════════════════════════════

class RuleBasedAgent:
    """If any line > 90 % or voltage outside [0.95,1.05], apply fixed actions."""

    def __init__(self, cfg):
        self.line_thr = cfg["baselines"]["rule_based"]["line_threshold"] * 100
        self.v_lo, self.v_hi = cfg["baselines"]["rule_based"]["voltage_band"]
        self.curtail = cfg["baselines"]["rule_based"]["curtail_pct"]
        self.dr = cfg["baselines"]["rule_based"]["demand_reduce_pct"]

    def predict(self, obs, net=None):
        """Return 4-dim action."""
        trigger = False
        if net is not None and net.converged:
            ll = net.res_line["loading_percent"].values
            vm = net.res_bus["vm_pu"].values
            if (ll > self.line_thr).any() or (vm < self.v_lo).any() or (vm > self.v_hi).any():
                trigger = True
        else:
            trigger = True

        if trigger:
            return np.array([0.5, 0.3, self.curtail, self.dr], dtype=np.float32)
        else:
            return np.zeros(4, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Baseline 2 — Supervised MLP (binary cascade classifier)
# ═══════════════════════════════════════════════════════════════════════════

class SupervisedMLP(nn.Module):
    def __init__(self, input_dim, hidden):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.2)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return torch.sigmoid(self.net(x))


def train_supervised_mlp(cfg):
    """Train MLP classifier on static dataset. Returns model + predictions."""
    tr = np.load(cfg["paths"]["train_data"])
    va = np.load(cfg["paths"]["val_data"])
    te = np.load(cfg["paths"]["test_data"])

    X_tr, y_tr = torch.FloatTensor(tr["X"]), torch.FloatTensor((tr["y"] > 0).astype(float))
    X_va, y_va = torch.FloatTensor(va["X"]), torch.FloatTensor((va["y"] > 0).astype(float))
    X_te, y_te = torch.FloatTensor(te["X"]), torch.FloatTensor((te["y"] > 0).astype(float))

    input_dim = X_tr.shape[1]
    hidden = cfg["baselines"]["supervised_mlp"]["hidden"]
    model = SupervisedMLP(input_dim, hidden)
    opt = optim.Adam(model.parameters(), lr=cfg["baselines"]["supervised_mlp"]["lr"])
    bce = nn.BCELoss()

    bs = cfg["baselines"]["supervised_mlp"]["batch_size"]
    epochs = cfg["baselines"]["supervised_mlp"]["epochs"]

    best_val_loss = float("inf")
    best_state = None

    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(X_tr))
        total_loss = 0.0
        for i in range(0, len(X_tr), bs):
            idx = perm[i:i+bs]
            pred = model(X_tr[idx]).squeeze()
            loss = bce(pred, y_tr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred = model(X_va).squeeze()
            val_loss = bce(val_pred, y_va).item()
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        test_probs = model(X_te).squeeze().numpy()
        train_probs = model(X_tr).squeeze().numpy()

    return model, test_probs, y_te.numpy(), train_probs, y_tr.numpy()


# ═══════════════════════════════════════════════════════════════════════════
# Baseline 3 — OPF
# ═══════════════════════════════════════════════════════════════════════════

def opf_baseline(base_net, weather, cfg, test_indices, hour=12):
    """Run OPF for each test scenario, then simulate N-k cascade on result.
    Uses per-scenario fixed RNG (idx+1000) for reproducibility.
    Returns predictions + timing."""
    from src.cascade_sim import _apply_scenario, _sample_contingency, \
        _trip_element, _check_violations, _shed_isolated_loads, \
        _run_pf_with_shedding

    predictions = []
    times = []

    for idx in test_indices:
        rng = np.random.default_rng(idx + 1000)
        net = copy.deepcopy(base_net)
        _apply_scenario(net, weather, idx, hour, cfg)

        t0 = time.time()
        try:
            # OPF needs cost functions; add if missing
            if len(net.poly_cost) == 0 and len(net.pwl_cost) == 0:
                for g in net.gen.index:
                    pp.create_poly_cost(net, g, "gen", cp1_eur_per_mw=1.0,
                                        cp0_eur=0)
                for eg in net.ext_grid.index:
                    pp.create_poly_cost(net, eg, "ext_grid", cp1_eur_per_mw=1.0,
                                        cp0_eur=0)
            pp.runopp(net, verbose=False)
            converged = getattr(net, "OPF_converged", True)
        except Exception:
            converged = False
            # Fallback: try regular power flow
            try:
                pp.runpp(net, algorithm="nr", max_iteration=30, tolerance_mva=1e-4)
                converged = net.converged
            except Exception:
                converged = False

        # After OPF, simulate N-k cascade on the resulting grid state
        if converged:
            total_load = max(net.load["p_mw"].sum(), 1.0)
            contingencies = _sample_contingency(net, rng)
            if contingencies:
                for etype, eidx in contingencies:
                    _trip_element(net, etype, eidx)
                # Mini cascade propagation
                cascade_sev = 0
                for _ in range(5):
                    _shed_isolated_loads(net)
                    conv = _run_pf_with_shedding(net, max_shed_steps=2)
                    if not conv:
                        cascade_sev = 3
                        break
                    viol_l, viol_g = _check_violations(net, cfg, rng)
                    if not viol_l and not viol_g:
                        break
                    for li in viol_l:
                        _trip_element(net, "line", li)
                    for gi in viol_g:
                        _trip_element(net, "gen", gi)
                if cascade_sev == 0:
                    remaining = 0.0
                    try:
                        remaining = net.res_load["p_mw"].sum()
                    except Exception:
                        pass
                    shed = max(0.0, 1.0 - remaining / total_load)
                    eps = cfg["cascade"]["load_shed_eps"] / total_load
                    cascade_sev = 1 if shed > eps else 0
            else:
                cascade_sev = 0
            predictions.append(1 if cascade_sev > 0 else 0)
        else:
            predictions.append(1)  # assume cascade if OPF fails

        elapsed = time.time() - t0
        times.append(elapsed)

    return np.array(predictions), np.array(times)


# ═══════════════════════════════════════════════════════════════════════════
# Metrics helper
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(y_true, y_pred, y_prob=None, method_name=""):
    """Compute and return a dict of classification metrics."""
    y_true_bin = (y_true > 0).astype(int) if y_true.max() > 1 else y_true.astype(int)
    y_pred_bin = (y_pred > 0).astype(int) if y_pred.max() > 1 else y_pred.astype(int)

    cm = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    fpr_val = fp / (fp + tn) if (fp + tn) > 0 else 0.0

    auc = 0.5
    if y_prob is not None and len(np.unique(y_true_bin)) > 1:
        try:
            auc = roc_auc_score(y_true_bin, y_prob)
        except Exception:
            auc = 0.5

    metrics = {
        "method": method_name,
        "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "auc": float(auc),
        "fpr": float(fpr_val),
    }
    return metrics


# ── CLI ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cfg = load_config()
    print("[baselines] Training supervised MLP …")
    model, probs, y_te, _, _ = train_supervised_mlp(cfg)
    preds = (probs > 0.5).astype(int)
    m = compute_metrics(y_te, preds, probs, "Supervised MLP")
    print(f"  MLP — Acc: {m['accuracy']:.3f}  F1: {m['f1']:.3f}  AUC: {m['auc']:.3f}")
    print("[baselines] ✅ Day 9 – MLP done.")
