"""
Day 7 — DDPG Implementation
Actor, Critic, Target networks, Replay Buffer, Ornstein-Uhlenbeck noise.
"""

import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


# ═══════════════════════════════════════════════════════════════════════════
# Networks
# ═══════════════════════════════════════════════════════════════════════════

class Actor(nn.Module):
    """Deterministic policy:  s → a ∈ [-1,1]^4"""
    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int]):
        super().__init__()
        layers = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, act_dim))
        layers.append(nn.Tanh())
        self.net = nn.Sequential(*layers)

    def forward(self, s):
        return self.net(s)


class Critic(nn.Module):
    """Action-value function:  (s, a) → Q"""
    def __init__(self, obs_dim: int, act_dim: int, hidden: list[int]):
        super().__init__()
        layers = []
        prev = obs_dim + act_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU()]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, s, a):
        return self.net(torch.cat([s, a], dim=-1))


# ═══════════════════════════════════════════════════════════════════════════
# Replay Buffer
# ═══════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity: int, obs_dim: int, act_dim: int):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.s  = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.a  = np.zeros((capacity, act_dim), dtype=np.float32)
        self.r  = np.zeros(capacity, dtype=np.float32)
        self.s2 = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.d  = np.zeros(capacity, dtype=np.float32)

    def store(self, s, a, r, s2, done):
        i = self.ptr % self.capacity
        self.s[i]  = s
        self.a[i]  = a
        self.r[i]  = r
        self.s2[i] = s2
        self.d[i]  = float(done)
        self.ptr += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        idx = rng.integers(0, self.size, size=batch_size)
        return (
            torch.FloatTensor(self.s[idx]),
            torch.FloatTensor(self.a[idx]),
            torch.FloatTensor(self.r[idx]).unsqueeze(-1),
            torch.FloatTensor(self.s2[idx]),
            torch.FloatTensor(self.d[idx]).unsqueeze(-1),
        )

    def __len__(self):
        return self.size


# ═══════════════════════════════════════════════════════════════════════════
# Prioritized Replay Buffer  (for ablation: DDPG + PER vs uniform)
# ═══════════════════════════════════════════════════════════════════════════

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, capacity, obs_dim, act_dim, alpha=0.6, beta=0.4):
        super().__init__(capacity, obs_dim, act_dim)
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0

    def store(self, s, a, r, s2, done):
        i = self.ptr % self.capacity
        self.priorities[i] = self.max_priority
        super().store(s, a, r, s2, done)

    def sample(self, batch_size, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        probs = self.priorities[:self.size] ** self.alpha
        probs /= probs.sum()

        idx = rng.choice(self.size, size=batch_size, p=probs, replace=False
                         if self.size >= batch_size else True)

        weights = (self.size * probs[idx]) ** (-self.beta)
        weights /= weights.max()

        batch = (
            torch.FloatTensor(self.s[idx]),
            torch.FloatTensor(self.a[idx]),
            torch.FloatTensor(self.r[idx]).unsqueeze(-1),
            torch.FloatTensor(self.s2[idx]),
            torch.FloatTensor(self.d[idx]).unsqueeze(-1),
        )
        return batch, torch.FloatTensor(weights).unsqueeze(-1), idx

    def update_priorities(self, idx, td_errors):
        self.priorities[idx] = np.abs(td_errors) + 1e-6
        self.max_priority = max(self.max_priority, self.priorities[idx].max())


# ═══════════════════════════════════════════════════════════════════════════
# Ornstein-Uhlenbeck Noise
# ═══════════════════════════════════════════════════════════════════════════

class OUNoise:
    def __init__(self, dim, theta=0.15, sigma=0.2, sigma_min=0.05,
                 decay=0.9995, seed=42):
        self.dim = dim
        self.theta = theta
        self.sigma = sigma
        self.sigma_min = sigma_min
        self.decay = decay
        self.rng = np.random.default_rng(seed)
        self.state = np.zeros(dim)

    def reset(self):
        self.state = np.zeros(self.dim)

    def sample(self):
        dx = self.theta * (-self.state) + self.sigma * self.rng.standard_normal(self.dim)
        self.state += dx
        return self.state.astype(np.float32)

    def anneal(self):
        self.sigma = max(self.sigma_min, self.sigma * self.decay)


# ═══════════════════════════════════════════════════════════════════════════
# DDPG Agent
# ═══════════════════════════════════════════════════════════════════════════

class DDPGAgent:
    def __init__(self, obs_dim, act_dim, cfg, use_per=False, seed=42):
        self.cfg = cfg
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.gamma = cfg["rl"]["gamma"]
        self.tau = cfg["rl"]["tau"]
        self.batch_size = cfg["rl"]["batch_size"]
        self.warmup = cfg["rl"]["warmup_steps"]
        self.use_per = use_per

        torch.manual_seed(seed)

        # Networks
        h = cfg["network"]["actor_hidden"]
        self.actor = Actor(obs_dim, act_dim, h)
        self.actor_target = copy.deepcopy(self.actor)
        self.critic = Critic(obs_dim, act_dim, cfg["network"]["critic_hidden"])
        self.critic_target = copy.deepcopy(self.critic)

        self.actor_opt = optim.Adam(self.actor.parameters(), lr=cfg["rl"]["actor_lr"])
        self.critic_opt = optim.Adam(self.critic.parameters(), lr=cfg["rl"]["critic_lr"])

        # Replay
        cap = cfg["rl"]["replay_capacity"]
        if use_per:
            self.buffer = PrioritizedReplayBuffer(cap, obs_dim, act_dim)
        else:
            self.buffer = ReplayBuffer(cap, obs_dim, act_dim)

        # Noise
        self.noise = OUNoise(
            act_dim,
            theta=cfg["rl"]["ou_theta"],
            sigma=cfg["rl"]["ou_sigma"],
            sigma_min=cfg["rl"]["ou_sigma_min"],
            decay=cfg["rl"]["ou_decay"],
            seed=seed,
        )

        self.total_steps = 0

    # ── action selection ─────────────────────────────────────────────────
    @torch.no_grad()
    def select_action(self, obs, add_noise=True):
        s = torch.FloatTensor(obs).unsqueeze(0)
        a = self.actor(s).squeeze(0).numpy()
        if add_noise:
            a = a + self.noise.sample()
        # Clip to valid action range
        a[:2] = np.clip(a[:2], -1.0, 1.0)
        a[2:] = np.clip(a[2:], 0.0, 1.0)
        return a

    # ── learning step ────────────────────────────────────────────────────
    def update(self):
        if len(self.buffer) < self.warmup:
            return None, None

        if self.use_per:
            batch, weights, idx = self.buffer.sample(self.batch_size)
        else:
            batch = self.buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size, 1)
            idx = None

        s, a, r, s2, d = batch

        # ── Critic update (Eq. 10-11) ────────────────────────────────────
        with torch.no_grad():
            a2 = self.actor_target(s2)
            target_q = r + self.gamma * (1 - d) * self.critic_target(s2, a2)
        current_q = self.critic(s, a)
        td_error = (current_q - target_q)
        critic_loss = (weights * td_error.pow(2)).mean()

        self.critic_opt.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_opt.step()

        # ── Actor update (Eq. 12) ────────────────────────────────────────
        actor_loss = -self.critic(s, self.actor(s)).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_opt.step()

        # ── Soft target update ───────────────────────────────────────────
        for p, tp in zip(self.actor.parameters(), self.actor_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)
        for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
            tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        # PER priority update
        if self.use_per and idx is not None:
            self.buffer.update_priorities(idx, td_error.detach().squeeze().numpy())

        return float(critic_loss), float(actor_loss)

    # ── persistence ──────────────────────────────────────────────────────
    def save(self, path):
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_target": self.actor_target.state_dict(),
            "critic_target": self.critic_target.state_dict(),
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location="cpu")
        self.actor.load_state_dict(ckpt["actor"])
        self.critic.load_state_dict(ckpt["critic"])
        self.actor_target.load_state_dict(ckpt["actor_target"])
        self.critic_target.load_state_dict(ckpt["critic_target"])


# ── Quick shape test ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    import yaml
    cfg = load_config()
    obs_dim = 502
    act_dim = 4
    agent = DDPGAgent(obs_dim, act_dim, cfg)

    s = np.random.randn(obs_dim).astype(np.float32)
    a = agent.select_action(s)
    print(f"obs dim: {obs_dim}, action: {a}, shape: {a.shape}")

    # Store some fake transitions and try update
    for _ in range(200):
        s2 = np.random.randn(obs_dim).astype(np.float32)
        agent.buffer.store(s, a, 1.0, s2, False)
        s = s2
        a = agent.select_action(s)

    c_loss, a_loss = agent.update()
    print(f"Critic loss: {c_loss:.4f}, Actor loss: {a_loss:.4f}")
    print("[ddpg] ✅ Day 7 complete.")
