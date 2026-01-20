"""
Deep Deterministic Policy Gradient (DDPG) Agent for Cascading Failure Mitigation
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random
import yaml


class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise"""

    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state


class ReplayBuffer:
    """Experience replay buffer"""

    def __init__(self, capacity=1000000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones)
        )

    def __len__(self):
        return len(self.buffer)


class Actor(nn.Module):
    """Actor network: maps states to actions"""

    def __init__(self, state_dim, action_dim, hidden_dims=[512, 256, 128], dropout=0.2):
        super(Actor, self).__init__()

        layers = []
        prev_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        # Remove dropout from last layer
        layers = layers[:-1]

        self.hidden = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, action_dim)
        self.tanh = nn.Tanh()

    def forward(self, state):
        x = self.hidden(state)
        x = self.output(x)
        return self.tanh(x)


class Critic(nn.Module):
    """Critic network: estimates Q(s,a)"""

    def __init__(self, state_dim, action_dim, hidden_dims=[512, 256, 128], dropout=0.2):
        super(Critic, self).__init__()

        # State pathway
        self.state_net = nn.Sequential(
            nn.Linear(state_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.ReLU()
        )

        # Action pathway
        self.action_net = nn.Sequential(
            nn.Linear(action_dim, 64),
            nn.ReLU()
        )

        # Combined pathway
        combined_dim = hidden_dims[0] + 64
        layers = []
        prev_dim = combined_dim

        for hidden_dim in hidden_dims[1:]:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        layers = layers[:-1]  # Remove last dropout

        self.combined = nn.Sequential(*layers)
        self.output = nn.Linear(prev_dim, 1)

    def forward(self, state, action):
        s = self.state_net(state)
        a = self.action_net(action)
        x = torch.cat([s, a], dim=1)
        x = self.combined(x)
        return self.output(x)


class DDPGAgent:
    """DDPG Agent for cascading failure prediction and mitigation"""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        cfg = self.config['ddpg']
        self.state_dim = cfg['state_dim']
        self.action_dim = cfg['action_dim']

        # Device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"ðŸ–¥ï¸  Using device: {self.device}")

        # Networks
        self.actor = Actor(
            self.state_dim, 
            self.action_dim,
            cfg['actor']['hidden_layers'],
            cfg['actor']['dropout']
        ).to(self.device)

        self.critic = Critic(
            self.state_dim,
            self.action_dim,
            cfg['critic']['hidden_layers'],
            cfg['critic']['dropout']
        ).to(self.device)

        # Target networks
        self.actor_target = Actor(
            self.state_dim,
            self.action_dim,
            cfg['actor']['hidden_layers'],
            cfg['actor']['dropout']
        ).to(self.device)

        self.critic_target = Critic(
            self.state_dim,
            self.action_dim,
            cfg['critic']['hidden_layers'],
            cfg['critic']['dropout']
        ).to(self.device)

        # Initialize target networks
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_optimizer = optim.Adam(
            self.actor.parameters(),
            lr=cfg['actor']['learning_rate']
        )

        self.critic_optimizer = optim.Adam(
            self.critic.parameters(),
            lr=cfg['critic']['learning_rate']
        )

        # Replay buffer
        self.replay_buffer = ReplayBuffer(cfg['training']['buffer_size'])

        # Exploration noise
        noise_cfg = cfg['training']['exploration_noise']
        self.noise = OUNoise(
            self.action_dim,
            mu=noise_cfg['mu'],
            theta=noise_cfg['theta'],
            sigma=noise_cfg['sigma']
        )

        # Hyperparameters
        self.gamma = cfg['training']['gamma']
        self.tau = cfg['training']['tau']
        self.batch_size = cfg['training']['batch_size']

        print(f"âœ… DDPG Agent initialized")
        print(f"   State dim: {self.state_dim}")
        print(f"   Action dim: {self.action_dim}")
        print(f"   Actor params: {sum(p.numel() for p in self.actor.parameters()):,}")
        print(f"   Critic params: {sum(p.numel() for p in self.critic.parameters()):,}")

    def select_action(self, state, add_noise=True):
        """Select action given state"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        self.actor.eval()
        with torch.no_grad():
            action = self.actor(state).cpu().numpy()[0]
        self.actor.train()

        if add_noise:
            action += self.noise.sample()
            action = np.clip(action, -1, 1)

        return action

    def train_step(self):
        """Single training step"""
        if len(self.replay_buffer) < self.batch_size:
            return None, None

        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.unsqueeze(1).to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.unsqueeze(1).to(self.device)

        # Update critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = rewards + (1 - dones) * self.gamma * self.critic_target(next_states, next_actions)

        current_q = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Update actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor, self.actor_target)
        self._soft_update(self.critic, self.critic_target)

        return critic_loss.item(), actor_loss.item()

    def _soft_update(self, source, target):
        """Soft update target network"""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def save(self, filepath):
        """Save model weights"""
        torch.save({
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'actor_target_state_dict': self.actor_target.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict()
        }, filepath)
        print(f"ðŸ’¾ Model saved to {filepath}")

    def load(self, filepath):
        """Load model weights"""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
        print(f"ðŸ“‚ Model loaded from {filepath}")


if __name__ == "__main__":
    # Test DDPG agent
    agent = DDPGAgent()

    # Test forward pass
    dummy_state = np.random.randn(247)
    action = agent.select_action(dummy_state)

    print(f"\n=== Test Action ===")
    print(f"Action shape: {action.shape}")
    print(f"Action values: {action}")
    print(f"Action range: [{action.min():.3f}, {action.max():.3f}]")