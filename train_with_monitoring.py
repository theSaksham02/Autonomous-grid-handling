"""
DDPG Training with Real-time Monitoring and Early Stopping
"""
import torch
import numpy as np
from ddpg_Agent import DDPGAgent
from grid_env import GridEnv
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os
import time
from datetime import datetime, timedelta

def estimate_time(episodes_completed, episodes_total, elapsed_time):
    """Estimate remaining training time"""
    if episodes_completed == 0:
        return "N/A"
    avg_time_per_episode = elapsed_time / episodes_completed
    remaining_episodes = episodes_total - episodes_completed
    remaining_time = remaining_episodes * avg_time_per_episode
    
    hours = int(remaining_time // 3600)
    minutes = int((remaining_time % 3600) // 60)
    return f"{hours}h {minutes}m"

def train_ddpg_monitored(episodes=100, save_interval=20, early_stopping_patience=30):
    """Train DDPG agent with monitoring and early stopping"""
    
    # Initialize
    agent = DDPGAgent()
    env = GridEnv()
    
    # Tracking
    episode_rewards = []
    episode_lengths = []
    cascade_probs = []
    actor_losses = []
    critic_losses = []
    
    # Early stopping
    best_reward = -np.inf
    patience_counter = 0
    
    print("\n" + "="*80)
    print("🚀 DDPG TRAINING WITH MONITORING")
    print("="*80)
    print(f"Target episodes: {episodes}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print(f"Checkpoint interval: {save_interval}")
    print("="*80 + "\n")
    
    start_time = time.time()
    
    for episode in tqdm(range(episodes), desc="Training Progress"):
        episode_start = time.time()
        
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        episode_actor_loss = 0
        episode_critic_loss = 0
        train_steps = 0
        
        agent.noise.reset()
        
        for step in range(96):  # 24 hours × 4 (15-min intervals)
            # Select action
            action = agent.select_action(state, add_noise=True)
            
            # Environment step
            next_state, reward, terminated, truncated, info = env.step(action)
            
            # Store transition
            done = terminated or truncated
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            
            # Train
            if len(agent.replay_buffer) >= agent.batch_size:
                critic_loss, actor_loss = agent.train_step()
                episode_actor_loss += actor_loss
                episode_critic_loss += critic_loss
                train_steps += 1
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        # Average losses for episode
        if train_steps > 0:
            episode_actor_loss /= train_steps
            episode_critic_loss /= train_steps
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        cascade_probs.append(info.get('cascading_prob', 0))
        actor_losses.append(episode_actor_loss)
        critic_losses.append(episode_critic_loss)
        
        # Early stopping check
        recent_avg_reward = np.mean(episode_rewards[-10:]) if len(episode_rewards) >= 10 else episode_reward
        if recent_avg_reward > best_reward:
            best_reward = recent_avg_reward
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Logging
        if (episode + 1) % 10 == 0:
            elapsed = time.time() - start_time
            avg_reward = np.mean(episode_rewards[-10:])
            avg_cascade = np.mean(cascade_probs[-10:])
            avg_actor_loss = np.mean(actor_losses[-10:])
            avg_critic_loss = np.mean(critic_losses[-10:])
            episode_time = time.time() - episode_start
            est_remaining = estimate_time(episode + 1, episodes, elapsed)
            
            print(f"\n📊 Episode {episode+1:4d}/{episodes}")
            print(f"   ├─ Reward (10-ep avg): {avg_reward:8.3f}")
            print(f"   ├─ Cascade Prob: {avg_cascade:6.3f}")
            print(f"   ├─ Actor Loss: {avg_actor_loss:8.4f}")
            print(f"   ├─ Critic Loss: {avg_critic_loss:8.4f}")
            print(f"   ├─ Elapsed: {elapsed/60:.1f}min | Est. Remaining: {est_remaining}")
            print(f"   └─ Patience: {patience_counter}/{early_stopping_patience}")
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            os.makedirs('models/trained_weights', exist_ok=True)
            agent.save(f'models/trained_weights/ddpg_ep{episode+1}.pth')
            print(f"   💾 Checkpoint saved: ddpg_ep{episode+1}.pth")
        
        # Early stopping
        if patience_counter >= early_stopping_patience and episode >= 20:
            print(f"\n⚠️  Early stopping triggered (patience exceeded)")
            break
    
    # Save final model
    agent.save('models/trained_weights/ddpg_final.pth')
    
    # Save training history
    os.makedirs('results', exist_ok=True)
    with open('results/training_history.pkl', 'wb') as f:
        pickle.dump({
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'cascade_probs': cascade_probs,
            'actor_losses': actor_losses,
            'critic_losses': critic_losses
        }, f)
    
    # Generate plots
    plot_training_history(episode_rewards, cascade_probs, actor_losses, critic_losses)
    
    elapsed_total = time.time() - start_time
    print("\n" + "="*80)
    print(f"✅ Training completed in {elapsed_total/60:.1f} minutes ({elapsed_total/3600:.2f} hours)")
    print(f"   Total episodes: {len(episode_rewards)}")
    print(f"   Best 10-episode avg reward: {max([np.mean(episode_rewards[max(0,i-9):i+1]) for i in range(len(episode_rewards))]):.3f}")
    print(f"   Final reward: {episode_rewards[-1]:.3f}")
    print("="*80 + "\n")
    
    return agent, episode_rewards

def plot_training_history(rewards, cascades, actor_losses, critic_losses):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Rewards
    axes[0, 0].plot(rewards, alpha=0.7, label='Episode Reward')
    axes[0, 0].plot([np.mean(rewards[max(0,i-9):i+1]) for i in range(len(rewards))], 
                    linewidth=2, label='10-ep Moving Average', color='red')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Cascade probability
    axes[0, 1].plot(cascades, alpha=0.7, label='Cascade Prob')
    axes[0, 1].plot([np.mean(cascades[max(0,i-9):i+1]) for i in range(len(cascades))], 
                    linewidth=2, label='10-ep Moving Average', color='red')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Probability')
    axes[0, 1].set_title('Cascading Failure Probability')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Actor loss
    axes[1, 0].plot(actor_losses, alpha=0.7)
    axes[1, 0].plot([np.mean(actor_losses[max(0,i-9):i+1]) for i in range(len(actor_losses))], 
                    linewidth=2, label='10-ep MA', color='red')
    axes[1, 0].set_xlabel('Episode')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Actor Network Loss')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Critic loss
    axes[1, 1].plot(critic_losses, alpha=0.7)
    axes[1, 1].plot([np.mean(critic_losses[max(0,i-9):i+1]) for i in range(len(critic_losses))], 
                    linewidth=2, label='10-ep MA', color='red')
    axes[1, 1].set_xlabel('Episode')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Critic Network Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/training_history.png', dpi=150)
    print(f"📈 Training plots saved to results/training_history.png")
    plt.close()

if __name__ == "__main__":
    # Run with 100 episodes for testing (adjust as needed)
    # For full training, use episodes=5000
    agent, rewards = train_ddpg_monitored(
        episodes=100,  # Start with 100, increase for full training
        save_interval=20,
        early_stopping_patience=30
    )
