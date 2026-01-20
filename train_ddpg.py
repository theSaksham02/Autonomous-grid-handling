"""
DDPG Training Script for Cascading Failure Mitigation --> Google collab
"""
import torch
import numpy as np
from ddpg_Agent import DDPGAgent
from grid_env import GridEnv
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import os

def train_ddpg(episodes=5000, save_interval=500):
    """Train DDPG agent"""
    
    # Initialize
    agent = DDPGAgent()
    env = GridEnv()
    
    # Tracking
    episode_rewards = []
    episode_lengths = []
    cascade_probs = []
    
    print(f"\n🚀 Starting DDPG Training for {episodes} episodes...")
    
    for episode in tqdm(range(episodes)):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
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
            
            episode_reward += reward
            episode_length += 1
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        cascade_probs.append(info.get('cascading_prob', 0))
        
        # Logging
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_cascade = np.mean(cascade_probs[-100:])
            print(f"\nEpisode {episode+1}/{episodes}")
            print(f"  Avg Reward: {avg_reward:.3f}")
            print(f"  Avg Cascade Prob: {avg_cascade:.3f}")
        
        # Save checkpoint
        if (episode + 1) % save_interval == 0:
            os.makedirs('models/trained_weights', exist_ok=True)
            agent.save(f'models/trained_weights/ddpg_ep{episode+1}.pth')
    
    # Save final model
    agent.save('models/trained_weights/ddpg_final.pth')
    
    # Save training history
    with open('results/training_history.pkl', 'wb') as f:
        pickle.dump({
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'cascade_probs': cascade_probs
        }, f)
    
    print("\n✅ Training completed!")
    return agent, episode_rewards

if __name__ == "__main__":
    os.makedirs('results', exist_ok=True)
    agent, rewards = train_ddpg(episodes=100)  # Start with 100 for testing
