"""
Quick test script to verify project setup (runs 5 episodes for testing)
"""
import torch
import numpy as np
import os
import sys

print("=" * 60)
print("AUTONOMOUS GRID HEALING - QUICK SETUP TEST")
print("=" * 60)

# Test 1: Import all modules
print("\n[1/5] Testing imports...")
try:
    from ddpg_Agent import DDPGAgent
    from grid_env import GridEnv
    from grid_simulator import GridSimulator
    from weather_injector import WeatherInjector
    from baseline_methods import OPFBaseline, RuleBasedBaseline, FeedforwardNN
    from failure_engine import CascadingFailureEngine
    print("✅ All imports successful!")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Test 2: Load config
print("\n[2/5] Loading configuration...")
try:
    import yaml
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print(f"✅ Config loaded!")
    print(f"   - Grid: IEEE {config['grid']['buses']}-bus system")
    print(f"   - State dim: {config['ddpg']['state_dim']}")
    print(f"   - Action dim: {config['ddpg']['action_dim']}")
except Exception as e:
    print(f"❌ Config error: {e}")
    sys.exit(1)

# Test 3: Initialize environment
print("\n[3/5] Initializing Grid Environment...")
try:
    env = GridEnv()
    state, _ = env.reset()
    print(f"✅ Environment initialized!")
    print(f"   - State shape: {state.shape}")
    print(f"   - Action space: {env.action_space}")
except Exception as e:
    print(f"❌ Environment error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Initialize DDPG agent
print("\n[4/5] Initializing DDPG Agent...")
try:
    agent = DDPGAgent()
    print(f"✅ DDPG Agent initialized!")
    print(f"   - Actor network: {sum(p.numel() for p in agent.actor.parameters())} parameters")
    print(f"   - Critic network: {sum(p.numel() for p in agent.critic.parameters())} parameters")
except Exception as e:
    print(f"❌ Agent error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Quick training loop (5 episodes)
print("\n[5/5] Running quick training test (5 episodes)...")
try:
    os.makedirs('models/trained_weights', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    episode_rewards = []
    
    for episode in range(5):
        state, _ = env.reset()
        episode_reward = 0
        agent.noise.reset()
        
        for step in range(10):  # Just 10 steps per episode for quick test
            action = agent.select_action(state, add_noise=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            agent.replay_buffer.push(state, action, reward, next_state, float(done))
            
            if len(agent.replay_buffer) >= agent.batch_size:
                critic_loss, actor_loss = agent.train_step()
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        episode_rewards.append(episode_reward)
        print(f"   Episode {episode+1}/5 - Reward: {episode_reward:.3f}")
    
    print(f"✅ Training test successful!")
    print(f"   - Average reward: {np.mean(episode_rewards):.3f}")
    
    # Save test model
    agent.save('models/trained_weights/ddpg_test.pth')
    print(f"✅ Test model saved to models/trained_weights/ddpg_test.pth")
    
except Exception as e:
    print(f"❌ Training error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED - PROJECT SETUP IS WORKING!")
print("=" * 60)
print("\nNext steps:")
print("1. Run full training: python train_ddpg.py")
print("2. Run evaluation: python evaluation.py")
print("=" * 60)
