"""
Evaluation and Comparison Script
"""
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from ddpg_Agent import DDPGAgent
from grid_env import GridEnv
from baseline_methods import OPFBaseline, RuleBasedBaseline, FeedforwardNN
import pickle

def evaluate_all_methods(dataset_path='data/processed/test_dataset.pkl'):
    """Evaluate DDPG and all baselines"""
    
    # Load test dataset
    with open(dataset_path, 'rb') as f:
        test_data = pickle.load(f)
    
    print(f"📊 Evaluating on {len(test_data)} scenarios...")
    
    # Initialize methods
    ddpg_agent = DDPGAgent()
    ddpg_agent.load('models/trained_weights/ddpg_final.pth')
    
    opf = OPFBaseline()
    rules = RuleBasedBaseline()
    fnn = FeedforwardNN()
    
    # Extract features and labels
    X = np.stack([s['grid_state'][:247] for s in test_data])
    y = np.array([s['is_cascading'] for s in test_data])
    
    # Train FNN on training data (simulate)
    train_data = pickle.load(open('data/processed/train_dataset.pkl', 'rb'))
    X_train = np.stack([s['grid_state'][:247] for s in train_data])
    y_train = np.array([s['is_cascading'] for s in train_data])
    fnn.train(X_train, y_train)
    
    # Get predictions
    results = {}
    
    # DDPG
    ddpg_preds = []
    for state in X:
        action = ddpg_agent.select_action(state, add_noise=False)
        # Use action magnitude as cascade probability proxy
        cascade_score = np.linalg.norm(action)
        ddpg_preds.append(1 if cascade_score > 0.5 else 0)
    results['DDPG'] = np.array(ddpg_preds)
    
    # OPF
    results['OPF'] = np.array([opf.predict(s) for s in X])
    
    # Rule-based
    results['Rule-Based'] = np.array([rules.predict(s) for s in X])
    
    # FNN
    results['FNN'] = np.array([fnn.predict(s) for s in X])
    
    # Compute metrics
    metrics_table = []
    
    for method_name, predictions in results.items():
        metrics = {
            'Method': method_name,
            'Accuracy': accuracy_score(y, predictions),
            'Precision': precision_score(y, predictions),
            'Recall': recall_score(y, predictions),
            'F1-Score': f1_score(y, predictions)
        }
        metrics_table.append(metrics)
    
    metrics_df = pd.DataFrame(metrics_table)
    print("\n" + "="*70)
    print("PERFORMANCE COMPARISON")
    print("="*70)
    print(metrics_df.to_string(index=False))
    
    # Save results
    metrics_df.to_csv('results/comparison_metrics.csv', index=False)
    
    # Plot comparison
    plot_comparison(metrics_df)
    
    return metrics_df

def plot_comparison(metrics_df):
    """Plot performance comparison"""
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        sns.barplot(data=metrics_df, x='Method', y=metric, ax=ax, palette='viridis')
        ax.set_title(metric, fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.set_ylabel('Score')
        ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig('results/performance_comparison.png', dpi=300, bbox_inches='tight')
    print("\n📊 Saved comparison plot to results/performance_comparison.png")

if __name__ == "__main__":
    evaluate_all_methods()
