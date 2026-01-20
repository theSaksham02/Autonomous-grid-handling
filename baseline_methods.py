"""
Baseline Methods: OPF, Rule-Based, Feedforward NN
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.neural_network import MLPClassifier

class OPFBaseline:
    """Optimal Power Flow Baseline"""
    
    def predict(self, state):
        """Simple threshold-based prediction"""
        voltage_min = state[:118].min()
        
        if voltage_min < 0.92:
            return 1  # Cascading likely
        else:
            return 0  # Safe

class RuleBasedBaseline:
    """Rule-based heuristic"""
    
    def predict(self, state):
        """Fixed decision rules"""
        voltage_min = state[:118].min()
        voltage_mean = state[:118].mean()
        
        # Rules from paper
        if voltage_min < 0.90:
            return 1
        elif voltage_mean < 0.95:
            return 1
        else:
            return 0

class FeedforwardNN:
    """Simple feedforward neural network"""
    
    def __init__(self, input_dim=247):
        self.model = MLPClassifier(
            hidden_layer_sizes=(512, 256, 128),
            activation='relu',
            solver='adam',
            learning_rate_init=0.001,
            max_iter=500,
            verbose=False
        )
    
    def train(self, X, y):
        """Train on dataset"""
        self.model.fit(X, y)
    
    def predict(self, state):
        """Predict cascading failure"""
        return self.model.predict([state])[0]
    
    def predict_proba(self, state):
        """Predict probability"""
        return self.model.predict_proba([state])[0][1]
