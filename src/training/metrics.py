from typing import Dict
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

class MetricsCalculator:
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics"""
        self.predictions = []
        self.targets = []
    
    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        """Update metrics with new predictions"""
        preds = preds.argmax(dim=1).cpu().numpy()
        targets = targets.cpu().numpy()
        
        self.predictions.extend(preds)
        self.targets.extend(targets)
    
    def compute(self) -> Dict[str, float]:
        """Compute all metrics"""
        accuracy = accuracy_score(self.targets, self.predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(
            self.targets,
            self.predictions,
            average='binary'  # Change to 'macro' for multi-class
        )
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics 