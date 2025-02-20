import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import numpy as np
from pathlib import Path

class TrainingVisualizer:
    def __init__(self, save_dir: str = 'visualizations'):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def plot_training_history(self, 
                            train_losses: List[float], 
                            val_losses: List[float],
                            train_metrics: List[Dict[str, float]],
                            val_metrics: List[Dict[str, float]]):
        """Plot training history including loss and metrics"""
        epochs = range(1, len(train_losses) + 1)
        
        # Plot losses
        plt.figure(figsize=(10, 6))
        plt.plot(epochs, train_losses, 'b-', label='Training Loss')
        plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(self.save_dir / 'loss_history.png')
        plt.close()
        
        # Plot metrics
        metrics_names = train_metrics[0].keys()
        for metric in metrics_names:
            plt.figure(figsize=(10, 6))
            train_metric_values = [m[metric] for m in train_metrics]
            val_metric_values = [m[metric] for m in val_metrics]
            
            plt.plot(epochs, train_metric_values, 'b-', label=f'Training {metric}')
            plt.plot(epochs, val_metric_values, 'r-', label=f'Validation {metric}')
            plt.title(f'Training and Validation {metric}')
            plt.xlabel('Epochs')
            plt.ylabel(metric)
            plt.legend()
            plt.savefig(self.save_dir / f'{metric}_history.png')
            plt.close()
    
    def plot_confusion_matrix(self, true_labels: List[int], predictions: List[int]):
        """Plot confusion matrix"""
        cm = np.zeros((2, 2))  # Assuming binary classification
        for t, p in zip(true_labels, predictions):
            cm[t][p] += 1
            
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='g', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(self.save_dir / 'confusion_matrix.png')
        plt.close() 