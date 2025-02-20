import torch
import torch.nn as nn
from torch.optim import Adam
import json
from pathlib import Path

from src.data.data_loader import TextDataset, create_data_loaders
from src.models.transformer import TransformerModel
from src.training.trainer import Trainer
from src.models.model_handler import ModelHandler
from src.data.preprocessor import TextPreprocessor
from src.utils.visualizer import TrainingVisualizer
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def save_metrics(metrics: dict, epoch: int, save_dir: str = 'results'):
    Path(save_dir).mkdir(exist_ok=True)
    with open(f'{save_dir}/metrics_epoch_{epoch}.json', 'w') as f:
        json.dump(metrics, f, indent=4)

def main():
    # Load configuration
    config = load_config('configs/config.yaml')
    
    # Create dummy data for this example
    train_texts = ["sample text 1", "sample text 2", "sample text 3"]
    train_labels = [0, 1, 0]
    val_texts = ["sample text 4", "sample text 5"]
    val_labels = [1, 0]
    
    # Initialize preprocessor and visualizer
    preprocessor = TextPreprocessor()
    visualizer = TrainingVisualizer()
    
    # Preprocess and split data
    train_texts, val_texts, train_labels, val_labels = preprocessor.prepare_data(
        texts=train_texts + val_texts,  # Combine all texts
        labels=train_labels + val_labels,  # Combine all labels
        test_size=0.2
    )
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels)
    val_dataset = TextDataset(val_texts, val_labels)
    
    # Get vocabulary from the dataset
    vocab = train_dataset.vocab
    vocab_size = len(vocab)
    
    # Update model config with actual vocab size
    config['model']['vocab_size'] = vocab_size
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size=config['training']['batch_size']
    )
    
    # Initialize model
    model = TransformerModel(**config['model'])
    
    # Setup training components
    optimizer = Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )
    criterion = nn.CrossEntropyLoss()
    
    # Initialize trainer and model handler
    trainer = Trainer(model, optimizer, criterion)
    model_handler = ModelHandler(model)
    
    # Training loop
    num_epochs = config['training']['num_epochs']
    best_val_loss = float('inf')
    
    # Track metrics
    train_losses = []
    val_losses = []
    train_metrics_history = []
    val_metrics_history = []
    
    for epoch in range(num_epochs):
        # Training
        train_loss, train_metrics = trainer.train_epoch(train_loader)
        val_loss, val_metrics = trainer.evaluate(val_loader)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_metrics_history.append(train_metrics)
        val_metrics_history.append(val_metrics)
        
        # Check if this is the best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
        
        # Save checkpoint with vocabulary
        model_handler.save_model(
            epoch=epoch + 1,
            optimizer=optimizer,
            loss=val_loss,
            metrics=val_metrics,
            vocab=vocab,  # Add vocabulary here
            is_best=is_best
        )
        
        # Logging
        logger.info(f"Epoch {epoch+1}/{num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        logger.info(f"Train Metrics: {train_metrics}")
        logger.info(f"Val Metrics: {val_metrics}")
    
    # After training, visualize results
    visualizer.plot_training_history(
        train_losses,
        val_losses,
        train_metrics_history,
        val_metrics_history
    )
    
    # Plot final confusion matrix
    _, final_metrics = trainer.evaluate(val_loader)
    visualizer.plot_confusion_matrix(
        val_labels,
        trainer.metrics.predictions
    )

if __name__ == "__main__":
    main() 