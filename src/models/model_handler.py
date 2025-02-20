import torch
from typing import Dict, Any, Optional, Tuple
import json
from pathlib import Path
from ..models.transformer import TransformerModel
import torch.serialization

class ModelHandler:
    def __init__(self, model: TransformerModel, save_dir: str = 'model_checkpoints'):
        self.model = model
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(exist_ok=True)
    
    def save_model(self, epoch: int, optimizer: torch.optim.Optimizer, 
                  loss: float, metrics: Dict[str, float], vocab: Dict[str, int],
                  is_best: bool = False):
        """Save model checkpoint and metadata"""
        # Create checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'metrics': metrics,
            'vocab': vocab
        }
        
        # Save checkpoint
        checkpoint_path = self.save_dir / f'checkpoint_epoch_{epoch}.pth'
        torch.save(checkpoint, checkpoint_path, pickle_protocol=4)
        
        # Save best model separately
        if is_best:
            best_path = self.save_dir / 'best_model.pth'
            torch.save(checkpoint, best_path, pickle_protocol=4)
        
        # Save metadata
        metadata = {
            'epoch': epoch,
            'loss': loss,
            'metrics': metrics,
            'vocab_size': len(vocab)
        }
        metadata_path = self.save_dir / f'metadata_epoch_{epoch}.json'
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=4)
    
    @staticmethod
    def load_model(model_path: str, model_config: Dict[str, Any]) -> Tuple[TransformerModel, Dict[str, int]]:
        """Load model and vocabulary from checkpoint"""
        # Load checkpoint with weights_only=False
        checkpoint = torch.load(
            model_path,
            weights_only=False,
            map_location=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        )
        
        # Update config with vocab size from checkpoint
        model_config['vocab_size'] = len(checkpoint['vocab'])
        
        # Initialize model with config
        model = TransformerModel(**model_config)
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, checkpoint['vocab'] 