import torch
from typing import List
import argparse
from pathlib import Path
import yaml

from src.models.model_handler import ModelHandler
from src.data.preprocessor import TextPreprocessor
from src.utils.config import load_config
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class Predictor:
    def __init__(self, model_path: str, config_path: str):
        # Load config
        self.config = load_config(config_path)
        
        # Initialize preprocessor
        self.preprocessor = TextPreprocessor()
        
        # Load model and vocabulary
        self.model, self.vocab = ModelHandler.load_model(
            model_path,
            self.config['model']
        )
        self.model.eval()
        
        # Set device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
    
    def predict(self, texts: List[str]) -> List[int]:
        """Make predictions for input texts"""
        # Preprocess texts
        processed_texts = [self.preprocessor.clean_text(text) for text in texts]
        
        # Convert to tensors
        max_length = self.config['data']['max_length']
        tensors = []
        for text in processed_texts:
            indices = [self.vocab.get(char, 0) for char in text]
            if len(indices) > max_length:
                indices = indices[:max_length]
            else:
                indices += [0] * (max_length - len(indices))
            tensors.append(torch.tensor(indices, dtype=torch.long))
        
        # Stack tensors
        batch = torch.stack(tensors).to(self.device)
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(batch)
            predictions = outputs.argmax(dim=1).cpu().tolist()
        
        return predictions

def main():
    parser = argparse.ArgumentParser(description='Make predictions with trained model')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--config_path', type=str, default='configs/config.yaml', help='Path to config file')
    parser.add_argument('--input_texts', nargs='+', required=True, help='List of texts to predict')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = Predictor(args.model_path, args.config_path)
    
    # Make predictions
    predictions = predictor.predict(args.input_texts)
    
    # Print results
    for text, pred in zip(args.input_texts, predictions):
        logger.info(f"Text: {text}")
        logger.info(f"Prediction: {pred}")
        logger.info("---")

if __name__ == "__main__":
    main() 