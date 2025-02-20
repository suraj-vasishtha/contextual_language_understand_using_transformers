from typing import List, Dict
import torch
from ..models.model_handler import ModelHandler
from ..data.preprocessor import TextPreprocessor
from ..utils.config import load_config

class ModelService:
    def __init__(self, model_path: str, config_path: str):
        self.config = load_config(config_path)
        self.preprocessor = TextPreprocessor()
        
        # Load model and vocabulary
        self.model, self.vocab = ModelHandler.load_model(
            model_path,
            self.config['model']
        )
        self.model.eval()
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
    
    def predict(self, texts: List[str]) -> List[Dict[str, float]]:
        """Make predictions with confidence scores"""
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
        
        # Get predictions with probabilities
        with torch.no_grad():
            outputs = self.model(batch)
            probabilities = torch.softmax(outputs, dim=1)
            
        # Convert to list of dictionaries
        results = []
        for probs in probabilities:
            result = {
                'class_0_prob': float(probs[0]),
                'class_1_prob': float(probs[1]),
                'predicted_class': int(torch.argmax(probs))
            }
            results.append(result)
        
        return results 