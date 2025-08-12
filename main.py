import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import List, Dict, Union
import os
import warnings
import sys
from text_analyzer import TextAnalyzer
from visualizer import AnalysisVisualizer
import json
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

class ContextualLanguageModel:
    def __init__(self, model_name: str = "prajjwal1/bert-tiny"):
        """
        Initialize the contextual language model.
        
        Args:
            model_name (str): Name of the pre-trained model to use
        """
        try:
            # Set environment variable to disable symlinks warning
            os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'
            
            print("Loading tokenizer...", flush=True)
            sys.stdout.flush()
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            print("Loading model...", flush=True)
            sys.stdout.flush()
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=2,
                cache_dir="./model_cache",  # Local cache directory
                local_files_only=False  # Force download if not present
            )
            print("Model loaded successfully!", flush=True)
            sys.stdout.flush()
            
            # Initialize text analyzer and visualizer
            print("Initializing text analyzer and visualizer...", flush=True)
            self.text_analyzer = TextAnalyzer()
            self.visualizer = AnalysisVisualizer()
            print("Initialization complete!", flush=True)
            
        except Exception as e:
            print(f"Error loading model: {str(e)}", flush=True)
            raise
        
    def analyze_text(self, text: str) -> Dict[str, Union[float, List[float]]]:
        """
        Analyze the given text using the transformer model.
        
        Args:
            text (str): Input text to analyze
        Returns:
            Dict containing analysis results
        """
        try:
            print(f"Analyzing text: {text[:50]}...", flush=True)
            # Tokenize the text
            inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=128)
            
            # Get model predictions
            with torch.no_grad():
                outputs = self.model(**inputs)
                predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
                
            # Convert predictions to numpy array
            predictions = predictions.numpy()[0]
            
            # Get additional analysis
            sentiment_analysis = self.text_analyzer.analyze_sentiment(text)
            categories = ["positive", "negative", "neutral", "informative", "opinion"]
            classification = self.text_analyzer.classify_text(text, categories)
            
            result = {
                "confidence": float(np.max(predictions)),
                "predictions": predictions.tolist(),
                "sentiment": sentiment_analysis,
                "classification": classification
            }
            print(f"Analysis complete. Confidence: {result['confidence']:.2f}", flush=True)
            return result
            
        except Exception as e:
            print(f"Error analyzing text: {str(e)}", flush=True)
            return {
                "confidence": 0.0,
                "predictions": [0.0, 0.0],
                "sentiment": {"sentiment": "unknown", "score": 0.0},
                "classification": {"label": "unknown", "score": 0.0, "all_scores": {}}
            }
    
    def analyze_context(self, text: str, context: str) -> Dict[str, Union[float, List[float]]]:
        """
        Analyze text with additional context.
        
        Args:
            text (str): Main text to analyze
            context (str): Additional context
            
        Returns:
            Dict containing analysis results
        """
        try:
            print(f"\nAnalyzing with context:\nContext: {context}\nText: {text}", flush=True)
            
            # Get text analysis
            text_sentiment = self.text_analyzer.analyze_sentiment(text)
            text_classification = self.text_analyzer.classify_text(text, ["positive", "negative", "neutral", "informative", "opinion"])
            
            # Get context analysis
            context_sentiment = self.text_analyzer.analyze_sentiment(context)
            
            # Calculate context impact
            context_impact = abs(text_sentiment["score"] - context_sentiment["score"])
            
            # Prepare results
            result = {
                "text_analysis": {
                    "sentiment": text_sentiment,
                    "classification": text_classification
                },
                "context_analysis": {
                    "sentiment": context_sentiment
                },
                "combined_analysis": {
                    "context_impact": context_impact
                }
            }
            
            # Create visualizations
            plot_paths = self.visualizer.create_analysis_dashboard(result)
            result['visualizations'] = plot_paths
            
            return result
        except Exception as e:
            print(f"Error analyzing context: {str(e)}")
            return {
                "text_analysis": {
                    "sentiment": {"sentiment": "unknown", "score": 0.0},
                    "classification": {"label": "unknown", "score": 0.0, "all_scores": {}}
                },
                "context_analysis": {
                    "sentiment": {"sentiment": "unknown", "score": 0.0}
                },
                "combined_analysis": {
                    "context_impact": 0.0
                },
                "visualizations": {}
            }

def save_analysis_results(results: List[Dict], filename: str = None):
    """
    Save analysis results to a JSON file.
    
    Args:
        results (List[Dict]): List of analysis results
        filename (str, optional): Name of the output file
    """
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_results_{timestamp}.json"
    
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {filename}")
    except Exception as e:
        print(f"Error saving results: {str(e)}")

def main():
    try:
        print("Starting the program...", flush=True)
        # Create model cache directory if it doesn't exist
        os.makedirs("./model_cache", exist_ok=True)
        print("Model cache directory created/verified", flush=True)
        
        # Initialize the model
        print("Initializing model...", flush=True)
        model = ContextualLanguageModel()
        
        # Example texts
        examples = [
            {
                "text": "I love this movie!",
                "context": "The film received mixed reviews from critics."
            },
            {
                "text": "The service was terrible.",
                "context": "The restaurant was packed and understaffed."
            },
            {
                "text": "This book is a masterpiece!",
                "context": "The author won the Nobel Prize in Literature."
            }
        ]
        
        # Analyze examples
        print("\nAnalyzing examples with context:\n", flush=True)
        results = []
        for i, example in enumerate(examples, 1):
            print(f"\nExample {i}:", flush=True)
            print(f"Text: {example['text']}", flush=True)
            print(f"Context: {example['context']}", flush=True)
            result = model.analyze_context(example['text'], example['context'])
            print(f"Analysis result: {json.dumps(result, indent=2)}\n", flush=True)
            print(f"Visualizations created:", flush=True)
            for k, v in result['visualizations'].items():
                full_path = os.path.abspath(os.path.join('analysis_plots', v))
                print(f"  {k}: {full_path}", flush=True)
                try:
                    os.startfile(full_path)
                except Exception as e:
                    print(f"Could not open {full_path}: {e}", flush=True)
            results.append({
                "example": example,
                "analysis": result
            })
        
        # Save results
        save_analysis_results(results)
            
    except Exception as e:
        print(f"An error occurred: {str(e)}", flush=True)

if __name__ == "__main__":
    main() 