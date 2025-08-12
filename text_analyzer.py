from transformers import pipeline
import torch
from typing import Dict, List, Union, Tuple
import numpy as np

class TextAnalyzer:
    def __init__(self):
        """Initialize text analysis pipelines."""
        self.sentiment_analyzer = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")
        self.zero_shot_classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        
    def analyze_sentiment(self, text: str) -> Dict[str, Union[str, float]]:
        """
        Analyze the sentiment of the given text.
        
        Args:
            text (str): Input text to analyze
            
        Returns:
            Dict containing sentiment label and score
        """
        try:
            result = self.sentiment_analyzer(text)[0]
            return {
                "sentiment": result["label"],
                "score": result["score"]
            }
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return {"sentiment": "unknown", "score": 0.0}
    
    def classify_text(self, text: str, candidate_labels: List[str]) -> Dict[str, Union[str, float, List[float]]]:
        """
        Classify text into given categories.
        
        Args:
            text (str): Input text to classify
            candidate_labels (List[str]): List of possible categories
            
        Returns:
            Dict containing classification results
        """
        try:
            result = self.zero_shot_classifier(text, candidate_labels)
            return {
                "label": result["labels"][0],
                "score": result["scores"][0],
                "all_scores": dict(zip(result["labels"], result["scores"]))
            }
        except Exception as e:
            print(f"Error in text classification: {str(e)}")
            return {"label": "unknown", "score": 0.0, "all_scores": {}}
    
    def analyze_with_context(self, text: str, context: str) -> Dict[str, Union[Dict, List]]:
        """
        Perform comprehensive analysis of text with context.
        
        Args:
            text (str): Main text to analyze
            context (str): Additional context
            
        Returns:
            Dict containing various analysis results
        """
        try:
            # Analyze sentiment
            sentiment_result = self.analyze_sentiment(text)
            
            # Classify text
            categories = ["positive", "negative", "neutral", "informative", "opinion"]
            classification_result = self.classify_text(text, categories)
            
            # Analyze context impact
            context_sentiment = self.analyze_sentiment(context)
            
            return {
                "text_analysis": {
                    "sentiment": sentiment_result,
                    "classification": classification_result
                },
                "context_analysis": {
                    "sentiment": context_sentiment
                },
                "combined_analysis": {
                    "context_impact": abs(sentiment_result["score"] - context_sentiment["score"])
                }
            }
        except Exception as e:
            print(f"Error in comprehensive analysis: {str(e)}")
            return {
                "text_analysis": {"sentiment": {"sentiment": "unknown", "score": 0.0}},
                "context_analysis": {"sentiment": {"sentiment": "unknown", "score": 0.0}},
                "combined_analysis": {"context_impact": 0.0}
            } 