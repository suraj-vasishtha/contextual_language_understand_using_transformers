import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List
import numpy as np
from datetime import datetime
import os

class AnalysisVisualizer:
    def __init__(self):
        """Initialize the visualizer with default style settings."""
        sns.set_theme()  # Use seaborn's default theme
        self.colors = sns.color_palette("husl", 8)
        
    def create_sentiment_comparison(self, text_sentiment: Dict, context_sentiment: Dict, save_path: str = None):
        """
        Create a bar chart comparing text and context sentiment scores.
        
        Args:
            text_sentiment (Dict): Sentiment analysis of main text
            context_sentiment (Dict): Sentiment analysis of context
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Prepare data
        labels = ['Text', 'Context']
        scores = [text_sentiment['score'], context_sentiment['score']]
        sentiments = [text_sentiment['sentiment'], context_sentiment['sentiment']]
        
        # Create bar chart
        bars = plt.bar(labels, scores, color=self.colors[:2])
        
        # Add value labels on top of bars
        for bar, score, sentiment in zip(bars, scores, sentiments):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{score:.2f}\n({sentiment})',
                    ha='center', va='bottom')
        
        plt.title('Sentiment Comparison: Text vs Context')
        plt.ylabel('Sentiment Score')
        plt.ylim(0, 1.1)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved sentiment comparison plot: {save_path}")
        plt.close()
    
    def create_classification_heatmap(self, classification_scores: Dict, save_path: str = None):
        """
        Create a heatmap of classification scores.
        
        Args:
            classification_scores (Dict): Classification scores for different categories
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        
        # Prepare data
        categories = list(classification_scores.keys())
        scores = list(classification_scores.values())
        
        # Create heatmap
        sns.heatmap(np.array(scores).reshape(1, -1),
                   annot=True,
                   fmt='.2f',
                   cmap='YlOrRd',
                   xticklabels=categories,
                   yticklabels=['Score'])
        
        plt.title('Text Classification Scores')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved classification heatmap: {save_path}")
        plt.close()
    
    def create_context_impact_plot(self, text_sentiment: Dict, context_sentiment: Dict, 
                                 impact_score: float, save_path: str = None):
        """
        Create a radar plot showing context impact.
        
        Args:
            text_sentiment (Dict): Sentiment analysis of main text
            context_sentiment (Dict): Sentiment analysis of context
            impact_score (float): Context impact score
            save_path (str, optional): Path to save the plot
        """
        plt.figure(figsize=(8, 8))
        
        # Prepare data
        categories = ['Text Sentiment', 'Context Sentiment', 'Impact']
        values = [text_sentiment['score'], context_sentiment['score'], impact_score]
        
        # Create radar plot
        angles = np.linspace(0, 2*np.pi, len(categories), endpoint=False)
        values = np.concatenate((values, [values[0]]))  # complete the polygon
        angles = np.concatenate((angles, [angles[0]]))  # complete the polygon
        
        ax = plt.subplot(111, polar=True)
        ax.plot(angles, values, 'o-', linewidth=2, color=self.colors[0])
        ax.fill(angles, values, alpha=0.25, color=self.colors[0])
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(categories)
        ax.set_ylim(0, 1)
        
        plt.title('Context Impact Analysis')
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=300)
            print(f"Saved context impact plot: {save_path}")
        plt.close()
    
    def create_analysis_dashboard(self, analysis_results: Dict, output_dir: str = "analysis_plots"):
        """
        Create a complete dashboard of analysis visualizations.
        
        Args:
            analysis_results (Dict): Complete analysis results
            output_dir (str): Directory to save the plots
        """
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create sentiment comparison plot
        self.create_sentiment_comparison(
            analysis_results['text_analysis']['sentiment'],
            analysis_results['context_analysis']['sentiment'],
            os.path.join(output_dir, f'sentiment_comparison_{timestamp}.png')
        )
        
        # Create classification heatmap
        self.create_classification_heatmap(
            analysis_results['text_analysis']['classification']['all_scores'],
            os.path.join(output_dir, f'classification_heatmap_{timestamp}.png')
        )
        
        # Create context impact plot
        self.create_context_impact_plot(
            analysis_results['text_analysis']['sentiment'],
            analysis_results['context_analysis']['sentiment'],
            analysis_results['context_analysis']['combined_analysis']['context_impact'],
            os.path.join(output_dir, f'context_impact_{timestamp}.png')
        )
        
        return {
            'sentiment_comparison': f'sentiment_comparison_{timestamp}.png',
            'classification_heatmap': f'classification_heatmap_{timestamp}.png',
            'context_impact': f'context_impact_{timestamp}.png'
        } 