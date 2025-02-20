import re
from typing import List, Tuple
from sklearn.model_selection import train_test_split
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging

logger = logging.getLogger(__name__)

class TextPreprocessor:
    def __init__(self, remove_stopwords: bool = True, lowercase: bool = True):
        self.remove_stopwords = remove_stopwords
        self.lowercase = lowercase
        
        # Download all required NLTK data
        try:
            nltk.download('punkt')
            nltk.download('stopwords')
            nltk.download('averaged_perceptron_tagger')
            nltk.download('wordnet')
            nltk.download('omw-1.4')
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.warning(f"Could not download NLTK data: {e}")
            self.stop_words = set()

    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Convert to lowercase if specified
        if self.lowercase:
            text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Simple word tokenization (without using NLTK's word_tokenize)
        tokens = text.split()
        
        # Remove stopwords if specified
        if self.remove_stopwords:
            tokens = [t for t in tokens if t not in self.stop_words]
        
        return ' '.join(tokens)

    def prepare_data(self, texts: List[str], labels: List[int], 
                    test_size: float = 0.2, random_state: int = 42) -> Tuple[List[str], List[str], List[int], List[int]]:
        """Prepare and split data into train and validation sets"""
        # Clean texts
        cleaned_texts = [self.clean_text(text) for text in texts]
        
        # Split data
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            cleaned_texts, labels, test_size=test_size, random_state=random_state
        )
        
        return train_texts, val_texts, train_labels, val_labels 