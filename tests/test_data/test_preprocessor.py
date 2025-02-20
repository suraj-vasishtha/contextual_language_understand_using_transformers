import pytest
from src.data.preprocessor import TextPreprocessor

def test_text_preprocessor():
    preprocessor = TextPreprocessor()
    
    # Test text cleaning
    text = "Hello, World! 123"
    cleaned = preprocessor.clean_text(text)
    assert cleaned.islower()  # Check lowercase
    assert not any(c.isdigit() for c in cleaned)  # Check no digits
    assert not any(c in ',.!?' for c in cleaned)  # Check no punctuation
    
    # Test data preparation
    texts = ["Hello, World!", "Test text", "Another example"]
    labels = [0, 1, 0]
    
    train_texts, val_texts, train_labels, val_labels = preprocessor.prepare_data(
        texts=texts,
        labels=labels,
        test_size=0.34  # Should give 2 train, 1 val
    )
    
    assert len(train_texts) == 2
    assert len(val_texts) == 1
    assert len(train_labels) == 2
    assert len(val_labels) == 1 