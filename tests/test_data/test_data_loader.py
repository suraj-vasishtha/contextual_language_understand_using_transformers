import pytest
import torch
from src.data.data_loader import TextDataset, create_data_loaders

def test_text_dataset():
    # Test data
    texts = ["hello world", "test text", "example"]
    labels = [0, 1, 0]
    
    # Create dataset
    dataset = TextDataset(texts, labels)
    
    # Test length
    assert len(dataset) == 3
    
    # Test getting item
    text_tensor, label_tensor = dataset[0]
    assert isinstance(text_tensor, torch.Tensor)
    assert isinstance(label_tensor, torch.Tensor)
    assert label_tensor.item() == 0

def test_data_loaders():
    # Test data
    train_texts = ["train1", "train2"]
    train_labels = [0, 1]
    val_texts = ["val1"]
    val_labels = [1]
    
    # Create datasets
    train_dataset = TextDataset(train_texts, train_labels)
    val_dataset = TextDataset(val_texts, val_labels)
    
    # Create data loaders
    train_loader, val_loader = create_data_loaders(
        train_dataset,
        val_dataset,
        batch_size=2
    )
    
    # Test batch sizes
    assert len(train_loader) == 1  # 2 samples with batch_size=2
    assert len(val_loader) == 1    # 1 sample 