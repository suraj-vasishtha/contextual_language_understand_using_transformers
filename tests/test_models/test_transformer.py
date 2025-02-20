import pytest
import torch
from src.models.transformer import TransformerModel

def test_transformer_model():
    # Model parameters
    vocab_size = 100
    d_model = 32
    nhead = 4
    num_layers = 2
    dim_feedforward = 64
    num_classes = 2
    
    # Create model
    model = TransformerModel(
        vocab_size=vocab_size,
        d_model=d_model,
        nhead=nhead,
        num_layers=num_layers,
        dim_feedforward=dim_feedforward,
        num_classes=num_classes
    )
    
    # Test input
    batch_size = 4
    seq_length = 10
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    
    # Forward pass
    output = model(x)
    
    # Check output shape
    assert output.shape == (batch_size, num_classes)
    
    # Check if model can be trained
    loss = output.sum()
    loss.backward()
    
    # Check if gradients are computed
    assert model.fc.weight.grad is not None 