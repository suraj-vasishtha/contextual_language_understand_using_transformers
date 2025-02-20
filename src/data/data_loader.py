from typing import Tuple, List
import torch
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], max_length: int = 512):
        self.texts = texts
        self.labels = labels
        self.max_length = max_length
        
        # Create a simple vocabulary (in practice, you'd want to use a proper tokenizer)
        self.vocab = {char: idx + 1 for idx, char in enumerate(set(''.join(texts)))}
        self.vocab['<pad>'] = 0
        
    def __len__(self) -> int:
        return len(self.texts)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to indices
        indices = [self.vocab.get(char, 0) for char in text]
        
        # Pad or truncate to max_length
        if len(indices) > self.max_length:
            indices = indices[:self.max_length]
        else:
            indices += [0] * (self.max_length - len(indices))
        
        # Convert to tensors
        text_tensor = torch.tensor(indices, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return text_tensor, label_tensor

def create_data_loaders(
    train_data: TextDataset,
    val_data: TextDataset,
    batch_size: int
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    return train_loader, val_loader
