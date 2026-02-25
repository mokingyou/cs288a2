"""
Dataset classes for pre-training and fine-tuning.
Example submission.
"""
import json
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


class PretrainingDataset(Dataset):
    def __init__(self, file_path: str | Path, tokenizer, max_length: int = 256, stride: int | None = None):
        self.max_length = max_length
        self.stride = stride or max_length
        
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        
        # Tokenize ONCE during initialization
        tokens = tokenizer.encode(text)
        self.token_ids = torch.tensor(tokens, dtype=torch.long)
        
        if len(self.token_ids) <= max_length:
            self.num_sequences = 1
        else:
            self.num_sequences = (len(self.token_ids) - max_length) // self.stride + 1
    
    def __len__(self) -> int:
        return self.num_sequences
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        start_idx = idx * self.stride
        end_idx = start_idx + self.max_length + 1
        
        # Slicing a tensor is significantly faster than tokenizing strings
        chunk = self.token_ids[start_idx:end_idx]
        
        # Handle final partial batch with padding
        if len(chunk) < self.max_length + 1:
            padding = torch.zeros(self.max_length + 1 - len(chunk), dtype=torch.long)
            chunk = torch.cat([chunk, padding])
            
        return {
            "input_ids": chunk[:-1], 
            "labels": chunk[1:]
        }

class MultipleChoiceQADataset(Dataset):
    def __init__(self, data: List[Dict[str, Any]], tokenizer, max_length: int = 256, num_choices: int = 4):
        self.max_length = max_length
        self.num_choices = num_choices
        
        self.all_input_ids = []
        self.all_attention_masks = []
        self.labels = []

        print(f"Pre-tokenizing {len(data)} examples...")
        
        for example in data:
            context = example["context"]
            question = example["question"]
            choices = example["choices"]
            answer = example.get("answer", 0)
            
            choice_ids = []
            choice_masks = []
            
            for choice in choices:
                text = f"{context}\n\nQuestion: {question}\n\nAnswer: {choice}"
                
                # Use tokenizer.encode_plus for easier padding/masking
                encoded = tokenizer.encode_plus(
                    text,
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                
                choice_ids.append(encoded['input_ids'])
                choice_masks.append(encoded['attention_mask'])
            
            # Stack choices for this specific example
            self.all_input_ids.append(torch.cat(choice_ids, dim=0))
            self.all_attention_masks.append(torch.cat(choice_masks, dim=0))
            self.labels.append(answer)

        # Convert lists to massive tensors to reside in RAM
        self.all_input_ids = torch.stack(self.all_input_ids)
        self.all_attention_masks = torch.stack(self.all_attention_masks)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        print("Pre-tokenization complete.")
    
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        # __getitem__ is now just a lightning-fast memory lookup
        return {
            "input_ids": self.all_input_ids[idx],
            "attention_mask": self.all_attention_masks[idx],
            "labels": self.labels[idx],
        }
    
    @classmethod
    def from_json(cls, file_path: str | Path, tokenizer, **kwargs) -> "MultipleChoiceQADataset":
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data, tokenizer, **kwargs)


def create_pretraining_dataloader(file_path, tokenizer, batch_size=8, max_length=256, stride=None, shuffle=True, num_workers=0, persistent_workers=False):
    dataset = PretrainingDataset(file_path, tokenizer, max_length, stride)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)


def create_qa_dataloader(data, tokenizer, batch_size=4, max_length=256, num_choices=4, shuffle=True, num_workers=0, persistent_workers=False):
    if isinstance(data, (str, Path)):
        dataset = MultipleChoiceQADataset.from_json(data, tokenizer, max_length=max_length, num_choices=num_choices)
    else:
        dataset = MultipleChoiceQADataset(data, tokenizer, max_length=max_length, num_choices=num_choices)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, persistent_workers=persistent_workers)