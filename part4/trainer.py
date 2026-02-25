"""
Training utilities.
Example submission.
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from dataclasses import dataclass
from typing import Optional, Dict, Any, Callable
from pathlib import Path
import time
import sys

_parent = str(Path(__file__).parent.parent)
if _parent not in sys.path:
    sys.path.insert(0, _parent)

from part3.nn_utils import cross_entropy, gradient_clipping


@dataclass
class TrainingConfig:
    num_epochs: int = 3
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    max_grad_norm: float = 1.0
    batch_size: int = 8
    log_interval: int = 10
    save_interval: int = 500
    checkpoint_dir: Optional[str] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp: bool = False
    patience: Optional[int] = None


class Trainer:
    def __init__(self, model: nn.Module, config: TrainingConfig, train_dataloader: DataLoader, val_dataloader: Optional[DataLoader] = None, compute_loss_fn: Optional[Callable] = None):
        self.model = model.to(config.device)
        self.config = config
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.compute_loss_fn = compute_loss_fn or self._default_lm_loss
        self.optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
        total_steps = len(train_dataloader) * config.num_epochs
        if config.warmup_steps > 0:
            warmup = LinearLR(self.optimizer, start_factor=0.01, end_factor=1.0, total_iters=config.warmup_steps)
            main = CosineAnnealingLR(self.optimizer, T_max=total_steps - config.warmup_steps)
            self.scheduler = SequentialLR(self.optimizer, [warmup, main], milestones=[config.warmup_steps])
        else:
            self.scheduler = CosineAnnealingLR(self.optimizer, T_max=total_steps)
        self.global_step = 0
        self.best_val_loss = float("inf")
        self.train_losses = []
        self.val_losses = []
        self.scaler = torch.amp.GradScaler("cuda" ,enabled=self.config.use_amp)
    
    def _default_lm_loss(self, batch: Dict[str, torch.Tensor], model: nn.Module) -> torch.Tensor:
        input_ids = batch["input_ids"].to(self.config.device, non_blocking=True)
        labels = batch["labels"].to(self.config.device, non_blocking=True)
        logits = model(input_ids)
        batch_size, seq_len, vocab_size = logits.shape
        return cross_entropy(logits.view(-1, vocab_size), labels.view(-1))
    
    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        if type(self.config.device) == str:
            device_type = self.config.device
        else:
            device_type = self.config.device.type

        t_prev = time.perf_counter()
        for batch in self.train_dataloader:
            #TODO my code
            data_time= time.perf_counter() - t_prev
            t0 = time.perf_counter()
            
            
            with torch.autocast(device_type=device_type, dtype=torch.float16, enabled=self.config.use_amp):
                loss = self.compute_loss_fn(batch, self.model)

            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)

            self.scaler.step(self.optimizer)
            
            self.scaler.update()
            self.optimizer.zero_grad(set_to_none=True)
            self.scheduler.step()
            
            compute_time = time.perf_counter() - t0
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1
            
            t_prev = time.perf_counter()
            if num_batches == 1:
                print(
                    f"Total time for batch #{num_batches} is {compute_time + data_time:.2f}s"
                    f"compute time: {compute_time:2f}s"
                    f"data time: {data_time:2f}s"
                    )

        return total_loss / num_batches if num_batches > 0 else 0.0
    
    @torch.no_grad()
    def evaluate(self) -> float:
        if self.val_dataloader is None:
            return 0.0
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        for batch in self.val_dataloader:
            #batch = batch.to(self.device)
            loss = self.compute_loss_fn(batch, self.model)
            total_loss += loss.item()
            num_batches += 1
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def train(self) -> Dict[str, Any]:
        stalled_epochs = 0
        prev_val = float('inf')

        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            if self.val_dataloader:
                val_loss = self.evaluate()
                self.val_losses.append(val_loss)
                if val_loss < prev_val - 0.01:
                    stalled_epochs = 0
                    prev_val = val_loss
                else:
                    stalled_epochs += 1
                
                if stalled_epochs > self.config.patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
                print(f"Completed epoch #{epoch} with val_loss: {val_loss}")
            print(f"Completed epoch #{epoch} with train_loss {train_loss}")

        return {"train_losses": self.train_losses, "val_losses": self.val_losses}


def compute_qa_loss(batch: Dict[str, torch.Tensor], model: nn.Module, device: str = "cuda") -> torch.Tensor:
    input_ids = batch["input_ids"].to(device, non_blocking=True)
    attention_mask = batch["attention_mask"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    logits = model(input_ids, attention_mask)
    return cross_entropy(logits, labels)


def create_qa_loss_fn(device: str = "cuda") -> Callable:
    return lambda batch, model: compute_qa_loss(batch, model, device)
