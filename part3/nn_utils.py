"""
Neural network utilities for Transformer implementation.
Contains basic building blocks: softmax, cross-entropy, gradient clipping, token accuracy, perplexity.
"""
import torch
from torch import Tensor


def softmax(x: Tensor, dim: int = -1) -> Tensor:
    """
    Compute softmax along the specified dimension.
    
    Args:
        x: Input tensor of any shape
        dim: Dimension along which to compute softmax (default: -1)
    
    Returns:
        Tensor of same shape as input with softmax applied along dim
    """
    # TODO: Implement numerically stable softmax. You can re-use the same one 
    # used in part 2. But for this problem, you need to implement a numerically stable version to pass harder tests.
    maxes = torch.max(x, dim=-1, keepdim=True)[0]
    x_exp = torch.exp(x - maxes)

    total = torch.sum(x_exp, dim=dim, keepdim=True)
    softmax_row = x_exp / total

    return softmax_row

def log_softmax(x, dim=-1):
    x_max_values, max_indices = torch.max(x, dim=dim, keepdims=True)
    log_sum_exp = x_max_values + torch.log(torch.sum(torch.exp(x - x_max_values), dim=dim, keepdims=True))

    return x - log_sum_exp

def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """
    Compute cross-entropy loss.
    
    Args:
        logits: Unnormalized log probabilities of shape (N, C) where N is batch size
                and C is number of classes
        targets: Ground truth class indices of shape (N,)
    
    Returns:
        Scalar tensor containing the mean cross-entropy loss
    """
    #Stuck
    #What to do? 
    #We need to find the ground truth implementation of cross-entropy loss
    #Match their descriptions
    #Implement
    max_values, _ = torch.max(logits, dim=-1, keepdims=True)
    log_probs = logits - max_values - torch.log(torch.sum(torch.exp(logits - max_values), dim=-1, keepdims=True))
    #log_probs = log_softmax(logits, dim=1)
    # 2. Gather the log-probabilities corresponding to the target indices
    # Shape becomes (Batch Size)
    nll_loss = -log_probs.gather(dim=1, index=targets.unsqueeze(1)).squeeze(1)
    
    # 3. Return the mean loss
    return nll_loss.mean()

def gradient_clipping(parameters, max_norm: float) -> Tensor:
    """
    Clip gradients of parameters by global norm.
    
    Args:
        parameters: Iterable of parameters with gradients
        max_norm: Maximum allowed gradient norm
    
    Returns:
        The total norm of the gradients before clipping
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]

    grads = [p.grad for p in parameters if p.grad is not None]

    
    total = torch.linalg.norm(torch.stack([torch.linalg.norm(g) for g in grads]))
    
    if total > max_norm:
        for g in grads:
            with torch.no_grad():
                g.mul_(max_norm / (total + 1e-6))
    
    return total

def token_accuracy(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """
    Compute token-level accuracy for language modeling.
    
    Computes the fraction of tokens where the predicted token (argmax of logits)
    matches the target token, ignoring positions where target equals ignore_index.
    
    Args:
        logits: Predicted logits of shape (N, C) where N is the number of tokens
                and C is the vocabulary size
        targets: Ground truth token indices of shape (N,)
        ignore_index: Target value to ignore when computing accuracy (default: -100)
    
    Returns:
        Scalar tensor containing the accuracy (between 0 and 1)
    
    Example:
        >>> logits = torch.tensor([[2.0, 1.0, 0.5], [0.1, 3.0, 0.2], [1.0, 0.5, 2.5]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> token_accuracy(logits, targets)
        tensor(1.)  # All predictions correct: argmax gives [0, 1, 2]
        
        >>> logits = torch.tensor([[2.0, 1.0], [0.1, 3.0], [1.0, 0.5]])
        >>> targets = torch.tensor([1, 1, 0])
        >>> token_accuracy(logits, targets)
        tensor(0.6667)  # 2 out of 3 correct
    """
    non_ignored = torch.argwhere(targets != ignore_index)

    logits = logits[non_ignored, :]
    targets = targets[non_ignored]

    pred_classes = torch.argmax(logits, dim=-1)

    bool_mask = pred_classes == targets
    digits = bool_mask.to(torch.float32)
    accuracy = torch.mean(digits)

    return accuracy

def perplexity(logits: Tensor, targets: Tensor, ignore_index: int = -100) -> Tensor:
    """
    Compute perplexity for language modeling.
    
    Perplexity is defined as exp(cross_entropy_loss). It measures how well the
    probability distribution predicted by the model matches the actual distribution
    of the tokens. Lower perplexity indicates better prediction.
    
    Args:
        logits: Predicted logits of shape (N, C) where N is the number of tokens
                and C is the vocabulary size
        targets: Ground truth token indices of shape (N,)
        ignore_index: Target value to ignore when computing perplexity (default: -100)
    
    Returns:
        Scalar tensor containing the perplexity (always >= 1)
    
    Example:
        >>> # Perfect predictions (one-hot logits matching targets)
        >>> logits = torch.tensor([[10.0, 0.0, 0.0], [0.0, 10.0, 0.0], [0.0, 0.0, 10.0]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> perplexity(logits, targets)
        tensor(1.0001)  # Close to 1 (perfect)
        
        >>> # Uniform predictions (high uncertainty)
        >>> logits = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        >>> targets = torch.tensor([0, 1, 2])
        >>> perplexity(logits, targets)
        tensor(3.)  # Equal to vocab_size (worst case for uniform)
    """
    # TODO: Implement perplexity
    #get indices of the ignore_index

    #I think we have like double lists or something? 
    #From how? 
    non_ignored = torch.argwhere(targets != ignore_index)

    logits = logits[non_ignored, :]
    targets = targets[non_ignored]

    logits = logits.squeeze(1)
    targets = targets.squeeze(-1)
    
    return torch.exp(cross_entropy(logits, targets))

