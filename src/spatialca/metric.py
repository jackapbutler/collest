import torch
import torch.nn.functional as F


def cel(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    target = y.argmax(dim=0)  # [H, W] class indices
    return F.cross_entropy(logits, target.unsqueeze(0))


def mask_coverage(
    softmax: torch.Tensor, y: torch.Tensor, unseen: torch.Tensor, threshold_val: float
) -> torch.Tensor:
    known = unseen < 0.5
    correct_cls = softmax.argmax(dim=0) == y.argmax(dim=0)
    above_prob = softmax.max(dim=0).values >= threshold_val
    correct = (correct_cls & above_prob & ~known).sum() + known.sum()
    return correct / unseen.numel()
