"""multi-task loss = 1*WBCE(rev) + 2*Focal(mev) + 0.5*BCE(drop).

revert label can be -1 (no receipt observed) — masked out of the loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """focal loss for the rare MEV positive (~0.09%).

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)
    """

    def __init__(self, gamma=2.0, alpha=0.95):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = torch.sigmoid(logits)
        p_t = targets * p_t + (1 - targets) * (1 - p_t)
        w = (1 - p_t) ** self.gamma
        a = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (a * w * bce).mean()


def multi_task_loss(logits_dict, labels, revert_mask=None):
    """combined loss + per-task component dict (for logging)."""
    revert_logits = logits_dict["revert"].squeeze(-1)
    mev_logits = logits_dict["mev"].squeeze(-1)
    drop_logits = logits_dict["drop"].squeeze(-1)

    revert_labels = labels[..., 0]
    mev_labels = labels[..., 1]
    drop_labels = labels[..., 2]

    # weighted BCE on revert; mask the -1s
    if revert_mask is None:
        revert_mask = revert_labels >= 0
    if revert_mask.any():
        revert_loss = F.binary_cross_entropy_with_logits(
            revert_logits[revert_mask],
            revert_labels[revert_mask].clamp(0, 1),
            pos_weight=torch.tensor(10.0, device=revert_logits.device),
        )
    else:
        revert_loss = torch.tensor(0.0, device=revert_logits.device)

    mev_loss = FocalLoss(gamma=2.0, alpha=0.95)(mev_logits, mev_labels)
    drop_loss = F.binary_cross_entropy_with_logits(drop_logits, drop_labels)

    total = 1.0 * revert_loss + 2.0 * mev_loss + 0.5 * drop_loss
    return total, {
        "revert_loss": revert_loss.detach(),
        "mev_loss": mev_loss.detach(),
        "drop_loss": drop_loss.detach(),
    }
