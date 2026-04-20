"""one LightningModule for every neural model 

multi-task loss + logging + AdamW with cosine warmup

"""

import lightning as L
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from src.model.components.losses import multi_task_loss


class MempoolLitModule(L.LightningModule):
    """generic training module. expects model.forward(x_num, x_cat, delta_t=None)."""

    def __init__(self, model, lr=1e-3, weight_decay=0.01,
                 warmup_fraction=0.05, warmup_tokens=64, total_epochs=5):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_fraction = warmup_fraction
        self.warmup_tokens = warmup_tokens
        self.total_epochs = total_epochs
        self.save_hyperparameters(ignore=["model"])

    def _shared_step(self, batch, stage="train"):
        x_num = batch["x_num"]
        x_cat = batch["x_cat"]
        delta_t = batch.get("delta_t", None)
        labels = batch["labels"]
        mask = batch.get("mask", None)   # only sequence models supply one

        kwargs = {}
        if delta_t is not None:
            kwargs["delta_t"] = delta_t
        logits = self.model(x_num, x_cat, **kwargs)

        if mask is not None and self.warmup_tokens > 0:
            # mask is (B, L) bool, True == valid
            revert_mask = (labels[..., 0] >= 0) & mask
            # also blank labels/logits at warm-up positions so the loss
            # contribution there is exactly zero
            labels_m = labels.clone()
            labels_m[~mask] = 0
            logits_m = {
                k: v.clone() for k, v in logits.items()
            }
            for k in logits_m:
                logits_m[k][~mask.unsqueeze(-1).expand_as(logits_m[k])] = 0
            total_loss, loss_dict = multi_task_loss(
                logits_m, labels_m, revert_mask=revert_mask,
            )
        else:
            revert_mask = labels[..., 0] >= 0
            total_loss, loss_dict = multi_task_loss(logits, labels, revert_mask=revert_mask)

        self.log(f"{stage}/loss", total_loss, prog_bar=True, sync_dist=True)
        for k, v in loss_dict.items():
            self.log(f"{stage}/{k}", v, sync_dist=True)
        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def configure_optimizers(self):
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # estimate total steps from total_epochs, not trainer.max_epochs.
        # keeps the LR schedule shaped for the full plan even when we
        # stop early or resume from an intermediate ckpt.
        if self.trainer and self.trainer.estimated_stepping_batches:
            estimated = self.trainer.estimated_stepping_batches
            steps_per_epoch = estimated // self.trainer.max_epochs
            total_steps = steps_per_epoch * self.total_epochs
        else:
            total_steps = 1000  # fallback

        warmup_steps = max(1, int(total_steps * self.warmup_fraction))
        cosine_steps = total_steps - warmup_steps

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.01,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cosine_steps,
            eta_min=self.lr * 0.01,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }
