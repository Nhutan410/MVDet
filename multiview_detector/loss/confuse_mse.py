import torch
from torch import nn
import torch.nn.functional as F


class ConfuseMSE(nn.Module):
    """
    Variant of BRLGaussianMSE that drops the Gaussian pos_thr gate entirely.

    Three regions, no Gaussian smoothing used to define them:
      - positive:          hard ground-truth occupancy (target > 0), MSE to 1
      - confused-positive: NOT positive, AND prediction >= c (self.confuse_pred_thr)
      - background:        everything else, MSE to 0

    Known trade-off (discussed before running this experiment): because "positive"
    is now just the exact annotated points instead of the wider Gaussian region
    around them, the halo of pixels around a KNOWN/KEPT (non-dropped) person also
    gets pred >= c once the model learns correctly, and lands in confused-positive
    too -- weakening supervision there (mirror=True) or down-weighting it
    (mirror=False), even at drop_ratio=0. Kept deliberately for this experiment
    to see the effect empirically; see BRLGaussianMSE for the pos_thr-gated version.
    """

    def __init__(self, confuse_pred_thr=0.3, beta=0.1, mirror=True):
        super().__init__()
        self.confuse_pred_thr = confuse_pred_thr
        self.beta = beta
        self.mirror = mirror

    def _hard_target(self, x, target):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        return (target > 0).float()

    def _traget_transform(self, x, target, kernel=None):
        # kept for trainer.py's visualize=True path, which calls
        # criterion._traget_transform(map_res, map_gt, map_kernel) to imshow the target
        return self._hard_target(x, target)

    def forward(self, x, target, kernel=None):
        hard_gt = self._hard_target(x, target)

        pos_mask = hard_gt > 0
        bg_mask = ~pos_mask

        # assignment must not backprop through the thresholding
        confuse_mask = bg_mask & (x.detach() >= self.confuse_pred_thr)
        easy_neg_mask = bg_mask & ~confuse_mask

        loss = x.new_zeros(())
        count = x.new_zeros(())

        if pos_mask.any():
            loss = loss + F.mse_loss(x[pos_mask], hard_gt[pos_mask], reduction="sum")
            count = count + pos_mask.sum()

        if easy_neg_mask.any():
            loss = loss + F.mse_loss(
                x[easy_neg_mask], hard_gt[easy_neg_mask], reduction="sum"
            )
            count = count + easy_neg_mask.sum()

        if confuse_mask.any():
            if self.mirror:
                mirror_tgt = torch.ones_like(x[confuse_mask])
                loss = loss + self.beta * F.mse_loss(
                    x[confuse_mask], mirror_tgt, reduction="sum"
                )
            else:
                loss = loss + self.beta * F.mse_loss(
                    x[confuse_mask], hard_gt[confuse_mask], reduction="sum"
                )
            count = count + confuse_mask.sum()

        return loss / count.clamp(min=1).float()
