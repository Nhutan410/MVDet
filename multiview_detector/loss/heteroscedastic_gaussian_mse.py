import torch
from torch import nn
import torch.nn.functional as F


class HeteroscedasticGaussianMSE(nn.Module):
    """
    Gaussian-target MSE with a LEARNED per-pixel label-noise variance n (heteroscedastic
    regression, Kendall & Gal 2017; same assumption as eq. (6) of Huynh et al. CVPR 2022 but in
    its closed-form regression version -- no noise sampling).

    Per pixel, with residual r = x - soft_gt and variance n predicted by the model's noise head:

        l = (n_min / n) * r^2  +  n_min * log(n / n_min)

    which is 2*n_min * [ r^2 / (2n) + 1/2 log n ] up to a constant, i.e. the Gaussian NLL scaled
    so that at n = n_min the loss (and its gradient on x) is EXACTLY the plain GaussianMSE.
      - optimum in n is n* = r^2: pixels the model strongly disagrees with (likely a MISSING
        annotation under partial supervision) get a large n and are down-weighted;
      - log n penalises "forgiving" everywhere, so n stays at n_min where the label is fine;
      - n is bounded to [n_min, n_max] via a sigmoid, so sqrt(n_min) is the residual below which
        a pixel is never forgiven (plays the role of the old confuse threshold c, continuously),
        and n_min / n_max is the smallest weight a pixel can ever get.
    Unlike ConfuseGaussianMSE the target is never changed (no mirroring toward 1): gradient on x
    only ever gets WEAKER at suspicious pixels, it never flips sign.

    one_sided (default True): forgive only where x > soft_gt. Under partial annotation the only
    label noise is a MISSING person (gt 0 where the truth is 1), i.e. the x > soft_gt direction;
    kept labels are correct, so under-shooting them (x < soft_gt) must keep full supervision.
    Two-sided forgiveness is degenerate early in training: with x ~ 0 everywhere the largest
    residuals are exactly the kept persons (r = 1), the noise head learns "person-like feature ->
    large n" and down-weights the positives, and the model stays at x ~ 0 (observed on Wildtrack
    drop60: maxima 0.05, recall 1% after epoch 1 vs ~1.0 / 19% for plain MSE).

    forward(x, target, kernel, noise=None): `noise` is the raw noise-head output (same shape as
    x). When it is None (e.g. per-view image heads, which have no noise head) the loss falls
    back to plain GaussianMSE, so the same criterion object serves both heads in the trainer.
    """

    def __init__(self, n_min=0.09, n_max=1.0, one_sided=True):
        super().__init__()
        assert 0 < n_min < n_max
        self.n_min = n_min
        self.n_max = n_max
        self.one_sided = one_sided

    def _traget_transform(self, x, target, kernel):
        target = F.adaptive_max_pool2d(target, x.shape[2:])
        with torch.no_grad():
            target = F.conv2d(target, kernel.float().to(target.device), padding=int((kernel.shape[-1] - 1) / 2))
        return target

    def noise_to_variance(self, noise):
        # single place that defines the raw-head-output -> n mapping (also used for logging/vis)
        return self.n_min + (self.n_max - self.n_min) * torch.sigmoid(noise)

    def forward(self, x, target, kernel, noise=None):
        soft_gt = self._traget_transform(x, target, kernel)
        r2 = (x - soft_gt) ** 2
        if noise is None:
            return r2.mean()

        n = self.noise_to_variance(noise)
        if self.one_sided:
            n = torch.where(x.detach() > soft_gt, n, torch.full_like(n, self.n_min))
        loss = (self.n_min / n) * r2 + self.n_min * torch.log(n / self.n_min)
        return loss.mean()
