import math
from typing import Union

import torch
import torch.nn.functional as F


def soft_ce(logits, target, cfg, apply_symlog: bool = True):
    """Computes the cross entropy loss between logits and soft targets."""
    pred = F.log_softmax(logits, dim=-1)
    target = two_hot(target, cfg, apply_symlog=apply_symlog)
    return -(target * pred).sum(-1, keepdim=True)


@torch.jit.script
def log_std(x, low, dif):
    return low + 0.5 * dif * (torch.tanh(x) + 1)


@torch.jit.script
def _gaussian_residual(eps, log_std):
    return -0.5 * eps.pow(2) - log_std


def _gaussian_logprob(residual):
    return residual - 0.5 * math.log(2 * math.pi)


def gaussian_logprob(eps, log_std, size=None):
    """Compute Gaussian log probability."""
    residual = _gaussian_residual(eps, log_std).sum(-1, keepdim=True)
    if size is None:
        size = eps.size(-1)
    return _gaussian_logprob(residual) * size


def squash(mu, pi, log_pi_values):
    """Apply squashing function."""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    log_pi = log_pi_values - torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(
        -1, keepdim=True
    )
    return mu, pi, log_pi


@torch.jit.script
def symlog(x):
    """
    Symmetric logarithmic function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * torch.log(1 + torch.abs(x))


@torch.jit.script
def symexp(x):
    """
    Symmetric exponential function.
    Adapted from https://github.com/danijar/dreamerv3.
    """
    return torch.sign(x) * (torch.exp(torch.abs(x)) - 1)


def two_hot(x, cfg, apply_symlog=True):
    """Converts a batch of scalars to soft two-hot encoded targets for discrete regression."""
    assert cfg.num_bins > 1, "Number of bins must be greater than 1"

    assert x.dim() >= 2 and x.size(-1) == 1, "Input must be scalars"
    x = symlog(x) if apply_symlog else x

    # clamp the input to the range [vmin, vmax]
    x = torch.clamp(x, cfg.vmin, cfg.vmax).squeeze(-1)
    original_shape = x.shape
    x = x.flatten()
    bin_idx = torch.floor((x - cfg.vmin) / cfg.bin_size).long()

    # calculate the offset from the lower bin boundary
    bin_offset = ((x - cfg.vmin) / cfg.bin_size - bin_idx.float()).unsqueeze(-1)
    assert (
        bin_offset.dim() == 2 and bin_offset.size(1) == 1
    ), "Offset must be 2D tensors"

    # create a soft two-hot encoding
    soft_two_hot = torch.zeros((x.size(0), cfg.num_bins), device=x.device)
    soft_two_hot.scatter_(1, bin_idx.unsqueeze(1), 1 - bin_offset)
    soft_two_hot.scatter_(1, (bin_idx.unsqueeze(1) + 1) % cfg.num_bins, bin_offset)

    soft_two_hot = soft_two_hot.view(*original_shape, cfg.num_bins)
    return soft_two_hot


def two_hot_inv(logits, cfg, apply_symexp=True):
    """Converts a batch of soft two-hot encoded vectors to scalars."""
    assert cfg.num_bins > 1, "Number of bins must be greater than 1"
    assert (
        logits.dim() >= 2 and logits.size(-1) == cfg.num_bins
    ), "Logits dim must match number of bins"
    bins = torch.linspace(cfg.vmin, cfg.vmax, cfg.num_bins, device=logits.device)
    assert (
        bins.dim() == 1 and bins.size(0) == cfg.num_bins
    ), "number of bins must match logits"

    x = F.softmax(logits, dim=-1)
    x = torch.sum(x * bins, dim=-1, keepdim=True)
    assert x.dim() >= 2 and x.size(-1) == 1, "Output must be scalars"

    return symexp(x) if apply_symexp else x


def get_values_from_categoricals(categorical, v_atoms, apply_symexp: bool = True):
    """Converts a batch of categorical distributions to scalars."""
    assert categorical.dim() >= 2 and categorical.size(-1) == v_atoms.size(-1)
    assert v_atoms.dim() == 1
    values = torch.sum(categorical * v_atoms, dim=-1, keepdim=True)
    if apply_symexp:
        return symexp(values)
    else:
        return values


def categorical_atoms(vmin, vmax, num_bins, device=None):
    """Returns the support atoms for a categorical distribution."""
    support_atoms = torch.linspace(vmin, vmax, num_bins, device=device)
    return support_atoms


def categorical_l2_project(
    z_p: torch.Tensor, probs: torch.Tensor, z_q: torch.Tensor
) -> torch.Tensor:
    """
    Projects a categorical distribution (z_p, probs) onto a different support z_q
    in PyTorch.

    Args:
        z_p: support of distribution p.
        probs: probability values.
        z_q: support to project distribution (z_p, probs) onto.

    Returns:
        Projection of (z_p, probs) onto support z_q under Cramer distance.
    """
    # Assert that z_p, probs, and z_q are 1D tensors
    assert z_p.ndim == 1, "z_p must be a 1D tensor"
    assert probs.ndim == 1, "probs must be a 1D tensor"
    assert z_q.ndim == 1, "z_q must be a 1D tensor"

    kp = z_p.shape[0]
    kq = z_q.shape[0]

    # Construct helper arrays from z_q
    d_pos = torch.roll(z_q, shifts=-1)
    d_neg = torch.roll(z_q, shifts=1)

    # Clip z_p to be in new support range (vmin, vmax)
    z_p = torch.clip(z_p, z_q[0], z_q[-1]).unsqueeze(0)
    assert z_p.shape == torch.Size([1, kp])

    # Get the distance between atom values in support
    d_pos = (d_pos - z_q).unsqueeze(1)
    d_neg = (z_q - d_neg).unsqueeze(1)
    z_q = z_q.unsqueeze(1)
    assert z_q.shape == torch.Size([kq, 1])

    # Ensure that we do not divide by zero, in case of atoms of identical value
    d_neg = torch.where(d_neg > 0, 1.0 / d_neg, torch.zeros_like(d_neg))
    d_pos = torch.where(d_pos > 0, 1.0 / d_pos, torch.zeros_like(d_pos))

    delta_qp = z_p - z_q  # clip(z_p)[j] - z_q[i]
    d_sign = (delta_qp >= 0.0).float()
    assert delta_qp.shape == torch.Size([kq, kp])
    assert d_sign.shape == torch.Size([kq, kp])

    # Matrix of entries sgn(a_ij) * |a_ij|, with a_ij = clip(z_p)[j] - z_q[i]
    delta_hat = (d_sign * delta_qp * d_pos) - ((1.0 - d_sign) * delta_qp * d_neg)
    probs = probs.unsqueeze(0)
    assert delta_hat.shape == torch.Size([kq, kp])
    assert probs.shape == torch.Size([1, kp])

    # Projecting probabilities
    return torch.sum(torch.clip(1.0 - delta_hat, 0.0, 1.0) * probs, dim=-1)


def categorical_td_target(
    v_atoms: torch.Tensor,
    r_t: torch.Tensor,
    v_logits_t: torch.Tensor,
    discount_t: Union[float, torch.Tensor],
    use_log_scale: bool = True,
    stop_target_gradients: bool = True,
):
    """Implements TD-learning for categorical value distributions in PyTorch.

    This function assumes inputs are PyTorch tensors.

    Args:
        v_atoms: atoms of V distribution.
        r_t: reward at time t.
        discount_t: discount at time t.
        v_logits_t: logits of V distribution at time t.
        stop_target_gradients: bool indicating whether to apply stop gradient to targets.

    Returns:
        Categorical target distribution.
    """
    # Ensure inputs are tensors
    assert all(isinstance(x, torch.Tensor) for x in [v_atoms, r_t, v_logits_t])
    assert v_atoms.ndim == 1, "v_atoms must be a 1D tensor"
    logit_size = v_atoms.size(0)

    assert v_logits_t.ndim == 2, "v_logits_t must be a 2D tensor"
    assert v_logits_t.size(1) == logit_size, "v_logits_t must match v_atoms"

    batch_size = v_logits_t.size(0)
    assert r_t.shape == torch.Size([batch_size, 1]), "r_t must be a 2D tensor"

    v_atoms = v_atoms.unsqueeze(0).expand(batch_size, -1)

    if isinstance(discount_t, torch.Tensor):
        assert discount_t.shape == torch.Size([batch_size, 1])

    # Scale and shift time-t distribution atoms by discount and reward.
    if use_log_scale:
        target_z = symlog(r_t + discount_t * symexp(v_atoms))
    else:
        target_z = r_t + discount_t * v_atoms

    # Convert logits to distribution.
    v_t_probs = F.softmax(v_logits_t, dim=-1)

    # Project using the Cramer distance and maybe stop gradient flow to targets.
    td_target = torch.vmap(categorical_l2_project)(target_z, v_t_probs, v_atoms)
    assert td_target.shape == v_logits_t.shape

    if stop_target_gradients:
        td_target = td_target.detach()

    return td_target


class RunningScale:
    """Running trimmed scale estimator."""

    def __init__(self, tau, device):
        self.tau = tau
        self._value = torch.ones(1, dtype=torch.float32, device=device)
        self._percentiles = torch.tensor([5, 95], dtype=torch.float32, device=device)

    def state_dict(self):
        return {"value": self._value, "percentiles": self._percentiles}

    def load_state_dict(self, state_dict):
        self._value.data.copy_(state_dict["value"])
        self._percentiles.data.copy_(state_dict["percentiles"])

    @property
    def value(self):
        return self._value.cpu().item()

    def _percentile(self, x):
        x_dtype, x_shape = x.dtype, x.shape
        x = x.view(x.shape[0], -1)
        in_sorted, _ = torch.sort(x, dim=0)
        positions = self._percentiles * (x.shape[0] - 1) / 100
        floored = torch.floor(positions)
        ceiled = floored + 1
        ceiled[ceiled > x.shape[0] - 1] = x.shape[0] - 1
        weight_ceiled = positions - floored
        weight_floored = 1.0 - weight_ceiled
        d0 = in_sorted[floored.long(), :] * weight_floored[:, None]
        d1 = in_sorted[ceiled.long(), :] * weight_ceiled[:, None]
        return (d0 + d1).view(-1, *x_shape[1:]).type(x_dtype)

    def update(self, x):
        percentiles = self._percentile(x.detach())
        value = torch.clamp(percentiles[1] - percentiles[0], min=1.0)
        self._value.data.lerp_(value, self.tau)

    def __call__(self, x, update=False):
        if update:
            self.update(x)
        return x * (1 / self.value)

    def __repr__(self):
        return f"RunningScale(S: {self.value})"
