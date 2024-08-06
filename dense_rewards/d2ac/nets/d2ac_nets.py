import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal


class MlpLayer(nn.Module):
    """
    Simple MLP layer
    """

    def __init__(self, n_in, n_out):
        super().__init__()
        self.c_fc = nn.Linear(n_in, n_out, bias=False)
        self.c_ln = nn.LayerNorm(n_out)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.c_ln(x)
        x = self.activation(x)
        return x


class Mlp(nn.Module):
    """
    Simple MLP
    """

    def __init__(self, n_in, n_hidden, n_layers=2):
        super().__init__()
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_layers = n_layers

        self.layers = nn.ModuleList()
        self.layers.append(MlpLayer(n_in, n_hidden))
        for _ in range(n_layers - 1):
            self.layers.append(MlpLayer(n_hidden, n_hidden))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class MlpNet(nn.Module):
    def __init__(self, n_in, n_out, n_hidden, n_layers=2, init_last_layer_zero=True):
        super().__init__()
        self.mlp = Mlp(n_in, n_hidden, n_layers=n_layers)
        self.fc_out = nn.Linear(n_hidden, n_out, bias=False)
        if init_last_layer_zero:
            nn.init.zeros_(self.fc_out.weight)

    def forward(self, x):
        x = self.mlp(x)
        x = self.fc_out(x)
        return x


class EnsembleModel(nn.Module):
    """
    Ensemble model.
    """

    def __init__(self, models):
        super().__init__()
        assert len(models) > 0, "Ensemble requires at least one model"

        self._num_models = len(models)

        base_model = copy.deepcopy(models[0])
        base_model.to("meta")
        params, buffers = torch.func.stack_module_state(models)  # type: ignore

        self.params_keys = list(params.keys())
        self.params_list = nn.ParameterList([params[k] for k in self.params_keys])
        self.buffers = buffers  # type: ignore

        def call_single_model(model_params, model_buffers, data):
            return torch.func.functional_call(base_model, (model_params, model_buffers), (data,))  # type: ignore

        self.vmap = torch.vmap(call_single_model, (0, 0, None), randomness="different")

    def forward(self, data):
        params = dict(zip(self.params_keys, self.params_list))
        output = self.vmap(params, self.buffers, data)
        assert output.ndim == data.ndim + 1, "Ensemble output shape mismatch"
        return output


class EnsembleQfunc(nn.Module):
    def __init__(self, n_in, n_out, n_hidden, n_layers, n_models):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_models = n_models

        models = [
            MlpNet(n_in, n_out, n_hidden, n_layers=n_layers) for _ in range(n_models)
        ]
        self.q_ensemble = EnsembleModel(models)

    def forward(self, obs, act):
        q_inputs = torch.cat([obs, act], dim=-1)
        assert q_inputs.ndim == 2 and q_inputs.size(-1) == self.n_in
        q_ensemble_outputs = self.q_ensemble(q_inputs)
        assert q_ensemble_outputs.shape == torch.Size(
            [self.n_models, obs.size(0), self.n_out]
        )
        return q_ensemble_outputs


LOG_STD_MAX = 2
LOG_STD_MIN = -20


class SquashedGaussianMlpActor(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        n_hidden,
        n_layers,
        act_limit=1.0,
    ):
        super().__init__()
        self.net = Mlp(n_in, n_hidden, n_layers=n_layers)
        self.mu_layer = nn.Linear(n_hidden, n_out)
        self.logstd_layer = nn.Linear(n_hidden, n_out)
        self.act_limit = act_limit

    def forward(self, obs, deterministic=False):
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.logstd_layer(net_out)
        log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = torch.exp(log_std)

        pi_distribution = Normal(mu, std)
        if deterministic:
            pi_action = mu
        else:
            pi_action = pi_distribution.rsample()

        logp_pi = pi_distribution.log_prob(pi_action).sum(axis=-1)
        logp_pi -= (
            2 * (math.log(2) - pi_action - nn.functional.softplus(-2 * pi_action))
        ).sum(axis=-1)

        pi_action = torch.tanh(pi_action)
        pi_action = self.act_limit * pi_action

        return pi_action, logp_pi


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    trig_inputs = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(trig_inputs), torch.sin(trig_inputs)], dim=-1)
    if dim % 2 == 1:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


def get_train_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr_plus_one":
        weightings = snrs + 1
    elif weight_schedule == "edm":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated_snr":
        weightings = torch.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = torch.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings


def sample_sigma_indices(batch_size, num_sample_steps, device=None):
    if num_sample_steps > 1:
        # random sample from [0, num_sample_steps)
        sigma_indices = torch.randint(0, num_sample_steps, (batch_size,), device=device)
    else:
        sigma_indices = torch.zeros((batch_size,), device=device)
    return sigma_indices


def get_sigma_indices(batch_size, step_id, device=None):
    sigma_indices = torch.zeros((batch_size,), device=device) + step_id
    return sigma_indices


def get_snr(sigmas):
    return sigmas**-2


def get_rescaled_sigmas(sigmas):
    return 1000 * 0.25 * torch.log(sigmas + 1e-5)


class DiffusionMlpActor(nn.Module):
    def __init__(
        self,
        n_in,
        n_out,
        n_hidden,
        n_layers,
        n_time_embed: int = 0,
        act_limit: float = 1.0,
        sigma_data: float = 1.0,
        sigma_min: float = 0.1,
        sigma_max: float = 5.0,
        scalings: str = "uniform",
    ):
        super().__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_time_embed = n_time_embed
        self.act_limit = act_limit

        self.sigma_data: float = sigma_data
        self.sigma_min: float = sigma_min
        self.sigma_max: float = sigma_max
        self.scalings: str = scalings
        assert self.scalings in [
            "uniform",
            "edm",
        ], f"scalings {self.scalings} not implemented"
        self.rho: float = 7.0

        if self.scalings != "uniform":
            assert n_time_embed > 0, "time embedding required for non-uniform scalings"

        self.time_embed = None
        if self.n_time_embed > 0:
            self.time_embed = nn.Linear(n_hidden, n_time_embed)

        self.net = Mlp(n_in + n_time_embed, n_hidden, n_layers=n_layers)
        self.mu_delta_layer = nn.Linear(n_hidden, n_out)
        self.logstd_layer = nn.Linear(n_hidden, n_out)

        self.prior_mean = nn.Parameter(torch.zeros(1, n_out), requires_grad=False)
        self.prior_logstd = nn.Parameter(torch.zeros(1, n_out), requires_grad=False)

    def sample_from_prior(self, batch_size):
        mean = self.prior_mean.expand(batch_size, -1)
        std = self.prior_logstd.expand(batch_size, -1).exp() * self.sigma_max
        gaussian_dist = Normal(mean, std)
        prior_samples = gaussian_dist.rsample()
        logp_samples = gaussian_dist.log_prob(prior_samples).sum(axis=-1)
        return prior_samples, logp_samples

    def sample_sigma_indices(self, batch_size, num_sample_steps, device=None):
        return sample_sigma_indices(batch_size, num_sample_steps, device=device)

    def get_sigma_indices(self, batch_size, step_id, device=None):
        return get_sigma_indices(batch_size, step_id, device=device)

    def get_sigma_from_indices(self, indices, num_sample_steps):
        if num_sample_steps > 1:
            # indices in [0, num_sample_steps)
            # index = 0 -> sigma = sigma_max
            # index = num_sample_steps - 1 -> sigma = sigma_min
            sigma = self.sigma_max ** (1 / self.rho) + indices / (
                num_sample_steps - 1
            ) * (self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho))
            sigma = sigma**self.rho
        else:
            # resort to prior
            sigma = torch.zeros_like(indices) + self.sigma_max
        return sigma

    def get_noised_inputs(self, inputs, sigmas):
        assert inputs.ndim == 2 and sigmas.ndim == 1
        assert inputs.size(0) == sigmas.size(0)
        noise = torch.randn_like(inputs) * sigmas.unsqueeze(-1)
        return inputs + noise

    def _scalings_for_edm(self, sigmas: torch.Tensor):
        c_skip = self.sigma_data**2 / (sigmas**2 + self.sigma_data**2)
        c_out = sigmas * self.sigma_data / (sigmas**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigmas**2 + self.sigma_data**2) ** 0.5
        # ensure that scalings are 2-D tensors
        c_skip = c_skip.unsqueeze(-1)
        c_out = c_out.unsqueeze(-1)
        c_in = c_in.unsqueeze(-1)
        return c_skip, c_out, c_in

    def _scalings_uniform(self):
        c_skip = 1.0
        c_out = 1.0
        c_in = 1.0
        return c_skip, c_out, c_in

    def get_train_weightings(self, sigmas, weight_schedule="uniform"):
        return get_train_weightings(weight_schedule, get_snr(sigmas), self.sigma_data)

    def atanh(self, x):
        eps = 1e-5
        limit = 1 - eps
        x = torch.clamp(x / self.act_limit, -limit, limit)
        return torch.atanh(x)

    def tanh(self, x):
        pi_action = torch.tanh(x)
        pi_action = self.act_limit * pi_action
        return pi_action

    def forward(self, obs, mu_init, sigmas, sde=False, deterministic=False):
        c_skip, c_out, c_in = self._scalings_uniform()
        if self.scalings == "edm":
            c_skip, c_out, c_in = self._scalings_for_edm(sigmas)

        mu_inputs = c_in * mu_init
        pi_inputs = torch.cat([obs, mu_inputs], dim=-1)
        assert pi_inputs.ndim == 2 and pi_inputs.size(-1) == self.n_in

        if self.n_time_embed > 0:
            assert self.time_embed is not None
            sigmas_emb = self.time_embed(
                timestep_embedding(get_rescaled_sigmas(sigmas), self.n_hidden)
            )
            assert sigmas_emb.ndim == 2 and sigmas_emb.size(-1) == self.n_time_embed
            pi_inputs = torch.cat([pi_inputs, sigmas_emb], dim=-1)

        net_out = self.net(pi_inputs)
        mu_delta = self.mu_delta_layer(net_out)

        mu = c_skip * mu_init + c_out * mu_delta

        if sde:
            sampling_sigmas = (sigmas**2 - self.sigma_min**2) ** 0.5
            std = sampling_sigmas.unsqueeze(-1).expand(-1, self.n_out)
        else:
            log_std = self.logstd_layer(net_out)
            log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            std = torch.exp(log_std)

        gaussian_dist = Normal(mu, std)

        if deterministic:
            gaussian_action = mu
        else:
            gaussian_action = gaussian_dist.rsample()

        logp_gaussian_action = gaussian_dist.log_prob(gaussian_action).sum(axis=-1)
        logdet_tanh = -(
            2 * (math.log(2) - gaussian_action - F.softplus(-2 * gaussian_action))
        ).sum(axis=-1)

        logp_pi = logp_gaussian_action + logdet_tanh
        pi_action = self.tanh(gaussian_action)

        return (
            pi_action,
            logp_pi,
            gaussian_action,
            {
                "logdet_tanh": logdet_tanh,
                "logp_gaussian_action": logp_gaussian_action,
            },
        )
