import logging
import math
import os
import os.path as osp

import torch
import torch.nn as nn

from d2ac.nets import d2ac_nets as qzero_nets
from d2ac.utils import torch_utils, torchmath


class AgentD2AC(nn.Module):
    def __init__(
        self,
        obs_dim,
        act_dim,
        act_limit,
        args,
    ):
        super().__init__()

        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.act_limit = act_limit

        self.args = args
        assert (
            hasattr(self.args, "vmin")
            and hasattr(self.args, "vmax")
            and hasattr(self.args, "num_bins")
        ), "args must have vmin, vmax, num_bins"

        self.ensemble_size = args.n_models
        self.logit_size = args.num_bins

        v_atoms = torchmath.categorical_atoms(
            self.args.vmin, self.args.vmax, self.args.num_bins
        )
        self.v_atoms = nn.Parameter(v_atoms, requires_grad=False)

        pi_args = dict(
            n_in=self.obs_dim + self.act_dim,
            n_out=self.act_dim,
            n_hidden=args.hid,
            n_layers=args.n_layers,
            n_time_embed=self.args.n_time_embed,
            act_limit=self.act_limit,
            sigma_data=self.args.sigma_data,
            sigma_min=self.args.sigma_min,
            sigma_max=self.args.sigma_max,
            scalings=self.args.scalings,
        )

        self.pi = qzero_nets.DiffusionMlpActor(**pi_args)

        self.pi_target = None
        if self.args.consistency_model:
            self.pi_target = qzero_nets.DiffusionMlpActor(**pi_args)

        q_args = dict(
            n_in=self.obs_dim + self.act_dim,
            n_out=args.num_bins,
            n_hidden=args.hid,
            n_layers=args.n_layers,
            n_models=args.n_models,
        )

        self.criticQ = qzero_nets.EnsembleQfunc(**q_args)
        self.targetQ = qzero_nets.EnsembleQfunc(**q_args)

        self._init_target()

        self.log_alpha_parameter = nn.Parameter(
            torch.zeros(1) + math.log(args.alpha_init), requires_grad=True
        )

        self._dump_file_name = "agent"
        self._current_copy = 0

    def _init_target(self):
        torch_utils.copy_model_params_from_to(source=self.criticQ, target=self.targetQ)
        torch_utils.set_requires_grad(self.targetQ, allow_grad=False)

        if self.pi_target is not None:
            torch_utils.copy_model_params_from_to(source=self.pi, target=self.pi_target)
            torch_utils.set_requires_grad(self.pi_target, allow_grad=False)

    def freeze_critic(self):
        torch_utils.set_requires_grad(self.criticQ, allow_grad=False)

    def unfreeze_critic(self):
        torch_utils.set_requires_grad(self.criticQ, allow_grad=True)

    def reset(self):
        pass

    @staticmethod
    def _convert_tensor(x):
        x = torch_utils.to_tensor(x)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        return x

    @torch.no_grad()
    def get_action(
        self, obs, deterministic=False, n_sampling_steps=1, check_dtype=True
    ):
        if check_dtype:
            obs = self._convert_tensor(obs)

        batch_size = obs.size(0)
        device = obs.device
        assert obs.ndim == 2 and obs.size(-1) == self.obs_dim
        action_prior, _ = self.pi.sample_from_prior(obs.size(0))
        pi_action = action_prior

        for step_id in range(n_sampling_steps):
            sigma_indices = self.pi.get_sigma_indices(
                batch_size=batch_size, step_id=step_id, device=device
            )
            sigmas = self.pi.get_sigma_from_indices(
                indices=sigma_indices, num_sample_steps=n_sampling_steps
            )
            if self.args.action_sampling_mode == "sde":
                sde = bool(step_id < n_sampling_steps - 1)
            else:
                sde = False
            # pi returns: pi_action, logp_pi, gaussian_action, info
            pi_action, _, gaussian_action, _ = self.pi(
                obs,
                action_prior,
                sigmas=sigmas,
                sde=sde,
                deterministic=deterministic,
            )
            action_prior = gaussian_action

        return pi_action

    def get_actions(
        self,
        obs,
        action_prior,
        sigmas,
        target=False,
        sde=False,
        deterministic=False,
        check_dtype=True,
    ):
        if check_dtype:
            obs = self._convert_tensor(obs)

        pi_net = self.pi
        if target:
            assert self.pi_target is not None, "target network not initialized"
            pi_net = self.pi_target

        pi_action, logp_pi, gaussian_action, info = pi_net(
            obs,
            action_prior,
            sigmas=sigmas,
            sde=sde,
            deterministic=deterministic,
        )
        return pi_action, logp_pi, gaussian_action, info

    def get_critic_logits(self, obs, act, target=False, check_dtype=True):
        if check_dtype:
            obs = self._convert_tensor(obs)
            act = self._convert_tensor(act)

        batch_size = obs.size(0)
        q_net = self.targetQ if target else self.criticQ
        q_ensemble_outputs = q_net(obs, act)
        assert q_ensemble_outputs.shape == torch.Size(
            [self.ensemble_size, batch_size, self.logit_size]
        )
        return q_ensemble_outputs

    def forward(self, obs, target=False, n_sampling_steps=1, check_dtype=True):
        if check_dtype:
            obs = self._convert_tensor(obs)

        batch_size = obs.size(0)
        device = obs.device
        assert obs.ndim == 2 and obs.size(-1) == self.obs_dim
        action_prior, _ = self.pi.sample_from_prior(obs.size(0))
        pi_action, gaussian_action, info = action_prior, action_prior, {}

        for step_id in range(n_sampling_steps):
            sigma_indices = self.pi.get_sigma_indices(
                batch_size=batch_size, step_id=step_id, device=device
            )
            sigmas = self.pi.get_sigma_from_indices(
                indices=sigma_indices, num_sample_steps=n_sampling_steps
            )
            if self.args.action_sampling_mode == "sde":
                sde = bool(step_id < n_sampling_steps - 1)
            else:
                sde = False
            pi_action, _, gaussian_action, info = self.get_actions(
                obs,
                action_prior,
                sigmas=sigmas,
                sde=sde,
                deterministic=False,
                check_dtype=False,
            )
            action_prior = gaussian_action

        critic_logits = self.get_critic_logits(
            obs, pi_action, target=target, check_dtype=False
        )
        return critic_logits, pi_action, gaussian_action

    def save(self, path):
        # save to a temp file first, then rename to avoid corruption
        temp_pth = osp.join(path, f"{self._dump_file_name}_{self._current_copy + 1}.pt")
        model_pth = osp.join(path, f"{self._dump_file_name}_{self._current_copy}.pt")
        try:
            torch.save(self.state_dict(), temp_pth)
            os.replace(temp_pth, model_pth)
        except Exception as e:
            logging.error(f"Error saving model copy {self._current_copy}: {e}")

    def load(self, path):
        agent_state = None
        model_pth = osp.join(path, f"{self._dump_file_name}_{self._current_copy}.pt")
        if osp.exists(model_pth):
            try:
                agent_state = torch.load(model_pth)
                self.load_state_dict(agent_state)
                return True
            except RuntimeError:
                agent_state = torch.load(model_pth, map_location="cpu")
                self.load_state_dict(agent_state)
                return True
            except Exception as e:
                logging.error(f"Error loading model copy {self._current_copy}: {e}")
        logging.error("Agent Loading failed; No valid model file found.")
        return False
