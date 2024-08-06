import logging
import os
import os.path as osp

import torch
import torch.nn.functional as F
from torch.optim import Adam, AdamW

from d2ac.utils import torch_utils, torchmath


class Trainer:
    def __init__(
        self,
        agent,
        monitor,
        args,
        q_lr=1e-3,
        pi_lr=1e-3,
        alpha_lr=3e-4,
    ):
        self.agent = agent
        self.args = args
        self.monitor = monitor

        self.gamma = self.args.gamma
        self._polyak = self.args.polyak
        self._polyak_pi = self.args.polyak_pi
        self._targ_entropy_coef = self.args.targ_entropy_coef

        self._logscale = bool(self.args.discrete_mode == "log")
        print("Trainer:: using logscale:", self._logscale)

        self.n_sampling_steps_train = self.args.n_sampling_steps_train
        self.n_sampling_steps_inference = self.args.n_sampling_steps_inference

        self._alpha_optimizer = Adam([agent.log_alpha_parameter], lr=alpha_lr)

        self._q_optimizer = (
            Adam(agent.criticQ.parameters(), lr=q_lr)
            if not args.adamw
            else AdamW(
                agent.criticQ.parameters(), lr=q_lr, weight_decay=args.weight_decay
            )
        )
        self._pi_optimizer = (
            Adam(agent.pi.parameters(), lr=pi_lr)
            if not args.adamw
            else AdamW(agent.pi.parameters(), lr=pi_lr, weight_decay=args.weight_decay)
        )

        self._dump_file_name = "trainer"
        self._current_copy = 0

    def two_hot(self, values):
        return torchmath.two_hot(values, self.args, apply_symlog=self._logscale)

    def two_hot_inv(self, logits):
        return torchmath.two_hot_inv(logits, self.args, apply_symexp=self._logscale)

    def get_values_from_categoricals(self, categoricals):
        return torchmath.get_values_from_categoricals(
            categoricals, self.agent.v_atoms, apply_symexp=self._logscale
        )

    def categorical_td_target(self, reward, next_q_logits, discount_t):
        return torchmath.categorical_td_target(
            self.agent.v_atoms,
            reward,
            next_q_logits,
            discount_t=discount_t,
            use_log_scale=self._logscale,
        )

    @property
    def _alpha(self):
        return torch.exp(self.agent.log_alpha_parameter).item()

    @staticmethod
    def _convert_batch(batch):
        assert all(key in batch for key in ["obs", "act", "obs2", "reward", "done"])
        batch = torch_utils.dict_to_tensor(batch)
        return batch

    def td_backup_two_hot(self, q_logits, reward, done, next_q_logits):
        with torch.no_grad():
            next_q_values = self.two_hot_inv(next_q_logits)
            next_qs = torch.min(next_q_values, dim=0).values
            assert next_qs.shape == reward.shape

            # calculate backup values
            v_obs2 = self.gamma * next_qs
            backup_values = reward + (1.0 - done) * v_obs2
            q_ensemble_values = backup_values.unsqueeze(0).expand(
                self.args.n_models, -1, -1
            )
            q_ensemble_categoricals = self.two_hot(q_ensemble_values)

            # calculate L2 loss for diagnostics
            q_values = self.two_hot_inv(q_logits)
            q_l2_loss = F.mse_loss(q_values, q_ensemble_values)

        # calculate q losses
        log_probs_q = F.log_softmax(q_logits, dim=-1)
        q_losses = -(q_ensemble_categoricals * log_probs_q).sum(-1)
        loss_q = q_losses.mean()

        with torch.no_grad():
            probs_q = log_probs_q.exp()
            probs_q_max = probs_q.max(dim=-1).values
            q_entropies = -(probs_q * log_probs_q).sum(-1)
            entropy_mean = q_entropies.mean()

        self.monitor.store(
            loss_q=loss_q.item(),
            backup_values=backup_values.mean().item(),
            q_l2_loss=q_l2_loss.item(),
            q_values=q_values.mean().item(),
            reward=reward.mean().item(),
            entropy_mean=entropy_mean.item(),
            probs_q_max=probs_q_max.mean().item(),
        )
        return loss_q

    def td_backup_distributional(self, q_logits, reward, done, next_q_logits):
        assert next_q_logits.size(-1) == self.args.num_bins
        batch_size = q_logits.size(1)

        with torch.no_grad():
            next_q_values = self.two_hot_inv(next_q_logits)
            next_qs, indices = torch.min(next_q_values, dim=0)
            assert next_qs.shape == reward.shape
            indices = indices.expand(-1, self.args.num_bins).unsqueeze(0)
            assert indices.shape == torch.Size([1, batch_size, self.args.num_bins])

            min_next_q_logits = torch.gather(next_q_logits, 0, indices).squeeze(0)
            backup_target = self.categorical_td_target(
                reward, min_next_q_logits, self.gamma * (1 - done)
            )
            assert backup_target.shape == torch.Size([batch_size, self.args.num_bins])
            backup_values = self.get_values_from_categoricals(backup_target)
            q_ensemble_categoricals = backup_target.unsqueeze(0).expand(
                self.args.n_models, -1, -1
            )

            # calculate L2 loss for diagnostics
            q_ensemble_values = backup_values.unsqueeze(0).expand(
                self.args.n_models, -1, -1
            )
            q_values = self.two_hot_inv(q_logits)
            q_l2_loss = F.mse_loss(q_values, q_ensemble_values)

        # calculate q losses
        log_probs_q = F.log_softmax(q_logits, dim=-1)
        q_losses = -(q_ensemble_categoricals * log_probs_q).sum(-1)
        loss_q = q_losses.mean()

        probs_q = log_probs_q.exp()
        probs_q_max = probs_q.max(dim=-1).values
        q_entropies = -(probs_q * log_probs_q).sum(-1)
        entropy_mean = q_entropies.mean()

        q_entropy_loss = self.args.q_entropy_loss_coef * (-entropy_mean)
        if self.args.q_entropy_loss_coef > 0:
            loss_q = loss_q + q_entropy_loss

        logit_Z = torch.logsumexp(q_logits, dim=-1, keepdim=False)
        assert logit_Z.shape == torch.Size([self.args.n_models, batch_size])
        logit_Z_square = logit_Z**2
        logit_z_loss = self.args.z_loss_coef * logit_Z_square.mean()

        if self.args.z_loss_coef > 0:
            loss_q = loss_q + logit_z_loss

        self.monitor.store(
            loss_q=loss_q.item(),
            backup_values=backup_values.mean().item(),
            q_l2_loss=q_l2_loss.item(),
            q_values=q_values.mean().item(),
            reward=reward.mean().item(),
            entropy_mean=entropy_mean.item(),
            q_entropy_loss=q_entropy_loss.item(),
            probs_q_max=probs_q_max.mean().item(),
            q_logits_max=q_logits.max().item(),
            q_logits_min=q_logits.min().item(),
            q_logits_mean=q_logits.mean().item(),
            logit_z_loss=logit_z_loss.item(),
            logit_Z=logit_Z.mean().item(),
            logit_Z_square=logit_Z_square.mean().item(),
        )
        return loss_q

    def get_dpg_gaussian_action_target(self, pi_observations, gaussian_actions):
        # dpg gradients
        self.agent.freeze_critic()
        # define var in atanh space
        action_var = torch.autograd.Variable(
            gaussian_actions.detach(), requires_grad=True
        )
        # convert to tanh space
        action_var_tanh = self.agent.pi.tanh(action_var)
        # critics only see actions in tanh space
        q_pi_logits = self.agent.get_critic_logits(
            pi_observations, action_var_tanh, check_dtype=False
        )
        action_values = self.two_hot_inv(q_pi_logits)
        min_action_values = torch.min(action_values, dim=0).values
        dpg_objective = min_action_values.sum()
        dpg_objective.backward()
        self.agent.unfreeze_critic()

        action_grads = action_var.grad
        action_grads_norm = torch.norm(action_grads, dim=-1)
        self.monitor.store(
            min_action_values=min_action_values.mean().item(),
            action_grads_norm=action_grads_norm.mean().item(),
        )

        # the action targets are defined in the atanh space to supervise the gaussian
        action_targets = (action_var + action_grads).detach()
        assert action_targets.shape == gaussian_actions.shape
        return action_targets

    def get_sigmas(self, sigma_indices, num_sample_steps):
        sigmas = self.agent.pi.get_sigma_from_indices(
            sigma_indices, num_sample_steps=num_sample_steps
        )
        return sigmas

    def get_noised_inputs(self, inputs, sigmas):
        noised_inputs = self.agent.pi.get_noised_inputs(inputs.detach(), sigmas)
        return noised_inputs

    def get_train_weights(self, sigmas):
        train_weights = self.agent.pi.get_train_weightings(
            sigmas, weight_schedule=self.args.weight_schedule
        )
        return train_weights

    def cm_loss(self, obs, obs2, gaussian_act_next):
        batch_size = obs.size(0)
        num_sample_steps = self.n_sampling_steps_train

        # handle the one-step case; no bellman backup needed
        indices_one_step = self.agent.pi.get_sigma_indices(
            batch_size, step_id=num_sample_steps - 1, device=torch_utils.device
        )
        # last step id is (num_sample_steps - 1) because id starts from 0
        sigmas_one_step = self.get_sigmas(indices_one_step, num_sample_steps)
        weights_one_step = self.get_train_weights(sigmas_one_step)
        _, logp_pi, gaussian_action_one_step, _ = self.agent.get_actions(
            obs=obs2,
            action_prior=self.get_noised_inputs(gaussian_act_next, sigmas_one_step),
            sigmas=sigmas_one_step,
            target=False,
            sde=False,
            deterministic=False,
            check_dtype=False,
        )
        target_one_step = self.get_dpg_gaussian_action_target(
            obs2, gaussian_action_one_step.detach()
        )

        # handle the multi-step case; target provided by bellman backup
        indices_multi_step = self.agent.pi.sample_sigma_indices(
            batch_size, num_sample_steps=num_sample_steps - 1, device=torch_utils.device
        )
        # last step id is (num_sample_steps - 2) because we need (indices + 1) later
        sigmas_multi_step = self.get_sigmas(indices_multi_step, num_sample_steps)
        weights_multi_step = self.get_train_weights(sigmas_multi_step)
        _, _, gaussian_action_multi_step, _ = self.agent.get_actions(
            obs=obs2,
            action_prior=self.get_noised_inputs(gaussian_act_next, sigmas_one_step),
            sigmas=sigmas_one_step,
            target=False,
            sde=True,
            deterministic=True,
            check_dtype=False,
        )
        with torch.no_grad():
            sigmas_step_plus_one = self.get_sigmas(
                indices_multi_step + 1, num_sample_steps
            )
            _, _, target_multi_step, _ = self.agent.get_actions(
                obs=obs2,
                action_prior=self.get_noised_inputs(
                    gaussian_act_next, sigmas_step_plus_one
                ),
                sigmas=sigmas_step_plus_one,
                target=True,
                sde=True,
                deterministic=True,
                check_dtype=False,
            )

        weights = torch.cat([weights_one_step, weights_multi_step], dim=0)
        assert weights.shape == torch.Size([2 * batch_size])
        gaussian_action = torch.cat(
            [gaussian_action_one_step, gaussian_action_multi_step], dim=0
        )
        action_targets = torch.cat([target_one_step, target_multi_step], dim=0)
        loss_action_value = (
            weights * (0.5 * (gaussian_action - action_targets).pow(2)).sum(dim=-1)
        ).mean()
        loss_pi = loss_action_value + self._alpha * logp_pi.mean()
        return loss_pi, logp_pi

    def edm_loss(self, obs, act, obs2, gaussian_act_next):
        batch_size = obs.size(0)
        # deal with actions in atanh space (since prior is also in atanh space)
        act_atanh = self.agent.pi.atanh(act)

        # first compute the noised actions for obs under n_sampling_steps_train
        indices_train = self.agent.pi.sample_sigma_indices(
            batch_size,
            num_sample_steps=self.n_sampling_steps_train,
            device=torch_utils.device,
        )
        sigmas_train = self.get_sigmas(indices_train, self.n_sampling_steps_train)
        weights_train = self.get_train_weights(sigmas_train)
        obs_act_noised = self.get_noised_inputs(act_atanh, sigmas_train)

        # then compute the noised actions for obs2 under n_sampling_steps_inference
        indices_inference = self.agent.pi.sample_sigma_indices(
            batch_size,
            num_sample_steps=self.n_sampling_steps_inference,
            device=torch_utils.device,
        )
        sigmas_inference = self.get_sigmas(
            indices_inference, self.n_sampling_steps_inference
        )
        weights_inference = self.get_train_weights(sigmas_inference)
        obs_act_next_noised = self.get_noised_inputs(
            gaussian_act_next, sigmas_inference
        )

        # combine obs and obs2
        pi_obs = torch.cat([obs, obs2], dim=0)
        noised_gaussian_act = torch.cat([obs_act_noised, obs_act_next_noised], dim=0)
        sigmas = torch.cat([sigmas_train, sigmas_inference], dim=0)

        # compute the action targets
        _, logp_pi, gaussian_action, _ = self.agent.get_actions(
            pi_obs,
            noised_gaussian_act,
            sigmas=sigmas,
            target=False,
            sde=False,
            deterministic=False,
            check_dtype=False,
        )
        action_targets = self.get_dpg_gaussian_action_target(
            pi_obs, gaussian_action.detach()
        )

        weights = torch.cat([weights_train, weights_inference], dim=0)
        assert weights.shape == torch.Size([2 * batch_size])
        loss_action_value = (
            weights * (0.5 * (gaussian_action - action_targets).pow(2)).sum(dim=-1)
        ).mean()
        loss_pi = loss_action_value + self._alpha * logp_pi.mean()

        self.monitor.store(
            denoise_weights=weights.mean().item(),
            denoise_weights_max=weights.max().item(),
            denoise_weights_min=weights.min().item(),
            loss_action_value=loss_action_value.item(),
            sigmas=sigmas.mean().item(),
            sigmas_max=sigmas.max().item(),
            sigmas_min=sigmas.min().item(),
            indices_train_max=indices_train.max().item(),
            indices_train_min=indices_train.min().item(),
            indices_inference_max=indices_inference.max().item(),
            indices_inference_min=indices_inference.min().item(),
        )
        return loss_pi, logp_pi

    def train_step(self, data):
        data = self._convert_batch(data)
        obs = data["obs"]
        obs2 = data["obs2"]
        act = data["act"]
        batch_size = obs.size(0)
        reward, done = data["reward"], data["done"]
        assert reward.shape == (batch_size, 1)
        assert done.shape == (batch_size, 1)

        q_logits = self.agent.get_critic_logits(
            obs=obs, act=act, target=False, check_dtype=False
        )

        with torch.no_grad():
            next_q_logits, _, gaussian_act_next = self.agent(
                obs=obs2,
                target=True,
                n_sampling_steps=self.n_sampling_steps_inference,
                check_dtype=False,
            )

        if self.args.backup_method == "two_hot":
            loss_q = self.td_backup_two_hot(
                q_logits=q_logits,
                reward=reward,
                done=done,
                next_q_logits=next_q_logits,
            )
        elif self.args.backup_method == "distributional":
            loss_q = self.td_backup_distributional(
                q_logits=q_logits,
                reward=reward,
                done=done,
                next_q_logits=next_q_logits,
            )
        else:
            raise NotImplementedError

        self._q_optimizer.zero_grad()
        loss_q.backward()
        self._q_optimizer.step()

        for _ in range(self.args.pi_iters):
            if self.args.consistency_model:
                loss_pi, logp_pi = self.cm_loss(obs, obs2, gaussian_act_next)
            else:
                loss_pi, logp_pi = self.edm_loss(obs, act, obs2, gaussian_act_next)

            # log loss_pi
            self.monitor.store(loss_pi=loss_pi.item(), logp_pi=logp_pi.mean().item())
            self._pi_optimizer.zero_grad()
            loss_pi.backward()
            self._pi_optimizer.step()
            target_entropy = -self._targ_entropy_coef * self.agent.act_dim
            entropy_difference = -logp_pi - target_entropy
            loss_alpha = (
                torch.exp(self.agent.log_alpha_parameter) * entropy_difference.detach()
            ).mean()
            self._alpha_optimizer.zero_grad()
            loss_alpha.backward()
            self._alpha_optimizer.step()
            self.monitor.store(
                loss_alpha=loss_alpha.item(),
                alpha_value=self._alpha,
                target_entropy=target_entropy,
                entropy_difference=entropy_difference.mean().item(),
            )

        # update target networks
        torch_utils.target_soft_update(
            base_net=self.agent.criticQ,
            targ_net=self.agent.targetQ,
            polyak=self._polyak,
        )
        if self.agent.pi_target is not None:
            torch_utils.target_soft_update(
                base_net=self.agent.pi,
                targ_net=self.agent.pi_target,
                polyak=self._polyak_pi,
            )

    def state_dict(self):
        return {
            "q_optimizer": self._q_optimizer.state_dict(),
            "pi_optimizer": self._pi_optimizer.state_dict(),
            "alpha_optimizer": self._alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state_dict):
        self._q_optimizer.load_state_dict(state_dict["q_optimizer"])
        self._pi_optimizer.load_state_dict(state_dict["pi_optimizer"])
        self._alpha_optimizer.load_state_dict(state_dict["alpha_optimizer"])

    def save(self, path):
        trainer_state = self.state_dict()
        temp_pth = osp.join(path, f"{self._dump_file_name}_{self._current_copy + 1}.pt")
        model_pth = osp.join(path, f"{self._dump_file_name}_{self._current_copy}.pt")
        try:
            torch.save(trainer_state, temp_pth)
            os.replace(temp_pth, model_pth)
        except Exception as e:
            logging.error(f"Error saving trainer state copy {self._current_copy}: {e}")

    def load(self, path):
        trainer_state = None
        model_pth = osp.join(path, f"{self._dump_file_name}_{self._current_copy}.pt")
        if osp.exists(model_pth):
            try:
                trainer_state = torch.load(model_pth)
                self.load_state_dict(trainer_state)
                return True
            except RuntimeError:
                trainer_state = torch.load(model_pth, map_location="cpu")
                self.load_state_dict(trainer_state)
                return True
            except Exception as e:
                logging.error(
                    f"Error loading trainer state copy {self._current_copy}: {e}"
                )
        logging.error("Trainer Loading failed; No valid trainer state file found.")
        return False
