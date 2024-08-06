import logging
import os
import os.path as osp
from typing import Optional

import torch


class ReplayBuffer:
    def __init__(self, obs_dim, act_dim, horizon, buffer_size):
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.horizon = horizon
        self.buffer_size = buffer_size

        assert self.buffer_size % self.horizon == 0
        self.n_total_traj = self.buffer_size // self.horizon

        print("Replay buffer size: ", self.buffer_size)
        print("Number of trajectories to fill: ", self.n_total_traj)
        print("Replay horizon: ", self.horizon)

        self.buffers = {
            "obs": torch.zeros(
                (self.n_total_traj, self.horizon + 1, self.obs_dim), dtype=torch.float32
            ),
            "act": torch.zeros(
                (self.n_total_traj, self.horizon, self.act_dim), dtype=torch.float32
            ),
            "reward": torch.zeros(
                (self.n_total_traj, self.horizon, 1), dtype=torch.float32
            ),
            "done": torch.zeros(
                (self.n_total_traj, self.horizon, 1), dtype=torch.float32
            ),
            "traj_length": -torch.ones((self.n_total_traj,)).long(),
            "current_traj_ptr": 0,
            "traj_buffer_full": False,
            "transition_traj_idx": -torch.ones((self.buffer_size,)).long(),
            "transition_time_idx": -torch.ones((self.buffer_size,)).long(),
            "current_transition_ptr": 0,
            "transition_buffer_full": False,
        }

        self.n_active_traj = 0
        self.active_traj_idx: Optional[torch.Tensor] = None
        self.traj_alive: Optional[torch.Tensor] = None

        self._dump_file_name = "replay"
        # Counter for rotating saves and loads
        self._current_copy = 0

    def reset(self):
        # Resetting internal states
        self.buffers["obs"].zero_()
        self.buffers["act"].zero_()
        self.buffers["reward"].zero_()
        self.buffers["done"].zero_()
        self.buffers["traj_length"].fill_(-1)
        self.buffers["current_traj_ptr"] = 0
        self.buffers["traj_buffer_full"] = False
        self.buffers["transition_traj_idx"].fill_(-1)
        self.buffers["transition_time_idx"].fill_(-1)
        self.buffers["current_transition_ptr"] = 0
        self.buffers["transition_buffer_full"] = False
        self.n_active_traj = 0
        self.active_traj_idx = None
        self.traj_alive = None

    @property
    def current_num_traj(self):
        if self.buffers["traj_buffer_full"]:
            return self.n_total_traj
        else:
            return self.buffers["current_traj_ptr"]

    def _get_trajectory_idx(self, batch_size):
        # function: get the indices of the new active trajectories
        current_ptr = self.buffers["current_traj_ptr"]
        assert current_ptr < self.n_total_traj
        idx = torch.arange(current_ptr, current_ptr + batch_size)
        idx = torch.remainder(idx, self.n_total_traj)
        assert idx.size(0) == batch_size
        return idx

    def _increment_traj_ptr(self, batch_size):
        self.buffers["current_traj_ptr"] += batch_size
        if self.buffers["current_traj_ptr"] >= self.n_total_traj:
            self.buffers["traj_buffer_full"] = True
            self.buffers["current_traj_ptr"] = (
                self.buffers["current_traj_ptr"] % self.n_total_traj
            )

    @property
    def current_num_transitions(self):
        if self.buffers["transition_buffer_full"]:
            return self.buffer_size
        else:
            return self.buffers["current_transition_ptr"]

    def _get_transition_idx(self, batch_size):
        # function: get the indices of the new transitions
        current_ptr = self.buffers["current_transition_ptr"]
        assert current_ptr < self.buffer_size
        idx = torch.arange(current_ptr, current_ptr + batch_size)
        idx = torch.remainder(idx, self.buffer_size)
        assert idx.size(0) == batch_size
        return idx

    def _increment_transition_ptr(self, batch_size):
        self.buffers["current_transition_ptr"] += batch_size
        if self.buffers["current_transition_ptr"] >= self.buffer_size:
            self.buffers["transition_buffer_full"] = True
            self.buffers["current_transition_ptr"] = (
                self.buffers["current_transition_ptr"] % self.buffer_size
            )

    def reset_episodes(self, init_obs, batch_size):
        device = self.buffers["obs"].device
        init_obs = init_obs.to(device=device)
        assert init_obs.size(0) == batch_size
        self.n_active_traj = batch_size
        # get the new active trajectory indices
        trajectory_idx = self._get_trajectory_idx(batch_size=batch_size)
        # store the initial observations
        self.buffers["obs"][trajectory_idx, 0] = init_obs
        self.buffers["traj_length"][trajectory_idx] = 0
        # increment the trajectory pointer
        self._increment_traj_ptr(batch_size=batch_size)
        # set the active trajectory indices
        self.active_traj_idx = trajectory_idx
        # set the alive mask
        self.traj_alive = torch.ones((batch_size,)).bool()

    def store(self, act, reward, next_obs, done):
        assert (
            act.size(0)
            == reward.size(0)
            == next_obs.size(0)
            == done.size(0)
            == self.active_traj_idx.size(0)
            == self.n_active_traj
        )
        if reward.ndim == 1:
            reward = reward.unsqueeze(-1)
        if done.ndim == 1:
            done = done.unsqueeze(-1)

        # filter out the done trajectories
        assert self.traj_alive.ndim == 1
        trajectory_idx = self.active_traj_idx[self.traj_alive]
        assert trajectory_idx.ndim == 1
        batch_size = trajectory_idx.size(0)

        # calculate the new alive mask before filtering out the done trajectories
        new_traj_alive = torch.logical_and(
            self.traj_alive, torch.logical_not(done.squeeze(-1).bool())
        )

        # filter out the done trajectories
        act = act[self.traj_alive]
        reward = reward[self.traj_alive]
        next_obs = next_obs[self.traj_alive]
        done = done[self.traj_alive]

        # store the transition
        ep_len_idx = self.buffers["traj_length"][trajectory_idx]
        self.buffers["act"][trajectory_idx, ep_len_idx] = act
        self.buffers["reward"][trajectory_idx, ep_len_idx] = reward
        self.buffers["obs"][trajectory_idx, ep_len_idx + 1] = next_obs
        self.buffers["done"][trajectory_idx, ep_len_idx] = done
        self.buffers["traj_length"][trajectory_idx] += 1

        # save transition indices
        transition_idx = self._get_transition_idx(batch_size=batch_size)
        self.buffers["transition_traj_idx"][transition_idx] = trajectory_idx
        self.buffers["transition_time_idx"][transition_idx] = ep_len_idx
        self._increment_transition_ptr(batch_size=batch_size)

        # update the alive mask
        self.traj_alive = new_traj_alive

    def sample_transitions(self, batch_size):
        transition_idx = torch.randint(
            low=0, high=self.current_num_transitions, size=(batch_size,)
        )
        traj_idx = self.buffers["transition_traj_idx"][transition_idx]
        time_idx = self.buffers["transition_time_idx"][transition_idx]
        batch = {
            "obs": self.buffers["obs"][traj_idx, time_idx],
            "obs2": self.buffers["obs"][traj_idx, time_idx + 1],
            "act": self.buffers["act"][traj_idx, time_idx],
            "reward": self.buffers["reward"][traj_idx, time_idx],
            "done": self.buffers["done"][traj_idx, time_idx],
        }
        return batch

    def sample_sequences(self, batch_size, seq_length):
        # function: sample a batch of trajectory sequences
        assert seq_length >= 1, "seq_length must be >= 1"
        transition_idx = torch.randint(
            low=0, high=self.current_num_transitions, size=(batch_size,)
        )
        # sample the trajectory indices
        trajectory_idx = self.buffers["transition_traj_idx"][transition_idx]

        # get the trajectory lengths
        traj_lengths = self.buffers["traj_length"][trajectory_idx]
        traj_lengths = traj_lengths.unsqueeze(-1)
        assert traj_lengths.shape == torch.Size([batch_size, 1])

        # sample the starting time indices
        start_points = self.buffers["transition_time_idx"][transition_idx]
        start_points = start_points.unsqueeze(-1)
        assert torch.all(start_points < traj_lengths)

        # the time indices per sequence
        seq_time_idx = start_points + torch.arange(seq_length).unsqueeze(0)
        assert seq_time_idx.shape == torch.Size([batch_size, seq_length])

        # get valid mask
        valid_mask = seq_time_idx < traj_lengths
        assert valid_mask.shape == torch.Size([batch_size, seq_length])

        # truncate the time indices
        seq_time_idx = torch.minimum(seq_time_idx, traj_lengths - 1).clamp(min=0)
        assert seq_time_idx.max() < self.horizon

        flat_time_indices = seq_time_idx.flatten()
        flat_traj_indices = trajectory_idx.unsqueeze(-1).repeat(1, seq_length).flatten()

        # get the batch of sequences
        batch = {
            "obs": self.buffers["obs"][flat_traj_indices, flat_time_indices],
            "obs2": self.buffers["obs"][flat_traj_indices, flat_time_indices + 1],
            "act": self.buffers["act"][flat_traj_indices, flat_time_indices],
            "reward": self.buffers["reward"][flat_traj_indices, flat_time_indices],
            "done": self.buffers["done"][flat_traj_indices, flat_time_indices],
        }
        for key, value in batch.items():
            batch[key] = value.reshape(batch_size, seq_length, -1)

        batch["valid_mask"] = valid_mask
        # for testing purposes
        batch["trajectory_idx"] = trajectory_idx
        batch["start_points"] = start_points
        return batch

    def save(self, path):
        # save to a temp file first, then rename to avoid corruption
        temp_pth = osp.join(path, f"{self._dump_file_name}_{self._current_copy + 1}.pt")
        model_pth = osp.join(path, f"{self._dump_file_name}_{self._current_copy}.pt")
        try:
            torch.save(self.buffers, temp_pth)
            os.replace(temp_pth, model_pth)
        except Exception as e:
            logging.error(f"Error saving model copy {self._current_copy}: {e}")

    def load(self, path):
        model_pth = osp.join(path, f"{self._dump_file_name}_{self._current_copy}.pt")
        if osp.exists(model_pth):
            try:
                replay_state = torch.load(model_pth)
                self.buffers = replay_state
                return True
            except RuntimeError:
                replay_state = torch.load(model_pth, map_location="cpu")
                self.buffers = replay_state
                return True
            except Exception as e:
                logging.error(
                    f"Error loading replay buffer copy {self._current_copy}: {e}"
                )
        logging.error("Loading failed; No replay buffer found.")
        return False
