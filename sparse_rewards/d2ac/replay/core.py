import logging
import os
import os.path as osp
import threading

import numpy as np
import torch


def sample_her_transitions(buffer, reward_func, batch_size, future_p=1.0):
    assert all(k in buffer for k in ["ob", "ag", "bg", "a", "ag_changed"])
    buffer["o2"] = buffer["ob"][:, 1:, :]
    buffer["ag2"] = buffer["ag"][:, 1:, :]
    buffer["a2"] = buffer["a"][:, 1:, :]

    n_trajs = buffer["a"].shape[0]
    horizon = buffer["bg"].shape[1]
    assert n_trajs > 0, "Buffer is empty"
    ep_idxes = np.random.randint(0, n_trajs, size=batch_size)
    t_samples = np.random.randint(0, horizon, size=batch_size)
    batch = {key: buffer[key][ep_idxes, t_samples].copy() for key in buffer.keys()}

    her_indexes = np.where(np.random.uniform(size=batch_size) < future_p)

    # note that seq length of states and goals is (horizon + 1)
    future_offset = 1 + (
        np.random.uniform(size=batch_size) * (horizon - t_samples)
    ).astype(int)
    future_t = t_samples + future_offset

    pos_offset = (np.random.uniform(size=batch_size) * (horizon - future_t)).astype(
        int
    ) + future_offset
    neg_offset = (np.random.uniform(size=batch_size) * (future_offset - 1)).astype(int)
    batch["pos_offset"] = pos_offset.copy()
    batch["neg_offset"] = neg_offset.copy()

    batch["bg"][her_indexes] = buffer["ag"][
        ep_idxes[her_indexes], future_t[her_indexes]
    ]
    batch["future_ob"] = buffer["ob"][ep_idxes, future_t].copy()
    batch["future_ag"] = buffer["ag"][ep_idxes, future_t].copy()
    batch["offset"] = future_offset.copy()
    batch["r"] = reward_func(batch["ag2"], batch["bg"], None)

    assert all(batch[k].shape[0] == batch_size for k in batch.keys())
    assert all(
        k in batch
        for k in ["ob", "ag", "a2", "bg", "a", "o2", "ag2", "r", "future_ag", "offset"]
    )
    return batch


class Replay:
    def __init__(self, env_params, args, reward_func):
        self.env_params = env_params
        self.args = args
        self.reward_func = reward_func

        self.horizon = env_params["max_timesteps"]
        self.size = args.replay_size // self.horizon

        self.current_size = 0
        self.n_transitions_stored = 0

        self.buffers = dict(
            ob=np.zeros((self.size, self.horizon + 1, self.env_params["obs"])),
            ag=np.zeros((self.size, self.horizon + 1, self.env_params["goal"])),
            bg=np.zeros((self.size, self.horizon, self.env_params["goal"])),
            a=np.zeros((self.size, self.horizon + 1, self.env_params["action"])),
            ag_changed=np.zeros((self.size, self.horizon + 1)),
        )

        self.lock = threading.Lock()

        self._dump_file_name = "replay"
        # Counter for rotating saves and loads
        self._current_copy = 0

    def store(self, episodes):
        ob_list, ag_list, bg_list, a_list = (
            episodes["ob"],
            episodes["ag"],
            episodes["bg"],
            episodes["a"],
        )
        ag_changed_list = episodes["ag_changed"]
        batch_size = ob_list.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(batch_size=batch_size)
            self.buffers["ob"][idxs] = ob_list.copy()
            self.buffers["ag"][idxs] = ag_list.copy()
            self.buffers["bg"][idxs] = bg_list.copy()
            self.buffers["a"][idxs] = a_list.copy()
            self.buffers["ag_changed"][idxs] = ag_changed_list.copy()
            self.n_transitions_stored += self.horizon * batch_size

    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][: self.current_size]
        transitions = sample_her_transitions(
            temp_buffers, self.reward_func, batch_size, future_p=self.args.future_p
        )
        return transitions

    def _get_storage_idx(self, batch_size):
        if self.current_size + batch_size <= self.size:
            idx = np.arange(self.current_size, self.current_size + batch_size)
        elif self.current_size < self.size:
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, batch_size - len(idx_a))
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, batch_size)
        self.current_size = min(self.size, self.current_size + batch_size)
        if batch_size == 1:
            idx = idx[0]
        return idx

    def state_dict(self):
        return dict(
            current_size=self.current_size,
            n_transitions_stored=self.n_transitions_stored,
            buffers=self.buffers,
        )

    def load_state_dict(self, state_dict):
        self.current_size = state_dict["current_size"]
        self.n_transitions_stored = state_dict["n_transitions_stored"]
        self.buffers = state_dict["buffers"]

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
        model_pth = osp.join(path, f"{self._dump_file_name}_{self._current_copy}.pt")
        if osp.exists(model_pth):
            try:
                replay_state = torch.load(model_pth)
                self.load_state_dict(replay_state)
                print("loading replay: current_size =", self.current_size)
                print(
                    "loading replay: n_transitions_stored =", self.n_transitions_stored
                )
                return True
            except RuntimeError:
                replay_state = torch.load(model_pth, map_location="cpu")
                self.load_state_dict(replay_state)
                print("loading replay: current_size =", self.current_size)
                print(
                    "loading replay: n_transitions_stored =", self.n_transitions_stored
                )
                return True
            except Exception as e:
                logging.error(
                    f"Error loading replay buffer copy {self._current_copy}: {e}"
                )
        logging.error("Loading failed; No replay buffer found.")
        return False
