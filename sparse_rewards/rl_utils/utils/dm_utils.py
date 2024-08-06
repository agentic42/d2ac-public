from types import SimpleNamespace

import numpy as np
import torch
from dm_control import suite

from rl_utils.utils import torch_utils
from rl_utils.utils.vec_env import SubprocVecEnv


def get_dm_env(domain_name, task_name):
    assert (
        domain_name,
        task_name,
    ) in suite.ALL_TASKS, f"({domain_name}, {task_name}) not in {suite.ALL_TASKS}"
    env = suite.load(domain_name=domain_name, task_name=task_name)
    return env


class DMControlEnv:
    def __init__(self, domain_name, task_name):
        self._env = get_dm_env(domain_name, task_name)

        self._observation_spec = self._env.observation_spec()
        self._observation_dim = sum(
            [np.prod(v.shape, dtype=int) for v in self._observation_spec.values()]
        )

        self._action_spec = self._env.action_spec()
        self._action_minimum = self._action_spec.minimum
        self._action_maximum = self._action_spec.maximum
        self._action_range = self._action_maximum - self._action_minimum
        self._action_dim = self._action_spec.shape[0]

    def _process_observation(self, observation):
        # flatten observation
        obs = np.concatenate([v.flatten() for v in observation.values()])
        return obs

    def _process_action(self, action):
        # clip action to [-1, 1]
        action = np.clip(action, -1.0, 1.0)
        # scale action according to action space
        action = self._action_minimum + (action + 1.0) * 0.5 * self._action_range
        return action

    def reset(self):
        obs = self._env.reset().observation
        obs = self._process_observation(obs)
        return obs

    def step(self, action):
        action = self._process_action(action)
        time_step = self._env.step(action)
        obs = time_step.observation
        obs = self._process_observation(obs)
        reward = time_step.reward
        done = time_step.last() or time_step.discount == 0
        info = {}
        return obs, reward, done, info

    @property
    def env(self):
        return self._env

    def close(self):
        self._env.close()

    @property
    def action_dim(self):
        return self._action_dim

    @property
    def observation_dim(self):
        return self._observation_dim

    @property
    def action_space(self):
        return self._action_spec

    @property
    def observation_space(self):
        return self._observation_spec

    @property
    def _max_episode_steps(self):
        return self._env._step_limit


class TorchActionSpace:
    def __init__(self, num_envs, action_dim):
        self._num_envs = num_envs
        self._action_dim = action_dim

    @property
    def shape(self):
        return (self._num_envs, self._action_dim)

    @property
    def high(self):
        return np.ones(self.shape)

    @property
    def low(self):
        return -np.ones(self.shape)

    def sample(self):
        return (torch.rand(self._num_envs, self._action_dim) * 2 - 1).to(
            device=torch_utils.device
        )


class TorchDMControlEnv:
    def __init__(self, domain_name, task_name, num_envs=1):
        self._domain_name = domain_name
        self._task_name = task_name
        self._num_envs = num_envs

        env_fns = [
            lambda: DMControlEnv(domain_name, task_name) for _ in range(num_envs)
        ]
        print(f"creating SubprocVecEnv with {num_envs} envs ...")
        self._env = SubprocVecEnv(env_fns)

        self.action_space = TorchActionSpace(num_envs, self.action_dim)

    def reset(self):
        obs = self._env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32).to(device=torch_utils.device)
        return obs

    def step(self, action):
        action = torch_utils.to_numpy(action)
        obs, reward, done, info = self._env.step(action)
        obs = torch.as_tensor(obs, dtype=torch.float32).to(device=torch_utils.device)
        reward = torch.as_tensor(reward, dtype=torch.float32).to(
            device=torch_utils.device
        )
        done = torch.as_tensor(done, dtype=torch.float32).to(device=torch_utils.device)
        return obs, reward, done, info

    @property
    def base_env(self):
        return self._env.base_env

    @property
    def action_dim(self):
        return self.base_env.action_dim

    @property
    def observation_dim(self):
        return self.base_env.observation_dim

    @property
    def observation_space(self):
        return SimpleNamespace(
            shape=(self._num_envs, self.observation_dim),
        )
