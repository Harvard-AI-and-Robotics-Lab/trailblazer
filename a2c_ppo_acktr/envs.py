import gymnasium as gym
import torch
import numpy as np
from copy import deepcopy
from collections import OrderedDict
from typing import Any, Callable, List, Optional, Sequence, Type, Union

from stable_baselines3.common.vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.util import (
    dict_to_obs,
    obs_space_info,
)
from stable_baselines3.common.vec_env.base_vec_env import (
    VecEnv,
    VecEnvIndices,
    VecEnvObs,
    VecEnvStepReturn,
)
from stable_baselines3.common.vec_env.vec_normalize import VecNormalize as VecNormalize_

from RL_env import MutatorSelect
from a2c_ppo_acktr.utils import *


def make_env(args, obs_size, eval=False):
    def _thunk():
        env = MutatorSelect(args, obs_size, args.cuda_id, eval=eval)
        print("Finish Build Environment")
        return env

    return _thunk


def make_vec_envs(args, num_processes, obs_size, device, eval=False):
    envs = [make_env(args, obs_size, eval=eval)]
    envs = DummyVecEnv1(envs, num_processes)
    envs = VecPyTorch(envs, device)

    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        if device == -1:
            self.device = "cpu"
        else:
            self.device = f"cuda:{device}"

    def reset(self):
        obs = self.venv.reset()
        obs = torch.from_numpy(obs).float().to(self.device)
        return obs

    def step_async(self, actions):
        if isinstance(actions, torch.LongTensor):
            # Squeeze the dimension for discrete actions
            actions = actions.squeeze(1)
        actions = actions.cpu().numpy()
        self.venv.step_async(actions)

    def step_wait(self):
        obs, reward, done, info = self.venv.step_wait()
        obs = torch.from_numpy(obs).float().to(self.device)
        reward = torch.from_numpy(reward).unsqueeze(dim=1).float()
        return obs, reward, done, info


class VecNormalize(VecNormalize_):
    def __init__(self, *args, **kwargs):
        super(VecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.obs_rms:
            if self.training and update:
                self.obs_rms.update(obs)
            obs = np.clip(
                (obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                -self.clip_obs,
                self.clip_obs,
            )
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


class DummyVecEnv1(VecEnv):
    """
    Creates a simple vectorized wrapper for multiple environments, calling each environment in sequence on the current
    Python process. This is useful for computationally simple environment such as ``cartpole-v1``,
    as the overhead of multiprocess or multithread outweighs the environment computation time.
    This can also be used for RL methods that
    require a vectorized environment, but that you want a single environments to train with.
    :param env_fns: a list of functions
        that return environments to vectorize
    """

    def __init__(self, env_fns: List[Callable[[], gym.Env]], num_processes):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        obs_space = env.observation_space
        self.keys, shapes, dtypes = obs_space_info(obs_space)

        self.buf_obs = OrderedDict(
            [
                (k, np.zeros((num_processes,) + tuple(shapes[k]), dtype=dtypes[k]))
                for k in self.keys
            ]
        )
        self.buf_dones = np.zeros((num_processes,), dtype=bool)
        self.buf_rews = np.zeros((num_processes,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None
        self.metadata = env.metadata

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> VecEnvStepReturn:
        for env_idx in range(self.num_envs):
            obs, self.buf_rews, self.buf_dones, self.buf_infos[env_idx] = self.envs[
                env_idx
            ].step(self.actions)
            if self.buf_dones[env_idx]:
                # save final observation where user can get it, then reset
                # self.buf_infos[env_idx]["terminal_observation"] = obs
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            np.copy(self.buf_dones),
            deepcopy(self.buf_infos),
        )

    def seed(self, seed: Optional[int] = None) -> List[Union[None, int]]:
        if seed is None:
            seed = np.random.randint(0, 2**32 - 1)
        seeds = []
        for idx, env in enumerate(self.envs):
            seeds.append(env.seed(seed + idx))
        return seeds

    def reset(self) -> VecEnvObs:
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[np.ndarray]:
        return [env.render(mode="rgb_array") for env in self.envs]

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """
        Gym environment rendering. If there are multiple environments then
        they are tiled together in one image via ``BaseVecEnv.render()``.
        Otherwise (if ``self.num_envs == 1``), we pass the render call directly to the
        underlying environment.
        Therefore, some arguments such as ``mode`` will have values that are valid
        only when ``num_envs == 1``.
        :param mode: The rendering type.
        """
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        for key in self.keys:
            if key is None:
                # self.buf_obs[key][env_idx] = obs
                self.buf_obs[key] = obs
            else:
                # self.buf_obs[key][env_idx] = obs[key]
                self.buf_obs[key] = obs[key]

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, deepcopy(self.buf_obs))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        """Return attribute from vectorized environment (see base class)."""
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(
        self, attr_name: str, value: Any, indices: VecEnvIndices = None
    ) -> None:
        """Set attribute inside vectorized environments (see base class)."""
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(
        self,
        method_name: str,
        *method_args,
        indices: VecEnvIndices = None,
        **method_kwargs,
    ) -> List[Any]:
        """Call instance methods of vectorized environments."""
        target_envs = self._get_target_envs(indices)
        return [
            getattr(env_i, method_name)(*method_args, **method_kwargs)
            for env_i in target_envs
        ]

    def env_is_wrapped(
        self, wrapper_class: Type[gym.Wrapper], indices: VecEnvIndices = None
    ) -> List[bool]:
        """Check if worker environments are wrapped with a given wrapper"""
        target_envs = self._get_target_envs(indices)
        # Import here to avoid a circular import
        from stable_baselines3.common import env_util

        return [env_util.is_wrapped(env_i, wrapper_class) for env_i in target_envs]

    def _get_target_envs(self, indices: VecEnvIndices) -> List[gym.Env]:
        indices = self._get_indices(indices)
        return [self.envs[i] for i in indices]
