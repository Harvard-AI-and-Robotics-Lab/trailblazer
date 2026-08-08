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

from RL_env_seq import MutatorSelectSeq
from a2c_ppo_acktr.utils import *


def make_env_seq(args, obs_size, eval=False, history_size=10):
    """시퀀스 기반 환경을 생성하는 함수"""
    def _thunk():
        env = MutatorSelectSeq(args, obs_size, args.cuda_id, eval=eval, history_size=history_size)
        print("Finish Build Sequence-based Environment")
        return env

    return _thunk


def make_vec_envs_seq(args, num_processes, obs_size, device, eval=False, history_size=10):
    """시퀀스 기반 벡터화된 환경 생성"""
    env_fns = [make_env_seq(args, obs_size, eval=eval, history_size=history_size)]
    
    # 환경 벡터화
    envs = DummyVecEnv1(env_fns)

    # 정규화 래퍼 추가
    if getattr(args, 'use_vec_normalize', True):  # 기본값을 True로 설정
        if len(envs.observation_space.shape) == 1:
            envs = VecNormalize(envs, norm_obs=True, norm_reward=True)
        else:
            envs = VecNormalize(envs, norm_obs=False, norm_reward=True)

    return envs


class DummyVecEnv1(VecEnv):
    """
    시퀀스 기반 환경을 위한 벡터화 래퍼
    """
    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        VecEnv.__init__(self, len(env_fns), env.observation_space, env.action_space)
        shapes, dtypes = {}, {}
        self.keys = []
        obs_space = env.observation_space

        if isinstance(obs_space, gym.spaces.Dict):
            assert len(obs_space.spaces) == 1
            subspace = list(obs_space.spaces.values())[0]
            self.keys = [None]
            shapes, dtypes = obs_space_info(obs_space)
        else:
            self.keys = [None]
            shapes = {None: obs_space.shape}
            dtypes = {None: obs_space.dtype}

        self.buf_obs = OrderedDict([(k, np.zeros((self.num_envs,) + tuple(shapes[k]), dtype=dtypes[k])) for k in self.keys])
        self.buf_dones = np.zeros((self.num_envs,), dtype=np.bool_)
        self.buf_rews = np.zeros((self.num_envs,), dtype=np.float32)
        self.buf_infos = [{} for _ in range(self.num_envs)]
        self.actions = None

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        for env_idx in range(self.num_envs):
            # 액션 인덱스 범위 확인
            if env_idx < len(self.actions):
                action = self.actions[env_idx]
            else:
                action = self.actions[0] if len(self.actions) > 0 else 0
            
            step_result = self.envs[env_idx].step(action)
            obs, rew, done, info = step_result
            
            # 스칼라 값들을 배열로 변환
            if np.isscalar(rew):
                self.buf_rews[env_idx] = rew
            else:
                self.buf_rews[env_idx] = rew[0] if len(rew) > 0 else 0.0
                
            if np.isscalar(done):
                self.buf_dones[env_idx] = done
            else:
                self.buf_dones[env_idx] = done[0] if len(done) > 0 else False
                
            if isinstance(info, list) and len(info) > 0:
                self.buf_infos[env_idx] = info[0]
            else:
                self.buf_infos[env_idx] = info
                
            if self.buf_dones[env_idx]:
                obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return (self._obs_from_buf(), np.copy(self.buf_rews), np.copy(self.buf_dones), deepcopy(self.buf_infos))

    def reset(self):
        for env_idx in range(self.num_envs):
            obs = self.envs[env_idx].reset()
            self._save_obs(env_idx, obs)
        return self._obs_from_buf()

    def close(self):
        for env in self.envs:
            env.close()

    def get_images(self) -> Sequence[np.ndarray]:
        return [env.render(mode="rgb_array") for env in self.envs]

    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        if self.num_envs == 1:
            return self.envs[0].render(mode=mode)
        else:
            return super().render(mode=mode)

    def _save_obs(self, env_idx: int, obs: VecEnvObs) -> None:
        for key in self.keys:
            if key is None:
                self.buf_obs[key] = obs
            else:
                self.buf_obs[key] = obs[key]

    def _obs_from_buf(self) -> VecEnvObs:
        return dict_to_obs(self.observation_space, deepcopy(self.buf_obs))

    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> List[Any]:
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, attr_name) for env_i in target_envs]

    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        target_envs = self._get_target_envs(indices)
        for env_i in target_envs:
            setattr(env_i, attr_name, value)

    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> List[Any]:
        target_envs = self._get_target_envs(indices)
        return [getattr(env_i, method_name)(*method_args, **method_kwargs) for env_i in target_envs]

    def env_is_wrapped(self, wrapper_class, indices: VecEnvIndices = None):
        """Check if environments are wrapped with given wrapper class"""
        target_envs = self._get_target_envs(indices)
        return [hasattr(env_i, wrapper_class.__name__) for env_i in target_envs]


class VecNormalize(VecEnvWrapper):
    """
    벡터화된 환경 정규화 래퍼
    """
    def __init__(self, venv, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=10.0, gamma=0.99, epsilon=1e-8):
        VecEnvWrapper.__init__(self, venv)
        self.norm_obs = norm_obs
        self.norm_reward = norm_reward
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.gamma = gamma
        self.epsilon = epsilon

        if self.norm_obs:
            self.obs_rms = RunningMeanStd(shape=self.observation_space.shape)
        else:
            self.obs_rms = None

        if self.norm_reward:
            self.ret_rms = RunningMeanStd(shape=())
        else:
            self.ret_rms = None

        self.ret = np.zeros(self.num_envs)
        self.gamma = gamma

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        self.ret = self.ret * self.gamma + rews
        obs = self._obfilt(obs)
        if self.norm_reward:
            self.ret_rms.update(self.ret)
            rews = np.clip(rews / np.sqrt(self.ret_rms.var + self.epsilon), -self.clip_reward, self.clip_reward)
        self.ret[dones] = 0.0
        return obs, rews, dones, infos

    def _obfilt(self, obs):
        if self.norm_obs:
            self.obs_rms.update(obs)
            obs = np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon), -self.clip_obs, self.clip_obs)
            return torch.from_numpy(obs).float()
        else:
            return torch.from_numpy(obs).float()

    def reset(self):
        self.ret = np.zeros(self.num_envs)
        obs = self.venv.reset()
        return self._obfilt(obs)

    def get_obs_rms(self):
        return self.obs_rms


class RunningMeanStd(object):
    """Running mean and standard deviation calculator"""
    def __init__(self, epsilon=1e-4, shape=()):
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + np.square(delta) * self.count * batch_count / (self.count + batch_count)
        new_var = M2 / (self.count + batch_count)

        new_count = batch_count + self.count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count

