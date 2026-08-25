from __future__ import annotations
from random import choice

import torch
from torch import tensor, empty, randn, randint, full
from torch.nn import Module

from einops import repeat

# helpers

def exists(v):
    return v is not None

# mock env

class MockEnv(Module):
    def __init__(
        self,
        image_shape,
        reward_range = (-100, 100),
        num_envs = 1,
        vectorized = False,
        terminate_after_step = None,
        rand_terminate_prob = 0.05,
        can_truncate = False,
        rand_truncate_prob = 0.05,
    ):
        super().__init__()
        self.image_shape = image_shape
        self.reward_range = reward_range

        self.num_envs = num_envs if vectorized else 1
        self.vectorized = vectorized
        assert not (vectorized and num_envs == 1)

        # mocking termination and truncation

        self.can_terminate = exists(terminate_after_step)
        self.terminate_after_step = terminate_after_step
        self.rand_terminate_prob = rand_terminate_prob

        self.can_truncate = can_truncate
        self.rand_truncate_prob = rand_truncate_prob

        self.register_buffer('_step', tensor(0))

    def get_random_state(self):
        return randn(3, *self.image_shape)

    def reset(
        self,
        seed = None
    ):
        self._step.zero_()
        state = self.get_random_state()

        if self.vectorized:
            state = repeat(state, '... -> b ...', b = self.num_envs)

        return state

    def step(
        self,
        actions,
    ):
        state = self.get_random_state()

        reward = empty(()).uniform_(*self.reward_range)

        if self.vectorized:
            _actions = actions[0] if isinstance(actions, tuple) else actions
            assert _actions.shape[0] == self.num_envs, f'expected batch of actions for {self.num_envs} environments'

            state = repeat(state, '... -> b ...', b = self.num_envs)
            reward = repeat(reward, ' -> b', b = self.num_envs)

        shape = (self.num_envs,) if self.vectorized else ()
        valid_step = self._step > self.terminate_after_step if self.can_terminate else full(shape, True, dtype = torch.bool)

        terminated = (torch.rand(shape) < self.rand_terminate_prob) & valid_step if self.can_terminate else full(shape, False, dtype = torch.bool)

        truncated = full(shape, False, dtype = torch.bool)

        if self.can_truncate:
            truncated = (torch.rand(shape) < self.rand_truncate_prob) & valid_step & ~terminated

        self._step.add_(1)

        return state, reward, terminated, truncated, dict()

class MockDictEnv(Module):
    def __init__(
        self,
        image_shape,
        dim_proprio,
        num_envs = 1,
        vectorized = False,
        terminate_after_step = None,
    ):
        super().__init__()
        self.image_shape = image_shape
        self.dim_proprio = dim_proprio
        self.num_envs = num_envs if vectorized else 1
        self.vectorized = vectorized
        self.terminate_after_step = terminate_after_step

        self.register_buffer('_step', tensor(0))

    def reset(self, seed = None):
        self._step.zero_()

        image_shape = (self.num_envs, 3, *self.image_shape) if self.vectorized else (3, *self.image_shape)
        proprio_shape = (self.num_envs, self.dim_proprio) if self.vectorized else (self.dim_proprio,)

        return {
            'image': randn(image_shape),
            'proprio': randn(proprio_shape)
        }

    def step(self, actions):
        self._step.add_(1)

        image_shape = (self.num_envs, 3, *self.image_shape) if self.vectorized else (3, *self.image_shape)
        proprio_shape = (self.num_envs, self.dim_proprio) if self.vectorized else (self.dim_proprio,)

        obs = {
            'image': randn(image_shape),
            'proprio': randn(proprio_shape)
        }

        reward = randn(self.num_envs) if self.vectorized else randn(())

        shape = (self.num_envs,) if self.vectorized else ()
        terminated = torch.full(shape, False, dtype = torch.bool)

        if exists(self.terminate_after_step) and self._step >= self.terminate_after_step:
            terminated = ~terminated

        truncated = torch.full(shape, False, dtype = torch.bool)

        return obs, reward, terminated, truncated, dict()
