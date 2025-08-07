from typing import Optional

import jax
import numpy as np
from brax.envs.base import PipelineEnv
from brax.io import image as brax_image
from gymnasium import spaces

from stable_baselines3.common.vec_env import VecEnv


class BraxSB3Wrapper(VecEnv):
    """A wrapper that converts batched Brax Env to one that follows SB3 VecEnv API."""

    def __init__(
        self,
        env: PipelineEnv,
        seed: int = 0,
        backend: Optional[str] = None,
        keep_infos: bool = True,
    ) -> None:
        self._env = env
        self.metadata = {
            "render_modes": ["human", "rgb_array"],
            "video.frames_per_second": 1 / self._env.dt,
        }
        self.render_mode = "rgb_array"
        if not hasattr(self._env, "batch_size"):
            raise ValueError("underlying env must be batched")

        self.num_envs = self._env.batch_size
        self.seed(seed)
        self.backend = backend
        self._state = None
        self.keep_infos = keep_infos
        self.default_infos = [{} for _ in range(self.num_envs)]

        obs = np.inf * np.ones(self._env.observation_size, dtype=np.float32)
        self.observation_space = spaces.Box(-obs, obs, dtype=np.float32)

        action = jax.tree.map(np.array, self._env.sys.actuator.ctrl_range)
        self.action_space = spaces.Box(action[:, 0], action[:, 1], dtype=np.float32)

        def reset(key):
            key1, key2 = jax.random.split(key)
            state = self._env.reset(key2)
            return state, state.obs, key1

        self._reset = jax.jit(reset, backend=self.backend)

        def step(state, action):
            state = self._env.step(state, action)
            # Note: they don't seem to handle truncation properly
            info = {**state.metrics, **state.info}
            return state, state.obs, state.reward, state.done, info

        self._step = jax.jit(step, backend=self.backend)

    def reset(self) -> np.ndarray:
        self._state, obs, self._key = self._reset(self._key)
        return np.array(obs)

    def step_async(self, actions: np.ndarray) -> None:
        self.actions = actions

    def step_wait(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, list]:
        # TODO: add last observation too?
        self._state, obs, rewards, dones, info = self._step(self._state, self.actions)
        # Convert from dict of list to list of dicts
        if self.keep_infos:
            # May be slow with many envs
            infos = self.to_list(info)
        else:
            infos = self.default_infos

        return np.array(obs), np.array(rewards), np.array(dones).astype(bool), infos

    def seed(self, seed: int = 0) -> None:
        self._key = jax.random.PRNGKey(seed)

    def render(self, mode: str = "human") -> None:
        if mode == "rgb_array":
            if self._state is None:
                raise RuntimeError("Must call reset or step before rendering")
            return brax_image.render_array(
                self._env.sys, self._state.pipeline_state, 256, 256
            )
        else:
            # Use opencv to render
            return super().render(mode="human")

    def get_images(self):
        state_list = [self._state.take(i).pipeline_state for i in range(self.num_envs)]
        return brax_image.render_array(self._env.sys, state_list, width=256, height=256)

    def env_is_wrapped(self, wrapper_class, indices=None):
        # For compatibility with eval and monitor helpers
        return [False]

    def set_attr(self, attr_name, value, indices=None):
        raise NotImplementedError("Setting attributes is not supported.")

    def get_attr(self, attr_name, indices=None):
        # resolve indices
        if indices is None:
            indices = slice(None)
            num_indices = self.num_envs
        else:
            num_indices = len(indices)
        attr_val = getattr(self, attr_name)
        return [attr_val] * num_indices

    def env_method(self, method_name: str, *method_args, indices=None, **method_kwargs):
        raise NotImplementedError()

    def close(self) -> None:
        pass

    def to_list(self, info_dict: dict):
        infos = [dict.fromkeys(info_dict.keys()) for _ in range(self.num_envs)]
        # From https://github.com/isaac-sim/IsaacLab
        # fill-in information for each sub-environment
        # note: This loop becomes slow when number of environments is large.
        for idx in range(self.num_envs):
            # fill-in bootstrap information
            # TODO: use "truncation" key
            # infos[idx]["TimeLimit.truncated"] = truncated[idx] and not terminated[idx]
            # TODO: use first-obs?
            # infos[idx]["terminal_observation"] = None
            # fill-in information from extras
            for key, value in info_dict.items():
                try:
                    infos[idx][key] = value[idx]
                except TypeError:
                    # Note: doesn't work for State object
                    pass
        # return list of dictionaries
        return infos


# ['ant', 'halfcheetah', 'hopper', 'humanoid', 'humanoidstandup',
# 'inverted_pendulum', 'inverted_double_pendulum', 'pusher', 'reacher', 'walker2d']
# env_name = "hopper"
# backend = "generalized"  # ['mjx', 'generalized', 'positional', 'spring']

# n_envs = 1024
# # Create a vectorized environment
# vmap_env = envs.create(env_name, backend=backend, batch_size=n_envs)
# # vmap_env = training.VmapWrapper(base_env, batch_size=n_envs)
# vec_env = BraxSB3Wrapper(vmap_env, keep_infos=False)
# vec_env = VecMonitor(vec_env)
# vec_env = VecNormalize(vec_env, norm_reward=False)


# simba_hyperparams = dict(
#     batch_size=256,
#     # buffer_size=100_000,
#     learning_rate=3e-4,
#     policy_kwargs={
#         "optimizer_class": optax.adamw,
#         "net_arch": {"pi": [128], "qf": [256, 256]},
#         "n_critics": 2,
#     },
#     learning_starts=10_000,
#     # normalize={"norm_obs": True, "norm_reward": False},
#     # resets=[50000, 75000],
# )

# # model = PPO(
# #     "MlpPolicy",
# #     vec_env,
# #     n_steps=64,
# #     batch_size=1024,
# #     n_epochs=4,
# #     verbose=1,
# #     device="cpu",
# # )

# # model = TQC(
# #     "SimbaPolicy",
# #     vec_env,
# #     train_freq=5,
# #     gradient_steps=min(n_envs, 256),
# #     policy_delay=10,
# #     verbose=1,
# #     **simba_hyperparams,
# # )

# model = SAC(
#     "MlpPolicy",
#     vec_env,
#     train_freq=5,
#     gradient_steps=min(n_envs, 256),
#     policy_delay=10,
#     verbose=1,
# )

# # Training
# try:
#     model.learn(total_timesteps=int(3e7), progress_bar=True, log_interval=10)
# except KeyboardInterrupt:
#     pass

# # Evaluate the model
# print(evaluate_policy(model, vec_env, n_eval_episodes=10, render=True))