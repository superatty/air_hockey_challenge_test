from air_hockey_challenge.environments.brax_envs.env_single import AirHockeySingle
from jax import numpy as jnp
import jax
from brax.envs.base import State

class AirHockeyPrepare(AirHockeySingle):
    def __init__(self, timestep=1 / 1000, n_intermediate_steps=20):
        super().__init__(timestep=timestep, n_intermediate_steps=n_intermediate_steps)

        width_high = (
            self.env_info["table"]["width"] / 2
            - self.env_info["puck"]["radius"]
            - 0.002
        )
        width_low = (
            self.env_info["table"]["width"] / 2
            - self.env_info["puck"]["radius"]
            - self.env_info["mallet"]["radius"] * 2
        )

        self.side_range = jnp.array([[-0.8, -0.2], [width_low, width_high]])
        self.bottom_range = jnp.array(
            [[-0.94, -0.8], [self.env_info["table"]["goal_width"] / 2, width_high]]
        )

        self.side_area = (self.side_range[0, 1] - self.side_range[0, 0]) * (
            self.side_range[1, 1] - self.side_range[1, 0]
        )
        self.bottom_area = (self.bottom_range[0, 1] - self.bottom_range[0, 0]) * (
            self.bottom_range[1, 1] - self.bottom_range[1, 0]
        )

    def _setup(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        qpos, qvel = super()._setup(rng)

        rng, rng1, rng2, rng3 = jax.random.split(rng, 4)
        
        start_range = jax.lax.cond(
            jax.random.uniform(rng1) >= self.side_area / (self.side_area + self.bottom_area),
            lambda: self.bottom_range,
            lambda: self.side_range
        )
            
        puck_pos = jax.random.uniform(
            rng2, shape=(2)
        ) * (start_range[:, 1] - start_range[:, 0]) + start_range[:, 0]
        
        puck_pos_multiplier_y = jnp.array([1, -1])[jax.random.randint(rng3, (), minval=0, maxval=2)]
        puck_pos *= jnp.array([1, puck_pos_multiplier_y])

        qpos = qpos.at[self.puck_ids[:2]].set(puck_pos)

        return qpos, qvel
    
    def _is_absorbing(self, state: State) -> tuple[State, jax.Array]:
        obs = state.info["internal_obs"]
        
        puck_pos, puck_vel = self.get_puck(obs)
        
        parent_done = super()._is_absorbing(state)

        done = jnp.logical_or.reduce(
            jnp.array([
                puck_pos[0] > 0, # if puck is over the middle line
                jnp.abs(puck_pos[1]) < 1e-2, # if puck is almost at the center line
                parent_done[1],
            ])
        )

        return state, done * 1.0
        
if __name__ == "__main__":
    import os
    from tqdm import tqdm

    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_triton_gemm_any=True "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
    )

    jax.config.update("jax_enable_x64", True)

    env = AirHockeyPrepare()
    rng = jax.random.PRNGKey(0)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    state = jit_reset(rng)
    action = jnp.zeros(7)

    for i in tqdm(range(100)):
        state = jit_step(state, action)
