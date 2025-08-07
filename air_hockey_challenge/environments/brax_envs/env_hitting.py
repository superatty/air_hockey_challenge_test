from air_hockey_challenge.environments.brax_envs.env_double import AirHockeyDouble
from jax import numpy as jnp
import jax
from brax.envs.base import State

class AirHockeyHit(AirHockeyDouble):
    def __init__(
        self,
        opponent_agent=None,
        moving_init=True,
        timestep=1 / 1000,
        n_intermediate_steps=20,
    ):
        super().__init__(timestep=timestep, n_intermediate_steps=n_intermediate_steps)

        self.moving_init = moving_init
        hit_width = (
            self.env_info["table"]["width"] / 2
            - self.env_info["puck"]["radius"]
            - self.env_info["mallet"]["radius"] * 2
        )
        self.hit_range = jnp.array(
            [[-0.7, -0.2], [-hit_width, hit_width]]
        )  # Table Frame
        self.init_velocity_range = (0, 0.5)  # Table Frame
        self.init_ee_range = jnp.array([[0.60, 1.25], [-0.4, 0.4]])  # Robot Frame

        if opponent_agent is not None:
            self._opponent_agent = opponent_agent.draw_action
        else:
            self._opponent_agent = lambda state: (state, jnp.zeros(7))

    def _setup(self, rng: jax.Array) -> tuple[jax.Array, jax.Array]:
        qpos, qvel = super()._setup(rng)

        rng, rng1, rng2, rng3, rng4 = jax.random.split(rng, 5)

        puck_pos = (
            jax.random.uniform(rng1, shape=(2,))
            * (self.hit_range[:, 1] - self.hit_range[:, 0])
            + self.hit_range[:, 0]
        )

        qpos = qpos.at[self.puck_ids[:2]].set(puck_pos)

        if self.moving_init:
            lin_vel = (
                jax.random.uniform(rng2)
                * (self.init_velocity_range[1] - self.init_velocity_range[0])
                + self.init_velocity_range[0]
            )
            
            angle = jax.random.uniform(rng3, minval=-jnp.pi / 2 - 0.1, maxval=jnp.pi / 2 + 0.1)
            
            puck_vel = jnp.empty(3)
            puck_vel = puck_vel.at[0].set(-jnp.cos(angle) * lin_vel)
            puck_vel = puck_vel.at[1].set(jnp.sin(angle) * lin_vel)
            puck_vel = puck_vel.at[2].set(jax.random.uniform(rng4, minval=-2, maxval=2))
            
            qvel = qvel.at[self.puck_ids].set(puck_vel)

        return qpos, qvel

    def _preprocess_action(self, state: State, action: jax.Array):
        state, action = super()._preprocess_action(state, action)
        
        state, opponent_action = self._opponent_agent(state)
        
        return state, (action, opponent_action)
    
    def _is_absorbing(self, state: State) -> tuple[State, jax.Array]:
        puck_pos, puck_vel = self.get_puck(state.info["internal_obs"])
        
        parent_done = super()._is_absorbing(state)

        done = jnp.logical_or(
            jnp.logical_and(puck_pos[0] > 0, puck_vel[0] < 0), # if puck bounces back on the opponent's wall
            parent_done[1],
        )

        return state, done * 1.0
    
if __name__ == "__main__":
    import os

    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_triton_gemm_any=True "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
    )

    jax.config.update("jax_enable_x64", True)

    env = AirHockeyHit()
    rng = jax.random.PRNGKey(0)
    
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    
    state = jit_reset(rng)
    action = jnp.zeros(7)
    
    for i in range(100):
        state = jit_step(state, action)