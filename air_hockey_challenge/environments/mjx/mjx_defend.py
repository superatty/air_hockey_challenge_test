import jax
import tqdm
from air_hockey_challenge.environments.mjx.mjx_single import AirHockeyJaxSingle
from jax import numpy as jnp
from mujoco import mjx


class AirHockeyJaxDefend(AirHockeyJaxSingle):
    def __init__(self, batch_size: int, timestep=1 / 1000, n_intermediate_steps=20):
        super().__init__(batch_size, timestep, n_intermediate_steps)

        self.init_velocity_range = (1, 3)
        self.start_range = jnp.array([[0.29, 0.65], [-0.4, 0.4]])  # Table Frame

    def _setup(
        self,
        rng_key: jax.Array,
        mjx_data: mjx.Data,
        robot_model: mjx.Model,
        robot_data: mjx.Data,
        n_agents: int,
    ) -> tuple[mjx.Data, jax.Array, jax.Array]:
        mjx_data, u_joint_pos_prev, u_joint_vel_prev = super()._setup(rng_key, mjx_data, robot_model, robot_data, n_agents)

        subkeys = jax.random.split(rng_key, 4)

        puck_pos = (
            jax.random.uniform(subkeys[0], shape=(self.batch_size, 2))
            * (self.start_range[:, 1] - self.start_range[:, 0])
            + self.start_range[:, 0]
        )

        lin_vel = jax.random.uniform(
            subkeys[1],
            shape=(self.batch_size,),
            minval=self.init_velocity_range[0],
            maxval=self.init_velocity_range[1],
        )
        angle = jax.random.uniform(
            subkeys[2], shape=(self.batch_size), minval=-0.5, maxval=0.5
        )

        puck_vel = jnp.zeros((self.batch_size, 3))
        puck_vel = puck_vel.at[:, 0].set(-jnp.cos(angle) * lin_vel)
        puck_vel = puck_vel.at[:, 1].set(jnp.sin(angle) * lin_vel)
        puck_vel = puck_vel.at[:, 2].set(
            jax.random.uniform(
                subkeys[3], shape=(self.batch_size,), minval=-10, maxval=10
            )
        )

        qpos, qvel = mjx_data.qpos, mjx_data.qvel

        qpos = qpos.at[:, self.joint_name2id["puck_x"]].set(puck_pos[:, 0])
        qpos = qpos.at[:, self.joint_name2id["puck_y"]].set(puck_pos[:, 1])

        qvel = qvel.at[:, self.joint_name2id["puck_x"]].set(puck_vel[:, 0])
        qvel = qvel.at[:, self.joint_name2id["puck_y"]].set(puck_vel[:, 1])
        qvel = qvel.at[:, self.joint_name2id["puck_yaw"]].set(puck_vel[:, 2])

        return mjx_data.replace(qpos=qpos, qvel=qvel), u_joint_pos_prev, u_joint_vel_prev
    
    def _is_absorbing(self, obs: jax.Array) -> jax.Array:        
        puck_pos, puck_vel = self.get_puck(obs)
        
        return jnp.logical_or(
            jnp.logical_or(
                jnp.logical_and(puck_pos[:, 0] > 0, puck_vel[:, 0] > 0),
                jnp.linalg.norm(puck_vel[:, :2], axis=1) < 0.1,
            ),
            super()._is_absorbing(obs),
        )


if __name__ == "__main__":
    from mujoco.viewer import launch_passive

    batch_size = 2

    env = AirHockeyJaxDefend(batch_size=batch_size)

    action = (
        jax.random.uniform(
            jax.random.key(0), shape=(batch_size, 7), minval=-1, maxval=1
        )
        * 8
    )

    # data = mjx.get_data(env.model, env.mjx_data)
    # viewer = launch_passive(env.model, data[0])

    for _ in range(10):
        obs = env.reset()
        
        # mjx.get_data_into(data, env.model, env.mjx_data)
        # viewer.sync()

        for _ in tqdm.tqdm(range(50)):
            obs = env.step(action)

            # mjx.get_data_into(data, env.model, env.mjx_data)
            # viewer.sync()
