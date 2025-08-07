import jax
import tqdm
from air_hockey_challenge.environments.mjx.mjx_single import AirHockeyJaxSingle

from jax import numpy as jnp
from mujoco import mjx


class AirHockeyJaxPrepare(AirHockeyJaxSingle):
    def __init__(self, batch_size: int, timestep=1 / 1000, n_intermediate_steps=20):
        super().__init__(batch_size, timestep, n_intermediate_steps)

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

    def _setup(
        self,
        rng_key: jax.Array,
        mjx_data: mjx.Data,
        robot_model: mjx.Model,
        robot_data: mjx.Data,
        n_agents: int,
    ) -> tuple[mjx.Data, jax.Array, jax.Array]:
        mjx_data, u_joint_pos_prev, u_joint_vel_prev = super()._setup(
            rng_key, mjx_data, robot_model, robot_data, n_agents
        )

        subkeys = jax.random.split(rng_key, 3)

        start_range = jax.vmap(
            lambda prob: (
                jax.lax.cond(
                    prob < self.side_area / (self.side_area + self.bottom_area),
                    lambda _: self.bottom_range,
                    lambda _: self.side_range,
                    operand=None,
                )
            )
        )(jax.random.uniform(subkeys[0], shape=(self.batch_size,)))

        puck_pos = (
            jax.random.uniform(subkeys[1], shape=(self.batch_size, 2))
            * (start_range[:, :, 1] - start_range[:, :, 0])
            + start_range[:, :, 0]
        )

        puck_pos_multiplier_y = jnp.array([1, -1])[
            jax.random.randint(subkeys[2], shape=(batch_size,), minval=0, maxval=2)
        ]
        puck_pos_multiplier = jnp.stack(
            [jnp.ones_like(puck_pos[:, 0]), puck_pos_multiplier_y], axis=-1
        )
        puck_pos = puck_pos * puck_pos_multiplier

        qpos = mjx_data.qpos

        qpos = qpos.at[:, self.joint_name2id["puck_x"]].set(puck_pos[:, 0])
        qpos = qpos.at[:, self.joint_name2id["puck_y"]].set(puck_pos[:, 1])

        return mjx_data.replace(qpos=qpos), u_joint_pos_prev, u_joint_vel_prev

    def _is_absorbing(self, obs):
        puck_pos, puck_vel = self.get_puck(obs)

        return jnp.logical_or(
            jnp.logical_or(puck_pos[:, 0] > 0, jnp.abs(puck_pos[:, 1] < 1e-2)),
            super()._is_absorbing(obs),
        )


if __name__ == "__main__":
    from mujoco.viewer import launch_passive

    batch_size = 16

    env = AirHockeyJaxPrepare(batch_size=batch_size)

    action = (
        jax.random.uniform(
            jax.random.key(0), shape=(batch_size, 7), minval=-1, maxval=1
        )
        * 8
    )

    data = mjx.get_data(env.model, env.mjx_data)
    viewer = launch_passive(env.model, data[0])

    for _ in range(10):
        obs = env.reset()

        mjx.get_data_into(data, env.model, env.mjx_data)
        viewer.sync()

        for _ in tqdm.tqdm(range(50)):
            obs = env.step(action)

            mjx.get_data_into(data, env.model, env.mjx_data)
            viewer.sync()
