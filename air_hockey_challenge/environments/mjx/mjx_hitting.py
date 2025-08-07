from air_hockey_challenge.environments.mjx.mjx_double import AirHockeyJaxDouble
from jax import numpy as jnp
import jax
from mujoco import mjx


class AirHockeyJaxHit(AirHockeyJaxDouble):
    def __init__(
        self,
        batch_size: int,
        opponent_agent=None,
        moving_init=True,
        timestep=1 / 1000,
        n_intermediate_steps=20,
    ):
        super().__init__(
            batch_size=batch_size,
            timestep=timestep,
            n_intermediate_steps=n_intermediate_steps,
        )

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
            self._opponent_agent = lambda: jnp.zeros((1, 7))

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

        subkeys = jax.random.split(rng_key, 4)

        puck_pos = (
            jax.random.uniform(subkeys[0], shape=(self.batch_size, 2))
            * (self.hit_range[:, 1] - self.hit_range[:, 0])
            + self.hit_range[:, 0]
        )

        qpos, qvel = mjx_data.qpos, mjx_data.qvel

        qpos = qpos.at[:, self.joint_name2id["puck_x"]].set(puck_pos[:, 0])
        qpos = qpos.at[:, self.joint_name2id["puck_y"]].set(puck_pos[:, 1])

        if self.moving_init:
            lin_vel = jax.random.uniform(
                subkeys[1],
                shape=(self.batch_size,),
                minval=self.init_velocity_range[0],
                maxval=self.init_velocity_range[1],
            )
            angle = jax.random.uniform(
                subkeys[2],
                shape=(self.batch_size),
                minval=-jnp.pi / 2 - 0.1,
                maxval=jnp.pi / 2 + 0.1,
            )

            puck_vel = jnp.empty((self.batch_size, 3))
            puck_vel = puck_vel.at[:, 0].set(-jnp.cos(angle) * lin_vel)
            puck_vel = puck_vel.at[:, 1].set(jnp.sin(angle) * lin_vel)
            puck_vel = puck_vel.at[:, 2].set(
                jax.random.uniform(
                    subkeys[3], shape=(self.batch_size,), minval=-2, maxval=2
                )
            )

            qvel = qvel.at[:, self.joint_name2id["puck_x"]].set(puck_vel[:, 0])
            qvel = qvel.at[:, self.joint_name2id["puck_y"]].set(puck_vel[:, 1])
            qvel = qvel.at[:, self.joint_name2id["puck_yaw"]].set(puck_vel[:, 2])

        return (
            mjx_data.replace(qpos=qpos, qvel=qvel),
            u_joint_pos_prev,
            u_joint_vel_prev,
        )

    def _modify_observation(self, obs: jax.Array) -> jax.Array:
        obs = super()._modify_observation(obs)

        return jnp.split(obs, 2, axis=-1)[0]

    def _preprocess_action(self, action: jax.Array) -> jax.Array:
        action = super()._preprocess_action(action)

        opponent_action = self._opponent_agent()  # TODO add obs here as input

        return action, opponent_action

        # return jnp.concatenate(
        #     [
        #         action,
        #         opponent_action,
        #     ],
        #     axis=-1,
        # )

    def _is_absorbing(self, obs):
        puck_pos, puck_vel = self.get_puck(obs)

        # Stop if the puck bounces back on the opponents wall
        return jnp.logical_or(
            jnp.logical_and(puck_pos[:, 0] > 0, puck_vel[:, 0] < 0),
            super()._is_absorbing(obs),
        )


if __name__ == "__main__":
    # from mujoco.viewer import launch_passive
    import tqdm

    batch_size = 1

    env = AirHockeyJaxHit(batch_size=batch_size)

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
