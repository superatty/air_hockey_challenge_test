import jax
import numpy as np
import tqdm
from air_hockey_challenge.environments.mjx.mjx_base import AirHockeyJaxBase
from jax import numpy as jnp
from mujoco import mjx
from scipy.spatial.transform import Rotation as R
from air_hockey_challenge.utils.kinematics import inverse_kinematics

from air_hockey_challenge.utils.mjx.universal_joint_plugin import reset


class AirHockeyJaxSingle(AirHockeyJaxBase):
    def __init__(self, batch_size: int, timestep=1 / 1000, n_intermediate_steps=20):
        super().__init__(
            n_agents=1,
            batch_size=batch_size,
            timestep=timestep,
            n_intermediate_steps=n_intermediate_steps,
        )

        self._compute_init_state()

    def _compute_init_state(self):
        init_state = np.array([0.0, -0.1961, 0.0, -1.8436, 0.0, 0.9704, 0.0])

        success, self.init_state = inverse_kinematics(
            self.robot_model,
            self.robot_data,
            np.array([0.65, 0.0, 0.1645]),
            R.from_euler("xyz", [0, 5 / 6 * np.pi, 0]).as_matrix(),
            initial_q=init_state,
        )

        assert success is True

    def _setup(
        self,
        rng_key: jax.Array,
        mjx_data: mjx.Data,
        robot_model: mjx.Model,
        robot_data: mjx.Data,
        n_agents: int,
    ) -> tuple[mjx.Data, jnp.ndarray, jnp.ndarray]:
        mjx_data, _, _ = super()._setup(
            rng_key, mjx_data, robot_model, robot_data, n_agents
        )

        qpos = mjx_data.qpos

        for i in range(7):
            jnt_id = self.joint_name2id[f"iiwa_1/joint_{i + 1}"]
            qpos = qpos.at[:, jnt_id].set(self.init_state[i])

        mjx_data = mjx_data.replace(qpos=qpos)
        mjx_data, u_joint_pos_prev, u_joint_vel_prev = reset(
            self.mjx_model,
            mjx_data,
            robot_model,
            robot_data,
            n_agents,
            self.filter_ratio,
        )
        return mjx_data, u_joint_pos_prev, u_joint_vel_prev

    def get_ee(self):
        ee_pos = self.mjx_data.xpos[:, self.body_name2id["iiwa_1/striker_mallet"]]
        ee_vel = self.mjx_data.xvel[:, self.body_name2id["iiwa_1/striker_mallet"]]
        return ee_pos, ee_vel

    def get_joints(self, obs):
        q_pos = obs[:, self.env_info["joint_pos_ids"]]
        q_vel = obs[:, self.env_info["joint_vel_ids"]]
        return q_pos, q_vel

    def _create_observation(self, obs):
        q_pos, q_vel = self.get_joints(obs)
        q_vel_filter = (
            self.filter_ratio * q_vel + (1 - self.filter_ratio) * self.q_vel_prev
        )
        self.q_pos_prev = q_pos
        self.q_vel_prev = q_vel_filter
        obs = obs.at[:, self.env_info["joint_vel_ids"]].set(q_vel_filter)

        yaw_angle = obs[:, self.env_info["puck_pos_ids"][2]]
        obs = obs.at[:, self.env_info["puck_pos_ids"][2]].set(
            (yaw_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
        )

        return obs

    def _modify_observation(self, obs):
        puck_pos, puck_vel = self.get_puck(obs)

        puck_pos = jax.vmap(self._puck_pose_2d_in_robot_frame, in_axes=(0, None))(
            puck_pos, self.env_info["robot"]["base_frame"][0]
        )
        puck_vel = jax.vmap(self._puck_vel_2d_in_robot_frame, in_axes=(0, None))(
            puck_vel, self.env_info["robot"]["base_frame"][0]
        )

        obs = obs.at[:, self.env_info["puck_pos_ids"]].set(puck_pos)
        obs = obs.at[:, self.env_info["puck_vel_ids"]].set(puck_vel)

        return obs


if __name__ == "__main__":
    from mujoco.viewer import launch_passive

    batch_size = 1

    env = AirHockeyJaxSingle(batch_size=batch_size)

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
