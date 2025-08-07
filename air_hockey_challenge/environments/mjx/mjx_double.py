import numpy as np
import tqdm
from air_hockey_challenge.environments.mjx.mjx_base import AirHockeyJaxBase
from scipy.spatial.transform import Rotation as R

from jax import numpy as jnp
import jax
from mujoco import mjx

from air_hockey_challenge.utils.kinematics import inverse_kinematics
from air_hockey_challenge.utils.mjx.universal_joint_plugin import reset


class AirHockeyJaxDouble(AirHockeyJaxBase):
    def __init__(self, batch_size: int, timestep=1 / 1000, n_intermediate_steps=20):
        super().__init__(
            n_agents=2,
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
            iiwa1_jnt_id = self.joint_name2id[f"iiwa_1/joint_{i + 1}"]
            qpos = qpos.at[:, iiwa1_jnt_id].set(self.init_state[i])

            iiwa2_jnt_id = self.joint_name2id[f"iiwa_2/joint_{i + 1}"]
            qpos = qpos.at[:, iiwa2_jnt_id].set(self.init_state[i])

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

    def get_ee(self, robot=1):
        ee_pos = self.mjx_data.xpos[
            :, self.body_name2id[f"iiwa_{robot}/striker_mallet"]
        ]
        ee_vel = self.mjx_data.xvel[
            :, self.body_name2id[f"iiwa_{robot}/striker_mallet"]
        ]
        return ee_pos, ee_vel

    def get_joints(self, obs, robot=None):
        if robot == 1:
            q_pos = obs[:, self.env_info["joint_pos_ids"]]
            q_vel = obs[:, self.env_info["joint_vel_ids"]]
        elif robot == 2:
            q_pos = obs[:, 23 + jnp.asarray(self.env_info["joint_pos_ids"])]
            q_vel = obs[:, 23 + jnp.asarray(self.env_info["joint_vel_ids"])]
        else:
            assert robot is None

            q_pos = jnp.concatenate(
                [
                    obs[:, self.env_info["joint_pos_ids"]],
                    obs[:, 23 + jnp.asarray(self.env_info["joint_pos_ids"])],
                ],
                axis=1,
            )
            q_vel = jnp.concatenate(
                [
                    obs[:, self.env_info["joint_vel_ids"]],
                    obs[:, 23 + jnp.asarray(self.env_info["joint_vel_ids"])],
                ],
                axis=1,
            )

        return q_pos, q_vel

    def _create_observation(self, obs):
        q_pos, q_vel = self.get_joints(obs)
        q_vel_filter = (
            self.filter_ratio * q_vel + (1 - self.filter_ratio) * self.q_vel_prev
        )
        self.q_pos_prev = q_pos
        self.q_vel_prev = q_vel_filter

        obs = obs.at[:, self.env_info["joint_vel_ids"]].set(q_vel_filter[:, :7])
        obs = obs.at[:, 23 + jnp.asarray(self.env_info["joint_vel_ids"])].set(
            q_vel_filter[:, 7:]
        )

        yaw_angle = obs[:, self.env_info["puck_pos_ids"][2]]
        obs = obs.at[:, self.env_info["puck_pos_ids"][2]].set(
            (yaw_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi
        )

        return obs

    def _modify_observation(self, obs):
        puck_pos, puck_vel = self.get_puck(obs)

        puck_pos_1 = jax.vmap(self._puck_pose_2d_in_robot_frame, in_axes=(0, None))(
            puck_pos, self.env_info["robot"]["base_frame"][0]
        )
        puck_vel_1 = jax.vmap(self._puck_vel_2d_in_robot_frame, in_axes=(0, None))(
            puck_vel, self.env_info["robot"]["base_frame"][0]
        )
        obs = obs.at[:, self.env_info["puck_pos_ids"]].set(puck_pos_1)
        obs = obs.at[:, self.env_info["puck_vel_ids"]].set(puck_vel_1)

        opponent_ee_pos = obs[:, self.env_info["opponent_ee_ids"]]
        opponent_ee_pos = jnp.einsum("ij, bj->bi", np.linalg.inv(self.env_info["robot"]["base_frame"][0]), jnp.concatenate([opponent_ee_pos, jnp.ones((self.batch_size, 1))], axis=1))[:, :3]
        obs = obs.at[:, self.env_info["opponent_ee_ids"]].set(opponent_ee_pos)

        puck_pos_2 = jax.vmap(self._puck_pose_2d_in_robot_frame, in_axes=(0, None))(
            puck_pos, self.env_info["robot"]["base_frame"][1]
        )
        puck_vel_2 = jax.vmap(self._puck_vel_2d_in_robot_frame, in_axes=(0, None))(
            puck_vel, self.env_info["robot"]["base_frame"][1]
        )
        obs = obs.at[:, 23 + jnp.array(self.env_info["puck_pos_ids"])].set(puck_pos_2)
        obs = obs.at[:, 23 + jnp.array(self.env_info["puck_vel_ids"])].set(puck_vel_2)

        ee_pos = obs[:, 23 + jnp.array(self.env_info["opponent_ee_ids"])]
        ee_pos = jnp.einsum("ij, bj->bi", np.linalg.inv(self.env_info["robot"]["base_frame"][1]), jnp.concatenate([ee_pos, jnp.ones((self.batch_size, 1))], axis=1))[:, :3]
        obs = obs.at[:, 23 + jnp.array(self.env_info["opponent_ee_ids"])].set(
            ee_pos
        )

        return obs


if __name__ == "__main__":
    from mujoco.viewer import launch_passive, launch

    batch_size = 1

    env = AirHockeyJaxDouble(batch_size=batch_size)

    action = (
        jax.random.uniform(
            jax.random.key(1), shape=(batch_size, 14), minval=-1, maxval=1
        )
        * 8
    )
    # action = jnp.zeros((batch_size, 14))

    # data = mjx.get_data(env.model, env.mjx_data)
    # viewer = launch_passive(env.model, data[0])

    for _ in range(10):
        obs = env.reset()

        # mjx.get_data_into(data, env.model, env.mjx_data)
        # viewer = launch(env.model, data[0])
        # viewer.sync()

        for _ in tqdm.tqdm(range(50)):
            obs = env.step(action)

            # mjx.get_data_into(data, env.model, env.mjx_data)
            # viewer.sync()
