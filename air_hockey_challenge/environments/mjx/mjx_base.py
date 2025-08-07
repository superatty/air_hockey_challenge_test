import os
import jax
from mujoco import mjx
from mujoco.mjx._src import math as mjx_math
from jax import numpy as jnp
import mujoco
import numpy as np
from jax.scipy.spatial.transform import Rotation as R

from air_hockey_challenge.environments.data.mjx import __file__ as env_path
from air_hockey_challenge.utils.mjx.universal_joint_plugin import update


class AirHockeyJaxBase:
    def __init__(
        self, n_agents, batch_size: int, timestep=1 / 1000, n_intermediate_steps=20
    ):
        self.batch_size = batch_size
        self.timestep = timestep
        self.n_intermediate_steps = n_intermediate_steps
        self.n_agents = n_agents

        xml_file = os.path.join(
            os.path.dirname(os.path.abspath(env_path)),
            "single.xml" if n_agents == 1 else "double.xml",
        )

        self.model = mujoco.MjModel.from_xml_path(xml_file)

        if timestep is not None:
            self.model.opt.timestep = timestep
            self._timestep = timestep
        else:
            self._timestep = self.model.opt.timestep

        self.mjx_model = mjx.put_model(self.model)

        single_mjx_data = mjx.make_data(self.model)
        self.mjx_data = jax.vmap(lambda _, x: x, in_axes=(0, None))(
            jnp.arange(self.batch_size), single_mjx_data
        )

        joint_names = [
            "puck_x",
            "puck_y",
            "puck_yaw",
            "iiwa_1/joint_1",
            "iiwa_1/joint_2",
            "iiwa_1/joint_3",
            "iiwa_1/joint_4",
            "iiwa_1/joint_5",
            "iiwa_1/joint_6",
            "iiwa_1/joint_7",
        ]

        action_spec = [
            "iiwa_1/joint_1",
            "iiwa_1/joint_2",
            "iiwa_1/joint_3",
            "iiwa_1/joint_4",
            "iiwa_1/joint_5",
            "iiwa_1/joint_6",
            "iiwa_1/joint_7",
        ]

        body_names = [
            "iiwa_1/base",
            "iiwa_1/striker_joint_link",
            "iiwa_1/striker_mallet",
        ]

        if n_agents == 2:
            joint_names += [
                "iiwa_2/joint_1",
                "iiwa_2/joint_2",
                "iiwa_2/joint_3",
                "iiwa_2/joint_4",
                "iiwa_2/joint_5",
                "iiwa_2/joint_6",
                "iiwa_2/joint_7",
            ]

            body_names += [
                "iiwa_2/base",
                "iiwa_2/striker_joint_link",
                "iiwa_2/striker_mallet",
            ]

            action_spec += [
                "iiwa_2/joint_1",
                "iiwa_2/joint_2",
                "iiwa_2/joint_3",
                "iiwa_2/joint_4",
                "iiwa_2/joint_5",
                "iiwa_2/joint_6",
                "iiwa_2/joint_7",
            ]

        self.joint_name2id = {
            jnt_name: mjx.name2id(self.mjx_model, mujoco.mjtObj.mjOBJ_JOINT, jnt_name)
            for jnt_name in joint_names
        }

        self.body_name2id = {
            body_name: mjx.name2id(self.mjx_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            for body_name in body_names
        }

        self.action_indices = [
            mjx.name2id(self.mjx_model, mujoco.mjtObj.mjOBJ_ACTUATOR, jnt_name)
            for jnt_name in action_spec
        ]

        self.env_info = dict()
        self.env_info["table"] = {"length": 1.948, "width": 1.038, "goal_width": 0.25}
        self.env_info["puck"] = {"radius": 0.03165}
        self.env_info["mallet"] = {"radius": 0.04815}
        self.env_info["n_agents"] = self.n_agents
        self.env_info["robot"] = {
            "n_joints": 7,
            "ee_desired_height": 0.1645,
            "joint_vel_limit": jnp.array(
                [
                    [-85, -85, -100, -75, -130, -135, -135],
                    [85, 85, 100, 75, 130, 135, 135],
                ]
            )
            / 180.0
            * jnp.pi,
            "joint_acc_limit": jnp.array(
                [
                    [-85, -85, -100, -75, -130, -135, -135],
                    [85, 85, 100, 75, 130, 135, 135],
                ]
            )
            / 180.0
            * jnp.pi
            * 10,
            "base_frame": [],
            "universal_height": 0.0645,
            "control_frequency": 50,
        }

        self.env_info["dt"] = self.dt
        self.env_info["robot"]["joint_pos_limit"] = np.array(
            [self.model.joint(f"iiwa_1/joint_{i + 1}").range for i in range(7)]
        ).T
        self.env_info["puck_pos_ids"] = [0, 1, 2]
        self.env_info["puck_vel_ids"] = [3, 4, 5]
        self.env_info["joint_pos_ids"] = [6, 7, 8, 9, 10, 11, 12]
        self.env_info["joint_vel_ids"] = [13, 14, 15, 16, 17, 18, 19]
        if self.n_agents == 2:
            self.env_info["opponent_ee_ids"] = [20, 21, 22]
        else:
            self.env_info["opponent_ee_ids"] = []

        self.robot_model = mujoco.MjModel.from_xml_path(
            os.path.join(os.path.dirname(os.path.abspath(env_path)), "iiwa_only.xml")
        )
        self.robot_model.body("iiwa_1/base").pos = np.zeros(3)
        self.robot_data = mujoco.MjData(self.robot_model)

        robot_mjx_model = mjx.put_model(self.robot_model)
        single_robot_mjx_data = mjx.put_data(self.robot_model, self.robot_data)
        robot_mjx_data = jax.vmap(lambda _, x: x, in_axes=(0, None))(
            jnp.arange(self.batch_size), single_robot_mjx_data
        )

        self.env_info["robot"]["robot_mjx_model"] = robot_mjx_model
        self.env_info["robot"]["robot_mjx_data"] = robot_mjx_data

        frame_T = jnp.eye(4)
        temp = mjx_math.quat_to_mat(
            self.mjx_model.body_quat[self.body_name2id["iiwa_1/base"]]
        )
        frame_T = frame_T.at[:3, :3].set(temp)
        frame_T = frame_T.at[:3, 3].set(
            self.mjx_model.body_pos[self.body_name2id["iiwa_1/base"]]
        )
        self.env_info["robot"]["base_frame"].append(frame_T.copy())

        if self.n_agents == 2:
            temp = mjx_math.quat_to_mat(
                self.mjx_model.body_quat[self.body_name2id["iiwa_2/base"]]
            )
            frame_T = frame_T.at[:3, :3].set(temp)
            frame_T = frame_T.at[:3, 3].set(
                self.mjx_model.body_pos[self.body_name2id["iiwa_2/base"]]
            )
            self.env_info["robot"]["base_frame"].append(frame_T.copy())

        self.rng_key = jax.random.key(0)

        self.u_joint_pos_prev = None
        self.u_joint_vel_prev = jnp.zeros(2 * self.env_info["n_agents"])
        self.u_filter_ratio = 0.273

        self.filter_ratio = 0.274
        self.q_pos_prev = jnp.zeros(self.env_info["robot"]["n_joints"] * self.n_agents)
        self.q_vel_prev = jnp.zeros(self.env_info["robot"]["n_joints"] * self.n_agents)

        self._obs = None

        self._jit_step = jax.jit(self._step, static_argnames=("n_agents"))
        self._jit_reset = jax.jit(self._reset, static_argnames=("n_agents"))
        self._jit_get_obs = jax.jit(self._get_obs)
        self._jit_create_observation = jax.jit(self._create_observation)
        self._jit_modify_observation = jax.jit(self._modify_observation)
        self._jit_is_absorbing = jax.jit(self._is_absorbing)
        self._jit_reward = jax.jit(self._reward)
        self._jit_create_info_dictionary = jax.jit(self._create_info_dictionary)

        # self._jit_step = self._step
        # self._jit_reset = self._reset
        # self._jit_get_obs = self._get_obs
        # self._jit_create_observation = self._create_observation
        # self._jit_modify_observation = self._modify_observation
        # self._jit_is_absorbing = self._is_absorbing
        # self._jit_reward = self._reward
        # self._jit_create_info_dictionary = self._create_info_dictionary

    @property
    def dt(self):
        return self.timestep * self.n_intermediate_steps

    def set_seed(self, seed: int):
        self.rng_key = jax.random.key(seed)

    def step(self, action: jax.Array) -> jax.Array:
        cur_obs = self._obs.copy()

        action = self._preprocess_action(action)
        self._step_init(cur_obs, action)

        for _ in range(self.n_intermediate_steps):
            ctrl_action = self._compute_action(cur_obs, action)

            self.mjx_data, self.u_joint_pos_prev, self.u_joint_vel_prev = (
                self._jit_step(
                    self.mjx_data,
                    self.env_info["robot"]["robot_mjx_model"],
                    self.env_info["robot"]["robot_mjx_data"],
                    self.n_agents,
                    self.u_joint_pos_prev,
                    self.u_joint_vel_prev,
                    self.u_filter_ratio,
                    ctrl_action,
                )
            )

            cur_obs = self._jit_get_obs(self.mjx_data)
            cur_obs = self._jit_create_observation(cur_obs)

        absorbing = self._jit_is_absorbing(cur_obs)
        reward = self._jit_reward(self._obs, action, cur_obs, absorbing)
        info = self._jit_create_info_dictionary(cur_obs)

        self._obs = cur_obs

        return self._jit_modify_observation(cur_obs), reward, absorbing, info

    def reset(self) -> jax.Array:
        self.rng_key, rng_subkey = jax.random.split(self.rng_key)
        self.mjx_data, self.u_joint_pos_prev, self.u_joint_vel_prev = self._jit_reset(
            rng_subkey,
            self.env_info["robot"]["robot_mjx_model"],
            self.env_info["robot"]["robot_mjx_data"],
            self.n_agents,
        )

        obs = self._jit_get_obs(self.mjx_data)
        self._obs = self._jit_create_observation(obs)
        return self._jit_modify_observation(self._obs)

    def _is_absorbing(self, obs: jax.Array) -> jax.Array:
        boundary = (
            jnp.array(
                [self.env_info["table"]["length"], self.env_info["table"]["width"]]
            )
            / 2
        )
        puck_pos, puck_vel = self.get_puck(obs)

        return jnp.logical_or(
            jnp.any(jnp.abs(puck_pos[:, :2]) > boundary, axis=-1),
            jnp.linalg.norm(puck_vel, axis=-1) > 100,
        )

    def _reward(self, obs, action, next_obs, absorbing):
        return jnp.zeros((self.batch_size,))

    def _create_info_dictionary(self, obs: jax.Array) -> dict:
        return [{} for _ in range(self.batch_size)]

    def _get_obs(self, mjx_data: mjx.Data) -> jax.Array:
        puck_pos = mjx_data.qpos[
            :,
            [
                self.joint_name2id["puck_x"],
                self.joint_name2id["puck_y"],
                self.joint_name2id["puck_yaw"],
            ],
        ]
        puck_vel = mjx_data.qvel[
            :,
            [
                self.joint_name2id["puck_x"],
                self.joint_name2id["puck_y"],
                self.joint_name2id["puck_yaw"],
            ],
        ]

        joint_pos = mjx_data.qpos[
            :,
            [
                self.joint_name2id["iiwa_1/joint_1"],
                self.joint_name2id["iiwa_1/joint_2"],
                self.joint_name2id["iiwa_1/joint_3"],
                self.joint_name2id["iiwa_1/joint_4"],
                self.joint_name2id["iiwa_1/joint_5"],
                self.joint_name2id["iiwa_1/joint_6"],
                self.joint_name2id["iiwa_1/joint_7"],
            ],
        ]
        joint_vel = mjx_data.qvel[
            :,
            [
                self.joint_name2id["iiwa_1/joint_1"],
                self.joint_name2id["iiwa_1/joint_2"],
                self.joint_name2id["iiwa_1/joint_3"],
                self.joint_name2id["iiwa_1/joint_4"],
                self.joint_name2id["iiwa_1/joint_5"],
                self.joint_name2id["iiwa_1/joint_6"],
                self.joint_name2id["iiwa_1/joint_7"],
            ],
        ]

        obs = jnp.concatenate(
            [
                puck_pos,
                puck_vel,
                joint_pos,
                joint_vel,
            ],
            axis=-1,
        )

        if self.n_agents == 2:
            opponent_ee_pos = mjx_data.xpos[
                :, self.body_name2id["iiwa_2/striker_joint_link"]
            ]

            opponent_joint_pos = mjx_data.qpos[
                :,
                [
                    self.joint_name2id["iiwa_2/joint_1"],
                    self.joint_name2id["iiwa_2/joint_2"],
                    self.joint_name2id["iiwa_2/joint_3"],
                    self.joint_name2id["iiwa_2/joint_4"],
                    self.joint_name2id["iiwa_2/joint_5"],
                    self.joint_name2id["iiwa_2/joint_6"],
                    self.joint_name2id["iiwa_2/joint_7"],
                ],
            ]
            opponent_joint_vel = mjx_data.qvel[
                :,
                [
                    self.joint_name2id["iiwa_2/joint_1"],
                    self.joint_name2id["iiwa_2/joint_2"],
                    self.joint_name2id["iiwa_2/joint_3"],
                    self.joint_name2id["iiwa_2/joint_4"],
                    self.joint_name2id["iiwa_2/joint_5"],
                    self.joint_name2id["iiwa_2/joint_6"],
                    self.joint_name2id["iiwa_2/joint_7"],
                ],
            ]

            ee_pos = mjx_data.xpos[:, self.body_name2id["iiwa_1/striker_joint_link"]]

            obs = jnp.concatenate(
                [
                    obs,
                    opponent_ee_pos,
                    puck_pos,
                    puck_vel,
                    opponent_joint_pos,
                    opponent_joint_vel,
                    ee_pos,
                ],
                axis=-1,
            )

        return obs

    def _create_observation(self, obs: jax.Array) -> jax.Array:
        return obs

    def _modify_observation(self, obs: jax.Array) -> jax.Array:
        return obs

    def _step(
        self,
        mjx_data: mjx.Data,
        robot_model: mjx.Model,
        robot_data: mjx.Data,
        n_agents: int,
        u_joint_pos_prev: jax.Array,
        u_joint_vel_prev: jax.Array,
        filter_ratio: float,
        action: jax.Array,
    ) -> mjx.Data:

        ctrl = mjx_data.ctrl
        ctrl = ctrl.at[:, self.action_indices].set(action)
        mjx_data = mjx_data.replace(ctrl=ctrl)
        mjx_data, u_joint_pos_prev, u_joint_vel_prev = update(
            self.mjx_model,
            mjx_data,
            robot_model,
            robot_data,
            n_agents,
            u_joint_pos_prev,
            u_joint_vel_prev,
            filter_ratio,
        )
        mjx_data = jax.vmap(mjx.step, in_axes=(None, 0))(self.mjx_model, mjx_data)

        return mjx_data, u_joint_pos_prev, u_joint_vel_prev

    # def _reset_in_step(
    #         self,
    #         rng_key: jax.Array,
    #         mjx_data: mjx.Data,
    #         done: jax.Array,
    #         robot_model: mjx.Model,
    #         robot_data: mjx.Data,
    #         n_agents: int,
    # ):

    #     def _reset_single_mjx_data(i, x):
    #         single_mjx_data: mjx.Data = mjx.make_data(self.mjx_model)
    #         single_mjx_data, u_joint_pos_prev, u_joint_vel_prev = self._setup(
    #             rng_key, single_mjx_data, robot_model, robot_data, n_agents
    #         )
    #         single_mjx_data = mjx.forward(self.mjx_model, single_mjx_data)
    #         return single_mjx_data, u_joint_pos_prev, u_joint_vel_prev

    #     single_mjx_data: mjx.Data = mjx.make_data(self.mjx_model)

    #     # if done[i] is True, set mjx_data[i] to single_mjx_data

    #     mjx_data = jax.vmap(
    #         lambda i, x: jax.lax.cond(
    #             done[i],
    #             lambda: single_mjx_data,
    #             lambda: x,
    #         ),
    #         in_axes=(0, None),
    #     )(jnp.arange(self.batch_size), mjx_data)

    def _reset(
        self,
        rng_key: jax.Array,
        robot_model: mjx.Model,
        robot_data: mjx.Data,
        n_agents: int,
    ) -> mjx.Data:
        single_mjx_data: mjx.Data = mjx.make_data(self.mjx_model)

        mjx_data = jax.vmap(lambda _, x: x, in_axes=(0, None))(
            jnp.arange(self.batch_size), single_mjx_data
        )

        mjx_data, u_joint_pos_prev, u_joint_vel_prev = self._setup(
            rng_key, mjx_data, robot_model, robot_data, n_agents
        )

        mjx_data = jax.vmap(mjx.forward, in_axes=(None, 0))(self.mjx_model, mjx_data)

        return mjx_data, u_joint_pos_prev, u_joint_vel_prev

    def _setup(
        self,
        rng_key: jax.Array,
        mjx_data: mjx.Data,
        robot_model: mjx.Model,
        robot_data: mjx.Data,
        n_agents: int,
    ) -> tuple[mjx.Data, jax.Array, jax.Array]:
        return mjx_data, None, None

    def _preprocess_action(self, action: jax.Array) -> jax.Array:
        return action

    def _step_init(self, obs: jax.Array, action: jax.Array):
        pass

    def _compute_action(self, obs: jax.Array, action: jax.Array) -> jax.Array:
        return action

    def get_puck(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        puck_pos = obs[:, self.env_info["puck_pos_ids"]]
        puck_vel = obs[:, self.env_info["puck_vel_ids"]]

        return puck_pos, puck_vel

    @staticmethod
    @jax.jit
    def _puck_pose_2d_in_robot_frame(
        puck_in: jax.Array, robot_frame: jax.Array
    ) -> jax.Array:
        puck_w = jnp.eye(4)
        puck_w = puck_w.at[:2, 3].set(puck_in[:2])
        puck_w = puck_w.at[:3, :3].set(
            R.from_euler("xyz", [0, 0, puck_in[2]]).as_matrix()
        )

        puck_r = jnp.linalg.inv(robot_frame) @ puck_w
        puck_out = jnp.concatenate(
            [
                puck_r[:2, 3],
                R.from_matrix(puck_r[:3, :3]).as_euler("xyz")[2:3],
            ]
        )

        return puck_out

    @staticmethod
    @jax.jit
    def _puck_vel_2d_in_robot_frame(
        puck_in: jax.Array, robot_frame: jax.Array
    ) -> jax.Array:

        rot_mat = robot_frame[:3, :3]

        vel_lin = jnp.array([*puck_in[:2], 0.0])
        vel_ang = jnp.array([0.0, 0.0, puck_in[2]])

        vel_lin_r = rot_mat.T @ vel_lin
        vel_ang_r = rot_mat.T @ vel_ang

        puck_out = jnp.concatenate([vel_lin_r[:2], vel_ang_r[2:3]])

        return puck_out

    def get_ee(self):
        raise NotImplementedError

    def get_joints(self, obs):
        raise NotImplementedError


if __name__ == "__main__":
    # from jax import config
    # config.update("jax_debug_nans", True)
    from mujoco.viewer import launch_passive, launch

    batch_size = 1

    xml_file = os.path.join(
        os.path.dirname(os.path.abspath(env_path)),
        "single.xml",
    )

    mj_model = mujoco.MjModel.from_xml_path(xml_file)
    mj_data = mujoco.MjData(mj_model)

    mujoco.mj_resetData(mj_model, mj_data)

    mujoco.mj_forward(mj_model, mj_data)

    launch(mj_model, mj_data)

    # mjx_model = mjx.put_model(mj_model)
    # mjx_data = mjx.put_data(mj_model, mj_data)

    # mjx_data = mjx.forward(mjx_model, mjx_data)
    # pass
