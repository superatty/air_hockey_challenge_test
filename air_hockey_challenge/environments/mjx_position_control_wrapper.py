from jax import numpy as jnp
import jax
from jax import scipy
from mujoco import mjx
import numpy as np

from air_hockey_challenge.environments.mjx.mjx_defend import AirHockeyJaxDefend
from air_hockey_challenge.environments.mjx.mjx_double import AirHockeyJaxDouble
from air_hockey_challenge.environments.mjx.mjx_hitting import AirHockeyJaxHit
from air_hockey_challenge.environments.mjx.mjx_prepare import AirHockeyJaxPrepare
from air_hockey_challenge.utils.kinematics import inverse_kinematics

jax.config.update('jax_enable_x64', True)


class JaxPositionControl:
    def __init__(
        self,
        p_gain: list[float],
        d_gain: list[float],
        i_gain: list[float],
        interpolation_order: int = 3,
        *args,
        **kwargs,
    ):
        super(JaxPositionControl, self).__init__(*args, **kwargs)

        self.p_gain = jnp.array(p_gain * self.n_agents)
        self.d_gain = jnp.array(d_gain * self.n_agents)
        self.i_gain = jnp.array(i_gain * self.n_agents)

        self.prev_pos = jnp.zeros((self.batch_size, len(self.action_indices)))
        self.prev_vel = jnp.zeros((self.batch_size, len(self.action_indices)))
        self.prev_acc = jnp.zeros((self.batch_size, len(self.action_indices)))
        self.i_error = jnp.zeros((self.batch_size, len(self.action_indices)))
        self.prev_controller_cmd_pos = jnp.zeros(
            (self.batch_size, len(self.action_indices))
        )

        self.interp_order = (
            interpolation_order
            if type(interpolation_order) is tuple
            else (interpolation_order,)
        )

        self._num_env_joints = len(self.action_indices)
        self.n_robot_joints = self.env_info["robot"]["n_joints"]

        self.action_shape = [None] * self.n_agents

        for i in range(self.n_agents):
            if self.interp_order[i] is None:
                self.action_shape[i] = (
                    int(self.dt / self._timestep),
                    3,
                    self.n_robot_joints,
                )
            elif self.interp_order[i] in [1, 2]:
                self.action_shape[i] = (self.n_robot_joints,)
            elif self.interp_order[i] in [3, 4, -1]:
                self.action_shape[i] = (2, self.n_robot_joints)
            elif self.interp_order[i] == 5:
                self.action_shape[i] = (3, self.n_robot_joints)

        self.traj = None

        self.jerk = jnp.zeros((self.batch_size, self.n_robot_joints))

        self._jit_controller = jax.jit(
            self._controller, static_argnames=("n_agents", "n_robot_joints")
        )
        self._jit_interpolate_trajectory = jax.jit(
            jax.vmap(
                self._interpolate_trajectory, in_axes=(None, None, None, 0, 0, 0, 0, 0)
            ),
            static_argnames=("interp_order", "n_robot_joints"),
        )
        # self._jit_controller = self._controller
        # self._jit_interpolate_trajectory = jax.vmap(
        #     self._interpolate_trajectory, in_axes=(None, None, None, 0, 0, 0, 0, 0)
        # )

    def _controller(
        self,
        env_info: dict,
        n_agents: int,
        n_robot_joints: int,
        desired_pos: jax.Array,
        desired_vel: jax.Array,
        desired_acc: jax.Array,
        current_pos: jax.Array,
        current_vel: jax.Array,
        prev_controller_cmd_pos: jax.Array,
        i_error: jax.Array,
    ):
        robot_model = env_info["robot"]["robot_mjx_model"]
        robot_data = env_info["robot"]["robot_mjx_data"]

        clipped_pos, clipped_vel = self._enforce_safety_limits(
            desired_pos, desired_vel, prev_controller_cmd_pos
        )

        prev_controller_cmd_pos = clipped_pos.copy()

        pos_error = clipped_pos - current_pos
        vel_error = clipped_vel - current_vel
        i_error += self.i_gain * pos_error * self._timestep

        torque = self.p_gain * pos_error + self.d_gain * vel_error + i_error

        for i in range(n_agents):
            robot_joint_ids = jnp.arange(n_robot_joints) + self.n_robot_joints * i

            robot_data = robot_data.replace(qpos=current_pos[:, robot_joint_ids])
            robot_data = robot_data.replace(qvel=current_vel[:, robot_joint_ids])

            acc_ff = desired_acc[:, robot_joint_ids]

            robot_data = jax.vmap(mjx.forward, in_axes=(None, 0))(
                robot_model, robot_data
            )
            tau_ff = jax.vmap(mjx.mul_m, in_axes=(None, 0, 0))(
                robot_model, robot_data, acc_ff
            )

            torque = torque.at[:, robot_joint_ids].add(tau_ff)
            torque = torque.at[:, robot_joint_ids].add(robot_data.qfrc_bias)

            torque = torque.at[:, robot_joint_ids].set(
                jnp.minimum(
                    jnp.maximum(
                        torque[:, robot_joint_ids],
                        self.robot_model.actuator_ctrlrange[:, 0],
                    ),
                    self.robot_model.actuator_ctrlrange[:, 1],
                )
            )

        return torque, prev_controller_cmd_pos, i_error

    def reset(self) -> jax.Array:
        obs = super(JaxPositionControl, self).reset()

        self.prev_pos = self.mjx_data.qpos[:, self.action_indices]
        self.prev_vel = self.mjx_data.qvel[:, self.action_indices]
        self.prev_acc = jnp.zeros((self.batch_size, len(self.action_indices)))
        self.i_error = jnp.zeros((self.batch_size, len(self.action_indices)))
        self.prev_controller_cmd_pos = self.mjx_data.qpos[:, self.action_indices]

        return obs

    def _enforce_safety_limits(
        self,
        desired_pos: jax.Array,
        desired_vel: jax.Array,
        prev_control_cmd_pos: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        k = 20

        joint_pos_lim = jnp.tile(
            self.env_info["robot"]["joint_pos_limit"], (1, self.n_agents)
        )
        joint_vel_lim = jnp.tile(
            self.env_info["robot"]["joint_vel_limit"], (1, self.n_agents)
        )

        min_vel = jnp.minimum(
            jnp.maximum(
                -k * (prev_control_cmd_pos - joint_pos_lim[0]), joint_vel_lim[0]
            ),
            joint_vel_lim[1],
        )

        max_vel = jnp.minimum(
            jnp.maximum(
                -k * (prev_control_cmd_pos - joint_pos_lim[1]), joint_vel_lim[0]
            ),
            joint_vel_lim[1],
        )

        clipped_vel = jnp.minimum(jnp.maximum(desired_vel, min_vel), max_vel)

        min_pos = prev_control_cmd_pos + min_vel * self._timestep
        max_pos = prev_control_cmd_pos + max_vel * self._timestep

        clipped_pos = jnp.minimum(jnp.maximum(desired_pos, min_pos), max_pos)

        return clipped_pos, clipped_vel

    def _step_init(self, obs, action):
        super(JaxPositionControl, self)._step_init(obs, action)

        if self.n_agents == 1:
            self.traj = self._create_traj(self.interp_order[0], action, 0)
        else:

            def _traj():
                traj_1 = self._create_traj(self.interp_order[0], action[0], 0)
                traj_2 = self._create_traj(self.interp_order[1], action[1], 1)

                # for i in range(self.n_intermediate_steps):
                #     yield jnp.concatenate(
                #         [traj_1[0][i], traj_2[0][i]], axis=-1
                #     ), jnp.concatenate(
                #         [traj_1[1][i], traj_2[1][i]], axis=-1
                #     ), jnp.concatenate(
                #         [traj_1[2][i], traj_2[2][i]], axis=-1
                #     )

                for a1, a2 in zip(traj_1, traj_2):
                    yield jnp.concatenate([a1[0], a2[0]], axis=-1), jnp.concatenate(
                        [a1[1], a2[1]], axis=-1
                    ), jnp.concatenate([a1[2], a2[2]], axis=-1)

            self.traj = _traj()

    def _create_traj(self, interp_order, action, i=0):
        n_robot_joints = self.env_info["robot"]["n_joints"]

        if interp_order is None:
            return iter(action)
        (
            new_prev_pos,
            new_prev_vel,
            new_prev_acc,
            new_jerk,
            qs,
            qds,
            qdds,
        ) = self._jit_interpolate_trajectory(
            self.env_info,
            n_robot_joints,
            interp_order,
            action,
            self.prev_pos[:, i * n_robot_joints : (i + 1) * n_robot_joints],
            self.prev_vel[:, i * n_robot_joints : (i + 1) * n_robot_joints],
            self.prev_acc[:, i * n_robot_joints : (i + 1) * n_robot_joints],
            self.jerk[:, i * n_robot_joints : (i + 1) * n_robot_joints],
        )

        self.prev_pos = self.prev_pos.at[
            :, i * n_robot_joints : (i + 1) * n_robot_joints
        ].set(new_prev_pos)
        self.prev_vel = self.prev_vel.at[
            :, i * n_robot_joints : (i + 1) * n_robot_joints
        ].set(new_prev_vel)
        self.prev_acc = self.prev_acc.at[
            :, i * n_robot_joints : (i + 1) * n_robot_joints
        ].set(new_prev_acc)
        self.jerk = self.jerk.at[:, i * n_robot_joints : (i + 1) * n_robot_joints].set(
            new_jerk
        )

        return iter(zip(qs, qds, qdds))

    def _compute_action(self, obs, action):
        cur_pos, cur_vel = jax.jit(self.get_joints)(obs)

        desired_pos, desired_vel, desired_acc = next(self.traj)

        torque, self.prev_controller_cmd_pos, self.i_error = self._jit_controller(
            self.env_info,
            self.env_info["n_agents"],
            self.env_info["robot"]["n_joints"],
            desired_pos,
            desired_vel,
            desired_acc,
            cur_pos,
            cur_vel,
            self.prev_controller_cmd_pos,
            self.i_error,
        )

        return torque

    def _preprocess_action(self, action):
        action = super(JaxPositionControl, self)._preprocess_action(action)

        if self.n_agents == 1:
            assert action.shape[1:] == self.action_shape[0], (
                f"Unexpected action shape. Expected {self.action_shape[0]} but got"
                f" {action.shape[1:]}"
            )
        else:
            for i in range(self.n_agents):
                assert action[i].shape[1:] == self.action_shape[i], (
                    f"Unexpected action shape. Expected {self.action_shape[i]} but got"
                    f" {action[i].shape[1:]}"
                )

        return action

    def _interpolate_trajectory(
        self,
        env_info,
        n_robot_joints,
        interp_order,
        action,
        prev_pos,
        prev_vel,
        prev_acc,
        jerk,
    ):
        tf = env_info["dt"]

        if interp_order == 1 and action.ndim == 1:
            coef = jnp.array([[1, 0], [1, tf]])
            results = jnp.vstack([prev_pos, action])
        elif interp_order == 2 and action.ndim == 1:
            coef = jnp.array([[1, 0, 0], [1, tf, tf**2], [0, 1, 0]])
            if jnp.linalg.norm(action - prev_pos) < 1e-3:
                prev_vel = jnp.zeros_like(prev_vel)
            results = jnp.vstack([prev_pos, action, prev_vel])
        elif interp_order == 3 and action.shape[0] == 2:
            coef = jnp.array(
                [
                    [1, 0, 0, 0],
                    [1, tf, tf**2, tf**3],
                    [0, 1, 0, 0],
                    [0, 1, 2 * tf, 3 * tf**2],
                ]
            )
            results = jnp.vstack([prev_pos, action[0], prev_vel, action[1]])
        elif interp_order == 4 and action.shape[0] == 2:
            coef = jnp.array(
                [
                    [1, 0, 0, 0, 0],
                    [1, tf, tf**2, tf**3, tf**4],
                    [0, 1, 0, 0, 0],
                    [0, 1, 2 * tf, 3 * tf**2, 4 * tf**3],
                    [0, 0, 2, 0, 0],
                ]
            )
            results = jnp.vstack([prev_pos, action[0], prev_vel, action[1], prev_acc])
        elif interp_order == 5 and action.shape[0] == 3:
            coef = jnp.array(
                [
                    [1, 0, 0, 0, 0, 0],
                    [1, tf, tf**2, tf**3, tf**4, tf**5],
                    [0, 1, 0, 0, 0, 0],
                    [0, 1, 2 * tf, 3 * tf**2, 4 * tf**3, 5 * tf**4],
                    [0, 0, 2, 0, 0, 0],
                    [0, 0, 2, 6 * tf, 12 * tf**2, 20 * tf**3],
                ]
            )
            results = jnp.vstack(
                [prev_pos, action[0], prev_vel, action[1], prev_acc, action[2]]
            )
        elif interp_order == -1:
            # Interpolate position and velocity linearly
            pass
        else:
            raise ValueError(
                "Undefined interpolator order or the action dimension does not match!"
            )

        if interp_order > 0:
            A = scipy.linalg.block_diag(*[coef] * n_robot_joints)
            # y = results.reshape(-2, order="F")
            y = results.ravel(order="F")
            weights = jnp.linalg.solve(A, y).reshape(n_robot_joints, interp_order + 1)
            weights = weights[:, ::-1]
            weights_d = jnp.apply_along_axis(jnp.polyder, 1, weights)
            weights_dd = jnp.apply_along_axis(jnp.polyder, 1, weights_d)
        elif interp_order == -1:
            weights = jnp.vstack([prev_pos, (action[0] - prev_pos) / self.dt]).T[
                :, ::-1
            ]
            weights_d = jnp.vstack([prev_vel, (action[1] - prev_vel) / self.dt]).T[
                :, ::-1
            ]
            weights_dd = jnp.apply_along_axis(jnp.polyder, 1, weights_d)

        if interp_order in [3, 4, 5]:
            jerk = (
                jnp.abs(weights_dd[:, -2])
                + jnp.abs(weights_dd[:, -1] - prev_acc) / self._timestep
            )

        else:
            jerk = jnp.ones_like(prev_acc) * jnp.inf

        prev_pos = jnp.polyval(weights.T, tf)
        prev_vel = jnp.polyval(weights_d.T, tf)
        prev_acc = jnp.polyval(weights_dd.T, tf)

        qs = []
        qds = []
        qdds = []

        # TODO yield was used here, but not sure how to implement it in jax
        for t in jnp.linspace(self._timestep, self.dt, self.n_intermediate_steps):
            q = jnp.polyval(weights.T, t)
            qd = jnp.polyval(weights_d.T, t)
            qdd = jnp.polyval(weights_dd.T, t)

            qs.append(q)
            qds.append(qd)
            qdds.append(qdd)

        return prev_pos, prev_vel, prev_acc, jerk, qs, qds, qdds


class PositionControlIIWA(JaxPositionControl):
    def __init__(self, *args, **kwargs):
        p_gain = [1500.0, 1500.0, 1200.0, 1200.0, 1000.0, 1000.0, 500.0]
        d_gain = [60, 80, 60, 30, 10, 1, 0.5]
        i_gain = [0, 0, 0, 0, 0, 0, 0]

        super(PositionControlIIWA, self).__init__(
            p_gain=p_gain, d_gain=d_gain, i_gain=i_gain, *args, **kwargs
        )


class IiwaPositionHit(PositionControlIIWA, AirHockeyJaxHit):
    def __init__(
        self,
        batch_size: int,
        interpolation_order: int,
        opponent_agent=None,
        opponent_interp_order=-1,
        *args,
        **kwargs,
    ):
        super(IiwaPositionHit, self).__init__(
            batch_size=batch_size,
            interpolation_order=(interpolation_order, opponent_interp_order),
            *args,
            **kwargs,
        )

        # Use default agent when none is provided
        if opponent_agent is None:
            # self._opponent_agent_gen = self._default_opponent_action_gen()
            # self._opponent_agent = lambda: jnp.repeat(
            #     next(self._opponent_agent_gen)[jnp.newaxis, ...],
            #     self.batch_size,
            #     axis=0,
            # )

            self._opponent_agent = self._default_opponent_agent

    def reset(self) -> jax.Array:
        # self._opponent_agent_gen = self._default_opponent_action_gen()
        self.t = jnp.pi / 2
        self.prev_joint_pos = self.init_state

        return super().reset()

    def _default_opponent_agent(self):
        opponent_action, self.t, self.prev_joint_pos = (
            self._default_opponent_action_gen(self.t, self.prev_joint_pos)
        )

        return jnp.array(
            np.repeat(opponent_action[np.newaxis, ...], self.batch_size, axis=0)
        )

    def _default_opponent_action_gen(self, t, prev_joint_pos):
        vel = 3
        cart_offset = np.array([0.65, 0])

        t += vel * self.dt
        cart_pos = (
            np.array([0.1, 0.16]) * np.array([np.sin(t) * np.cos(t), np.cos(t)])
            + cart_offset
        )

        success, joint_pos = inverse_kinematics(
            self.robot_model,
            self.robot_data,
            np.concatenate(
                [cart_pos, [0.1 + self.env_info["robot"]["universal_height"]]]
            ),
            initial_q=prev_joint_pos,
        )
        assert success

        joint_vel = (joint_pos - prev_joint_pos) / self.dt

        prev_joint_pos = joint_pos

        return np.vstack([joint_pos, joint_vel]), t, prev_joint_pos


class IiwaPositionDefend(PositionControlIIWA, AirHockeyJaxDefend):
    pass


class IiwaPositionPrepare(PositionControlIIWA, AirHockeyJaxPrepare):
    pass


class IiwaPositionDouble(PositionControlIIWA, AirHockeyJaxDouble):
    pass


if __name__ == "__main__":
    # from mujoco.viewer import launch_passive
    import tqdm

    batch_size = 4

    env = IiwaPositionHit(batch_size=batch_size, interpolation_order=3)

    # action = (
    #     jax.random.uniform(
    #         jax.random.key(0), shape=(batch_size, 7), minval=-1, maxval=1
    #     )
    #     * 8
    # )
    action = jnp.zeros((batch_size, 2, 7))

    # data = mjx.get_data(env.model, env.mjx_data)
    # viewer = launch_passive(env.model, data[0])

    for _ in range(1):
        obs = env.reset()

        # mjx.get_data_into(data, env.model, env.mjx_data)
        # viewer.sync()

        for i in tqdm.tqdm(range(50)):
            obs = env.step(action)

            # if (i + 1) % 10 == 0:
            #     pass

            # mjx.get_data_into(data, env.model, env.mjx_data)
            # viewer.sync()

    # viewer.close()
