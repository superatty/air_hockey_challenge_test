from jax import numpy as jnp
import jax
from brax.envs.base import State
from jax import scipy
from air_hockey_challenge.utils.mjx.kinematics import inverse_kinematics

from mujoco import mjx
from air_hockey_challenge.environments import brax_envs

class PositionControl:
    def __init__(self, p_gain, d_gain, i_gain, interpolation_order=3, *args, **kwargs):
        super(PositionControl, self).__init__(*args, **kwargs)

        self.p_gain = jnp.array(p_gain * self.n_agents)
        self.d_gain = jnp.array(d_gain * self.n_agents)
        self.i_gain = jnp.array(i_gain * self.n_agents)
        self.interpolation_order = interpolation_order

        self.interp_order = (
            interpolation_order
            if type(interpolation_order) is tuple
            else (interpolation_order,)
        )

        self._num_env_joints = len(self.actuator_joint_ids)
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

    def reset(self, rng: jax.Array) -> State:
        state = super().reset(rng)

        all_joint_ids = (
            self.joint_ids
            if self.n_agents == 1
            else jnp.concatenate([self.joint_ids, self.opponent_joint_ids])
        )

        prev_pos = state.pipeline_state.qpos[all_joint_ids]
        prev_vel = state.pipeline_state.qvel[all_joint_ids]
        prev_controller_cmd_pos = state.pipeline_state.qpos[all_joint_ids]
        prev_acc = jnp.zeros_like(prev_vel)
        i_error = jnp.zeros_like(prev_pos)

        state.info.update(
            prev_pos=prev_pos,
            prev_vel=prev_vel,
            prev_acc=prev_acc,
            i_error=i_error,
            prev_controller_cmd_pos=prev_controller_cmd_pos,
            traj=jnp.zeros((self._n_intermediate_steps, 3, self.n_robot_joints * self.n_agents)),
        )

        return state

    def _step_init(self, state: State, action: jax.Array) -> State:
        state = super(PositionControl, self)._step_init(state, action)

        if self.n_agents == 1:
            state, traj = self._create_traj(state, action)
        else:
            state, traj_1 = self._create_traj(state, action[0], 0)
            state, traj_2 = self._create_traj(state, action[1], 1)

            traj = []

            for a1, a2 in zip(traj_1, traj_2):
                traj.append(jnp.hstack([a1, a2]))

            traj = jnp.stack(traj, axis=0)

        state.info.update(
            traj=traj,
            step=0,
        )

        return state

    def _compute_action(
        self, state: State, action: jax.Array
    ) -> tuple[State, jax.Array]:
        cur_pos, cur_vel = self.get_joints(state.info["internal_obs"])

        traj = state.info["traj"]
        step = state.info["step"]

        desired = traj[step]

        state, torque = self._controller(
            state, desired[0], desired[1], desired[2], cur_pos, cur_vel
        )

        state.info.update(
            step=step + 1,
        )

        # jax.debug.print("Step: {step}, Torque: {torque}", step=step, torque=torque)
        # jax.debug.print("Desired vel: {desired_vel}, Current vel: {cur_vel}",
        #                 desired_vel=desired[1], cur_vel=cur_vel)
        # jax.debug.print("Desired vel: {desired_vel}", 
        #                 desired_vel=desired[1])
        # jax.debug.print("Current vel: {cur_vel}", 
        #                 cur_vel=cur_vel)
        # jax.debug.print("")

        return state, torque

    def _controller(
        self,
        state: State,
        desired_pos,
        desired_vel,
        desired_acc,
        current_pos,
        current_vel,
    ):
        state, clipped_pos, clipped_vel = self._enforce_safety_limits(
            state, desired_pos, desired_vel
        )

        i_error = state.info["i_error"]

        pos_error = clipped_pos - current_pos
        vel_error = clipped_vel - current_vel

        i_error += self.i_gain * pos_error * self._timestep
        torque = self.p_gain * pos_error + self.d_gain * vel_error + i_error

        for i in range(self.n_agents):
            robot_joint_ids = jnp.arange(self.n_robot_joints) + self.n_robot_joints * i

            robot_model = self.env_info["robot"]["robot_mjx_model"]
            robot_data = self.env_info["robot"]["robot_mjx_data"]

            robot_data = robot_data.replace(qpos=current_pos[robot_joint_ids])
            robot_data = robot_data.replace(qvel=current_vel[robot_joint_ids])

            acc_ff = desired_acc[robot_joint_ids]

            robot_data = mjx.forward(robot_model, robot_data)
            tau_ff = mjx.mul_m(robot_model, robot_data, acc_ff)

            torque = torque.at[robot_joint_ids].add(tau_ff)
            torque = torque.at[robot_joint_ids].add(robot_data.qfrc_bias)

            torque = torque.at[robot_joint_ids].set(
                jnp.minimum(
                    jnp.maximum(
                        torque[robot_joint_ids],
                        self.robot_model.actuator_ctrlrange[:, 0],
                    ),
                    self.robot_model.actuator_ctrlrange[:, 1],
                )
            )

        state.info.update(
            i_error=i_error,
        )

        return state, torque

    def _enforce_safety_limits(self, state: State, desired_pos, desired_vel):
        pos = state.info["prev_controller_cmd_pos"]
        k = 20

        joint_pos_lim = jnp.tile(
            self.env_info["robot"]["joint_pos_limit"], (1, self.n_agents)
        )
        joint_vel_lim = jnp.tile(
            self.env_info["robot"]["joint_vel_limit"], (1, self.n_agents)
        )

        min_vel = jnp.minimum(
            jnp.maximum(-k * (pos - joint_pos_lim[0]), joint_vel_lim[0]),
            joint_vel_lim[1],
        )

        max_vel = jnp.minimum(
            jnp.maximum(-k * (pos - joint_pos_lim[1]), joint_vel_lim[0]),
            joint_vel_lim[1],
        )

        clipped_vel = jnp.minimum(jnp.maximum(desired_vel, min_vel), max_vel)

        min_pos = pos + min_vel * self._timestep
        max_pos = pos + max_vel * self._timestep

        clipped_pos = jnp.minimum(jnp.maximum(desired_pos, min_pos), max_pos)

        state.info.update(
            prev_controller_cmd_pos=clipped_pos.copy(),
        )

        return state, clipped_pos, clipped_vel

    def _create_traj(self, state, action, i=0):
        if self.interp_order[i] is None:
            return state, action

        return self._interpolate_trajectory(state, action, i)

    def _interpolate_trajectory(self, state, action, i=0):
        interp_order = self.interp_order[i]

        tf = self.dt

        prev_pos = state.info["prev_pos"][
            i * self.n_robot_joints : (i + 1) * self.n_robot_joints
        ]
        prev_vel = state.info["prev_vel"][
            i * self.n_robot_joints : (i + 1) * self.n_robot_joints
        ]
        prev_acc = state.info["prev_acc"][
            i * self.n_robot_joints : (i + 1) * self.n_robot_joints
        ]

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
            A = scipy.linalg.block_diag(*[coef] * self.n_robot_joints)
            # y = results.reshape(-2, order="F")
            y = results.ravel(order="F")
            weights = jnp.linalg.solve(A, y).reshape(
                self.n_robot_joints, interp_order + 1
            )
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

        prev_pos = (
            state.info["prev_pos"]
            .at[i * self.n_robot_joints : (i + 1) * self.n_robot_joints]
            .set(jnp.polyval(weights.T, tf))
        )

        prev_vel = (
            state.info["prev_vel"]
            .at[i * self.n_robot_joints : (i + 1) * self.n_robot_joints]
            .set(jnp.polyval(weights_d.T, tf))
        )

        prev_acc = (
            state.info["prev_acc"]
            .at[i * self.n_robot_joints : (i + 1) * self.n_robot_joints]
            .set(jnp.polyval(weights_dd.T, tf))
        )

        qs = []
        qds = []
        qdds = []

        # for t in jnp.linspace(self._timestep, self.dt, self._n_intermediate_steps):
        #     q = jnp.polyval(weights.T, t)
        #     qd = jnp.polyval(weights_d.T, t)
        #     qdd = jnp.polyval(weights_dd.T, t)

        #     qs.append(q)
        #     qds.append(qd)
        #     qdds.append(qdd)

        ts = jnp.linspace(self._timestep, self.dt, self._n_intermediate_steps)

        def per_t(t):
            return (
            jnp.polyval(weights.T,    t),
            jnp.polyval(weights_d.T,  t),
            jnp.polyval(weights_dd.T, t),
            )

        qs, qds, qdds = jax.lax.map(per_t, ts)

        state.info.update(
            prev_pos=prev_pos,
            prev_vel=prev_vel,
            prev_acc=prev_acc,
        )

        qs = jnp.stack(qs, axis=0)
        qds = jnp.stack(qds, axis=0)
        qdds = jnp.stack(qdds, axis=0)

        traj = jnp.stack([qs, qds, qdds], axis=1)

        return state, traj

    def _preprocess_action(self, state, action):
        state, action = super(PositionControl, self)._preprocess_action(state, action)

        if self.n_agents == 1:
            assert action.shape == self.action_shape[0], (
                f"Unexpected action shape. Expected {self.action_shape[0]} but got"
                f" {action.shape}"
            )
        else:
            for i in range(self.n_agents):
                assert action[i].shape == self.action_shape[i], (
                    f"Unexpected action shape. Expected {self.action_shape[i]} but got"
                    f" {action[i].shape}"
                )

        return state, action


class PositionControlIIWA(PositionControl):
    def __init__(self, *args, **kwargs):
        p_gain = [1500.0, 1500.0, 1200.0, 1200.0, 1000.0, 1000.0, 500.0]
        d_gain = [60, 80, 60, 30, 10, 1, 0.5]
        i_gain = [0, 0, 0, 0, 0, 0, 0]

        super(PositionControlIIWA, self).__init__(
            p_gain=p_gain, d_gain=d_gain, i_gain=i_gain, *args, **kwargs
        )


class IiwaPositionHit(PositionControlIIWA, brax_envs.AirHockeyHit):
    def __init__(
        self,
        interpolation_order=3,
        opponent_agent=None,
        opponent_interp_order=-1,
        *args,
        **kwargs,
    ):
        super().__init__(
            opponent_agent=opponent_agent,
            interpolation_order=(interpolation_order, opponent_interp_order),
            *args,
            **kwargs,
        )

        # Use default agent when none is provided
        if opponent_agent is None:
            self._opponent_agent = self._default_opponent_action

    def reset(self, rng) -> State:
        state = super().reset(rng)

        state.info.update(t=jnp.pi / 2, opponent_prev_joint_pos=self.init_state)
        return state

    def _default_opponent_action(self, state: State):
        t = state.info["t"]
        prev_joint_pos = state.info["opponent_prev_joint_pos"]

        vel = 3
        cart_offset = jnp.array([0.65, 0])

        t += vel * self.dt
        cart_pos = (
            jnp.array([0.1, 0.16]) * jnp.array([jnp.sin(t) * jnp.cos(t), jnp.cos(t)])
            + cart_offset
        )

        success, joint_pos = inverse_kinematics(
            self.env_info["robot"]["robot_mjx_model"],
            self.env_info["robot"]["robot_mjx_data"],
            jnp.concatenate(
                [
                    cart_pos,
                    jnp.array([0.1 + self.env_info["robot"]["universal_height"]]),
                ]
            ),
            initial_q=prev_joint_pos,
        )
        # assert jnp.all(success), "Inverse kinematics failed for opponent agent!"

        joint_vel = (joint_pos - prev_joint_pos) / self.dt

        prev_joint_pos = joint_pos

        state.info.update(
            t=t,
            opponent_prev_joint_pos=prev_joint_pos,
        )

        return state, jnp.vstack([joint_pos, joint_vel])


class IiwaPositionDefend(PositionControlIIWA, brax_envs.AirHockeyDefend):
    pass


class IiwaPositionPrepare(PositionControlIIWA, brax_envs.AirHockeyPrepare):
    pass


class IiwaPositionDouble(PositionControlIIWA, brax_envs.AirHockeyDouble):
    pass


class IiwaPositionTournament(PositionControlIIWA, brax_envs.AirHockeyTournament):
    pass

if __name__ == "__main__":
    import os

    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_triton_gemm_any=True "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
    )

    # jax.config.update("jax_enable_x64", True)

    from time import time
    from etils import epy
    t = time()

    env = IiwaPositionDefend()

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    # jit_reset = env.reset
    # jit_step = env.step

    rng = jax.random.PRNGKey(0)
    state = jit_reset(rng)

    # states = [state.pipeline_state]

    # action = jnp.zeros(env.action_shape[0])
    action = jnp.zeros((2, env.env_info["robot"]["n_joints"]))

    for i in range(100):
        state = jit_step(state, action)
        # qvel = state.obs[jnp.asarray(env.env_info["joint_vel_ids"])]
        # print(f"Step: {i}, qvel: {epy.pretty_repr(qvel)}")

    jax.block_until_ready(state)
    print("Time taken for 100 steps:", time() - t)

    # images = env.render(states)

    # import cv2

    # for img in images:
    #     cv2.imshow("Air Hockey", img[:, :, ::-1])
    #     cv2.waitKey(20)
