import copy

import jax.numpy as jnp
from typing import Dict
import jax

from air_hockey_challenge.utils.mjx.kinematics import forward_kinematics, jacobian


class ConstraintMjx:
    def __init__(self, env_info: Dict, output_dim, **kwargs):
        """
        Constructor

        Args
        ----
        env_info: Dict
            A dictionary contains information about the environment;
        output_dim: int
            The output dimension of the constraints.
        **kwargs: dict
            A dictionary contains agent related information.
        """
        self._env_info = env_info
        self._name = None

        self.output_dim = output_dim

        self._fun_value = jnp.zeros(self.output_dim)
        self._jac_value = jnp.zeros(
            (self.output_dim, 2 * env_info["robot"]["n_joints"])
        )
        self._q_prev = None
        self._dq_prev = None

    @property
    def name(self):
        """
        The name of the constraints

        """
        return self._name

    def fun(self, q, dq):
        """
        The function of the constraint.

        Args
        ----
        q: jnp.ndarray, (num_joints,)
            The joint position of the robot
        dq: jnp.ndarray, (num_joints,)
            The joint velocity of the robot

        Returns
        -------
        jnp.ndarray, (out_dim,):
            The value computed by the constraints function.
        """

        is_same_q = False
        if q is not None and self._q_prev is not None:
            is_same_q = jnp.all(jnp.equal(q, self._q_prev))

        is_same_dq = False
        if dq is not None and self._dq_prev is not None:
            is_same_dq = jnp.all(jnp.equal(dq, self._dq_prev))

        if is_same_q and is_same_dq:
            return self._fun_value
        else:
            # self._jacobian(q, dq)
            return self._fun(q, dq)

    def jacobian(self, q, dq):
        """
        Jacobian is the derivative of the constraint function w.r.t the robot joint position and velocity.

        Args
        ----
        q: jnp.ndarray, (num_joints,)
            The joint position of the robot
        dq: jnp.ndarray, (num_joints,)
            The joint velocity of the robot

        Returns
        -------
        jnp.ndarray, (dim_output, num_joints * 2):
            The flattened jacobian of the constraint function J = [dc / dq, dc / dq_dot]

        """

        is_same_q = False
        if q is not None and self._q_prev is not None:
            is_same_q = jnp.all(jnp.equal(q, self._q_prev))

        is_same_dq = False
        if dq is not None and self._dq_prev is not None:
            is_same_dq = jnp.all(jnp.equal(dq, self._dq_prev))

        if is_same_q and is_same_dq:
            return self._jac_value
        else:
            self._fun(q, dq)
            return self._jacobian(q, dq)

    def _fun(self, q, dq):
        raise NotImplementedError

    def _jacobian(self, q, dq):
        raise NotImplementedError


class ConstraintList:
    def __init__(self):
        self.constraints: Dict[str, ConstraintMjx] = {}

    def keys(self):
        return self.constraints.keys()

    def get(self, key: str) -> ConstraintMjx:
        return self.constraints.get(key)

    def add(self, c: ConstraintMjx) -> "ConstraintList":
        self.constraints[c.name] = c
        return self

    def delete(self, name: str):
        del self.constraints[name]

    def fun(self, q: jnp.ndarray, dq: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        return {key: c.fun(q, dq) for key, c in self.constraints.items()}

    def jacobian(self, q: jnp.ndarray, dq: jnp.ndarray) -> Dict[str, jnp.ndarray]:
        return {key: c.jacobian(q, dq) for key, c in self.constraints.items()}


class JointPositionConstraint(ConstraintMjx):
    def __init__(self, env_info, **kwargs):
        super().__init__(
            env_info, output_dim=2 * env_info["robot"]["n_joints"], **kwargs
        )
        self.joint_limits = self._env_info["robot"]["joint_pos_limit"] * 0.95
        self._name = "joint_pos_constr"

    def _fun(self, q: jnp.ndarray, dq: jnp.ndarray) -> jnp.ndarray:
        upper = q - self.joint_limits[1]
        lower = self.joint_limits[0] - q

        return jnp.concatenate((upper, lower), axis=0)

    def _jacobian(self, q: jnp.ndarray, dq: jnp.ndarray) -> jnp.ndarray:
        n = self._env_info["robot"]["n_joints"]
        eye = jnp.eye(n)
        zeros = jnp.zeros_like(eye)
        # [ I  0 ]
        # [-I  0 ]
        return jnp.block([[eye, zeros], [-eye, zeros]])


# TODO check JAX compatibility
class JointVelocityConstraint(ConstraintMjx):
    def __init__(self, env_info, **kwargs):
        super().__init__(
            env_info, output_dim=2 * env_info["robot"]["n_joints"], **kwargs
        )
        self.joint_limits = self._env_info["robot"]["joint_vel_limit"] * 0.95
        self._name = "joint_vel_constr"

    def _fun(self, q: jnp.ndarray, dq: jnp.ndarray) -> jnp.ndarray:
        upper = dq - self.joint_limits[1]
        lower = self.joint_limits[0] - dq

        return jnp.concatenate((upper, lower), axis=0)

    def _jacobian(self, q: jnp.ndarray, dq: jnp.ndarray) -> jnp.ndarray:
        n = self._env_info["robot"]["n_joints"]
        eye = jnp.eye(n)
        zeros = jnp.zeros_like(eye)
        # [ 0  I ]
        # [ 0 -I ]
        return jnp.block([[zeros, eye], [zeros, -eye]])


class EndEffectorConstraint(ConstraintMjx):
    def __init__(self, env_info, **kwargs):
        # 1 Dimension on x direction: x > x_lb
        # 2 Dimension on y direction: y > y_lb, y < y_ub
        # 2 Dimension on z direction: z > z_lb, z < z_ub
        super().__init__(env_info, output_dim=5, **kwargs)
        self._name = "ee_constr"
        tolerance = 0.02

        self.robot_model = copy.deepcopy(self._env_info["robot"]["robot_mjx_model"])
        self.robot_data = copy.deepcopy(self._env_info["robot"]["robot_mjx_data"])

        self.x_lb = -self._env_info["robot"]["base_frame"][0][0, 3] - (
            self._env_info["table"]["length"] / 2 - self._env_info["mallet"]["radius"]
        )
        self.y_lb = -(
            self._env_info["table"]["width"] / 2 - self._env_info["mallet"]["radius"]
        )
        self.y_ub = (
            self._env_info["table"]["width"] / 2 - self._env_info["mallet"]["radius"]
        )
        self.z_lb = self._env_info["robot"]["ee_desired_height"] - tolerance
        self.z_ub = self._env_info["robot"]["ee_desired_height"] + tolerance

    def _fun(self, q, dq):
        ee_pos, _ = forward_kinematics(self.robot_model, self.robot_data, q)

        fun_value = jnp.array(
            [
                -ee_pos[0] + self.x_lb,
                -ee_pos[1] + self.y_lb,
                ee_pos[1] - self.y_ub,
                -ee_pos[2] + self.z_lb,
                ee_pos[2] - self.z_ub,
            ]
        )
        return fun_value

    def _jacobian(self, q, dq):
        jac = jacobian(self.robot_model, self.robot_data, q)

        dc_dx = jnp.array(
            [
                [-1, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, -1.0],
                [0.0, 0.0, 1.0],
            ]
        )

        n = self._env_info["robot"]["n_joints"]

        jac_pos = jac[:3, :n]  # (3, n)
        projected = dc_dx @ jac_pos  # (5, n)
        zeros = jnp.zeros((5, n))
        cat = jnp.concatenate([projected, zeros], axis=1)  # (5, 2n)
        pad_bottom = jnp.zeros((self.output_dim - 5, 2 * n))
        return jnp.concatenate([cat, pad_bottom], axis=0)  # (output_dim, 2n)


class LinkConstraint(ConstraintMjx):
    def __init__(self, env_info, **kwargs):
        # 1 Dimension: wrist_z > minimum_height
        # 2 Dimension: elbow_z > minimum_height
        super().__init__(env_info, output_dim=2, **kwargs)
        self._name = "link_constr"

        self.robot_model = copy.deepcopy(self._env_info["robot"]["robot_mjx_model"])
        self.robot_data = copy.deepcopy(self._env_info["robot"]["robot_mjx_data"])

        self.z_lb = 0.25

    def _fun(self, q, dq):
        wrist_pos, _ = forward_kinematics(
            self.robot_model, self.robot_data, q, link="7"
        )
        elbow_pos, _ = forward_kinematics(
            self.robot_model, self.robot_data, q, link="4"
        )

        fun_value = jnp.array([-wrist_pos[2] + self.z_lb, -elbow_pos[2] + self.z_lb])

        return fun_value

    def _jacobian(self, q, dq):
        jac_wrist = jacobian(self.robot_model, self.robot_data, q, link="7")
        jac_elbow = jacobian(self.robot_model, self.robot_data, q, link="4")

        n = self._env_info["robot"]["n_joints"]

        jac_rows = jnp.vstack(
            [
                -jac_wrist[2, :n],
                -jac_elbow[2, :n],
            ]
        )  # shape: (2, n)

        zeros_rows = jnp.zeros((2, n))
        jac_rows = jnp.concatenate([jac_rows, zeros_rows], axis=1)  # (2, 2n)

        pad_rows = jnp.zeros((self.output_dim - 2, 2 * n))
        jac_full = jnp.concatenate([jac_rows, pad_rows], axis=0)

        return jac_full  # shape: (output_dim, 2n)


if __name__ == "__main__":
    from air_hockey_challenge.environments.brax_envs import AirHockeyBase

    env = AirHockeyBase(n_agents=1)
    env_info = env.env_info

    joint_pos_constr = JointPositionConstraint(env_info)
    joint_vel_constr = JointVelocityConstraint(env_info)
    ee_constr = EndEffectorConstraint(env_info)
    link_constr = LinkConstraint(env_info)

    # Dummy state
    q = jnp.zeros(env_info["robot"]["n_joints"])
    dq = jnp.ones(env_info["robot"]["n_joints"]) * 0.5

    print("JointPositionConstraint fun:")
    jp_fun = joint_pos_constr.fun(q, dq)
    print(jp_fun)
    print("Shape:", jp_fun.shape)

    print("JointPositionConstraint jacobian:")
    jp_jac = joint_pos_constr.jacobian(q, dq)
    print(jp_jac)
    print("Shape:", jp_jac.shape)

    print("\nJointVelocityConstraint fun:")
    jv_fun = joint_vel_constr.fun(q, dq)
    print(jv_fun)
    print("Shape:", jv_fun.shape)

    print("JointVelocityConstraint jacobian:")
    jv_jac = joint_vel_constr.jacobian(q, dq)
    print(jv_jac)
    print("Shape:", jv_jac.shape)

    print("\nEndEffectorConstraint fun:")
    ee_fun = ee_constr.fun(q, dq)
    print(ee_fun)
    print("Shape:", ee_fun.shape)

    print("EndEffectorConstraint jacobian:")
    ee_jac = ee_constr.jacobian(q, dq)
    print(ee_jac)
    print("Shape:", ee_jac.shape)

    print("\nLinkConstraint fun:")
    lc_fun = link_constr.fun(q, dq)
    print(lc_fun)
    print("Shape:", lc_fun.shape)

    print("LinkConstraint jacobian:")
    lc_jac = link_constr.jacobian(q, dq)
    print(lc_jac)
    print("Shape:", lc_jac.shape)
