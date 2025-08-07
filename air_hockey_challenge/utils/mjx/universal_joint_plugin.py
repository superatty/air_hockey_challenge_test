import einops
from jax import numpy as jnp
import jax

from mujoco import mjx
import mujoco
from air_hockey_challenge.utils.mjx.kinematics import forward_kinematics

def reset(
    env_model: mjx.Model,
    env_data: mjx.Data,
    robot_model: mjx.Model,
    robot_data: mjx.Data,
    n_agents: int,
    filter_ratio: float,
) -> tuple[mjx.Data, jnp.ndarray, jnp.ndarray]:

    env_data, u_joint_pos_prev, u_joint_vel_prev, u_joint_pos_des = (
        _control_universal_joint(
            env_model,
            env_data,
            robot_model,
            robot_data,
            n_agents,
            None,
            None,
            filter_ratio,
        )
    )

    universal_joint_ids = _get_universal_joint_ids(env_model, n_agents)
    universal_joint_ctrl_ids = _get_universal_joint_ctrl_ids(env_model, n_agents)

    u_joint_vel_prev = env_data.qvel[:, universal_joint_ids]

    qpos = env_data.qpos
    qpos = qpos.at[:, universal_joint_ctrl_ids].set(u_joint_pos_des)
    env_data = env_data.replace(qpos=qpos)

    return env_data, u_joint_pos_prev, u_joint_vel_prev

def update(
    env_model: mjx.Model,
    env_data: mjx.Data,
    robot_model: mjx.Model,
    robot_data: mjx.Data,
    n_agents: int,
    u_joint_pos_prev: jnp.ndarray,
    u_joint_vel_prev: jnp.ndarray,
    filter_ratio: float,
) -> tuple[mjx.Data, jnp.ndarray, jnp.ndarray]:
    return _control_universal_joint(
        env_model,
        env_data,
        robot_model,
        robot_data,
        n_agents,
        u_joint_pos_prev,
        u_joint_vel_prev,
        filter_ratio,
    )[:-1]


def _get_universal_joint_ids(env_model: mjx.Model, n_agents: int) -> list[int]:
    universal_joint_ids = []
    universal_joint_ids += [
        mjx.name2id(env_model, mujoco.mjtObj.mjOBJ_JOINT, "iiwa_1/striker_joint_1"),
        mjx.name2id(env_model, mujoco.mjtObj.mjOBJ_JOINT, "iiwa_1/striker_joint_2"),
    ]
    if n_agents >= 2:
        universal_joint_ids += [
            mjx.name2id(env_model, mujoco.mjtObj.mjOBJ_JOINT, "iiwa_2/striker_joint_1"),
            mjx.name2id(env_model, mujoco.mjtObj.mjOBJ_JOINT, "iiwa_2/striker_joint_2"),
        ]

    return universal_joint_ids


def _get_universal_joint_ctrl_ids(env_model: mjx.Model, n_agents: int) -> list[int]:
    universal_joint_ids = []
    universal_joint_ids += [
        mjx.name2id(env_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "iiwa_1/striker_joint_1"),
        mjx.name2id(env_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "iiwa_1/striker_joint_2"),
    ]
    if n_agents >= 2:
        universal_joint_ids += [
            mjx.name2id(
                env_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "iiwa_2/striker_joint_1"
            ),
            mjx.name2id(
                env_model, mujoco.mjtObj.mjOBJ_ACTUATOR, "iiwa_2/striker_joint_2"
            ),
        ]

    return universal_joint_ids


def _get_actuator_joint_ids(env_model: mjx.Model, n_agents: int) -> list[int]:

    action_spec = [
        "iiwa_1/joint_1",
        "iiwa_1/joint_2",
        "iiwa_1/joint_3",
        "iiwa_1/joint_4",
        "iiwa_1/joint_5",
        "iiwa_1/joint_6",
        "iiwa_1/joint_7",
    ]

    if n_agents >= 2:
        action_spec += [
            "iiwa_2/joint_1",
            "iiwa_2/joint_2",
            "iiwa_2/joint_3",
            "iiwa_2/joint_4",
            "iiwa_2/joint_5",
            "iiwa_2/joint_6",
            "iiwa_2/joint_7",
        ]

    actuator_joint_ids = [
        mjx.name2id(env_model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in action_spec
    ]

    return actuator_joint_ids


def _control_universal_joint(
    env_model: mjx.Model,
    env_data: mjx.Data,
    robot_model: mjx.Model,
    robot_data: mjx.Data,
    n_agents: int,
    u_joint_pos_prev: jnp.ndarray,
    u_joint_vel_prev: jnp.ndarray,
    filter_ratio: float,
) -> tuple[mjx.Data, jnp.ndarray, jnp.ndarray]:
    actuator_joint_ids = _get_actuator_joint_ids(env_model, n_agents)
    universal_joint_ids = _get_universal_joint_ids(env_model, n_agents)

    u_joint_pos_des = _compute_universal_joint(
        env_data,
        robot_model,
        robot_data,
        n_agents,
        actuator_joint_ids,
        u_joint_pos_prev,
    )
    u_joint_pos_prev = env_data.qpos[:, universal_joint_ids]

    if u_joint_vel_prev is not None:
        u_joint_vel_prev = (
            filter_ratio * env_data.qvel[:, universal_joint_ids]
            + (1 - filter_ratio) * u_joint_vel_prev
        )
    else:
        u_joint_vel_prev = env_data.qvel[:, universal_joint_ids]

    Kp = 4
    Kd = 0.31
    torque = Kp * (u_joint_pos_des - u_joint_pos_prev) - Kd * u_joint_vel_prev

    ctrl = env_data.ctrl
    ctrl = ctrl.at[:, universal_joint_ids].set(torque)
    env_data = env_data.replace(ctrl=ctrl)

    return env_data, u_joint_pos_prev, u_joint_vel_prev, u_joint_pos_des


def _compute_universal_joint(
    env_data: mjx.Data,
    robot_model: mjx.Model,
    robot_data: mjx.Data,
    n_agents: int,
    actuator_joint_ids: list,
    u_joint_pos_prev: jnp.ndarray,
) -> jnp.ndarray:
    batch_size = env_data.qpos.shape[0]
    u_joint_pos_des = jnp.zeros((batch_size, 2 * n_agents))

    for i in range(n_agents):
        q = env_data.qpos[:, actuator_joint_ids[i * 7 : (i + 1) * 7]]

        pos, rot_mat = forward_kinematics(
            robot_model,
            robot_data,
            q,
        )

        v_x = rot_mat[:, :, 0]
        v_y = rot_mat[:, :, 1]

        # The desired position of the x-axis is the cross product of the desired z (0, 0, 1).T
        # and the current y-axis. (0, 0, 1).T x v_y
        x_desired = jnp.array([-v_y[:, 1], v_y[:, 0], jnp.zeros(batch_size)]).T

        # Find the signed angle from the current to the desired x-axis around the y-axis
        # https://stackoverflow.com/questions/5188561/signed-angle-between-two-3d-vectors-with-same-origin-within-the-same-plane
        q1 = jnp.arctan2(
            jnp.einsum("bi, bi -> b", jax.vmap(_cross_3d)(v_x, x_desired), v_y),
            jnp.einsum("bi, bi -> b", v_x, x_desired),
        )

        if u_joint_pos_prev is not None:
            q1 = jnp.where(q1 - u_joint_pos_prev[:, 0] > jnp.pi, q1 - jnp.pi * 2, q1)
            q1 = jnp.where(q1 - u_joint_pos_prev[:, 0] < -jnp.pi, q1 + jnp.pi * 2, q1)

        # Rotate the X axis by the calculated amount
        w = einops.rearrange(
            jnp.array(
                [
                    [jnp.zeros(batch_size), -v_y[:, 2], v_y[:, 1]],
                    [v_y[:, 2], jnp.zeros(batch_size), -v_y[:, 0]],
                    [-v_y[:, 1], v_y[:, 0], jnp.zeros(batch_size)],
                ]
            ),
            "i j b -> b i j",
        )

        r = (
            jnp.eye(3)
            + jnp.einsum("bij, b -> bij", w, jnp.sin(q1))
            + jnp.einsum("bij, b -> bij", w**2, (1 - jnp.cos(q1)))
        )
        v_x_rotated = jnp.einsum("bij, bj -> bi", r, v_x)

        # The desired position of the y-axis is the negative cross product of the desired z (0, 0, 1).T and the current
        # x-axis, which is already rotated around y. The negative is there because the x-axis is flipped.
        # -((0, 0, 1).T x v_x))
        y_desired = jnp.array(
            [v_x_rotated[:, 1], -v_x_rotated[:, 0], jnp.zeros(batch_size)]
        ).T

        # Find the signed angle from the current to the desired y-axis around the new rotated x-axis
        q2 = jnp.arctan2(
            jnp.einsum("bi, bi -> b", jax.vmap(_cross_3d)(v_y, y_desired), v_x_rotated),
            jnp.einsum("bi, bi -> b", v_y, y_desired),
        )

        if u_joint_pos_prev is not None:
            q2 = jnp.where(q2 - u_joint_pos_prev[:, 1] > jnp.pi, q2 - jnp.pi * 2, q2)
            q2 = jnp.where(q2 - u_joint_pos_prev[:, 1] < -jnp.pi, q2 + jnp.pi * 2, q2)

        # if u_joint_pos_prev is not None:
        #     if q2 - u_joint_pos_prev[1] > jnp.pi:
        #         q2 -= jnp.pi * 2
        #     elif q2 - u_joint_pos_prev[1] < -jnp.pi:
        #         q2 += jnp.pi * 2

        alpha_y = jnp.minimum(jnp.maximum(q1, -jnp.pi / 2 * 0.95), jnp.pi / 2 * 0.95)
        alpha_x = jnp.minimum(jnp.maximum(q2, -jnp.pi / 2 * 0.95), jnp.pi / 2 * 0.95)

        u_joint_pos_des = u_joint_pos_des.at[:, i * 2 : i * 2 + 2].set(
            jnp.array([alpha_y, alpha_x]).T
        )

    return u_joint_pos_des


def _cross_3d(a: jnp.ndarray, b: jnp.ndarray) -> jnp.ndarray:
    return jnp.array(
        [
            a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0],
        ]
    )
