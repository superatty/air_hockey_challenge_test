import jax
from mujoco import mjx
from jax import numpy as jnp

import mujoco
from mujoco.mjx._src import math as mjx_math
from mujoco.mjx._src.forward import _integrate_pos

from jax.scipy.spatial.transform import Rotation as R

from brax.math import safe_norm
from collections import namedtuple

LINK_TO_XML_NAME = {
    "1": "iiwa_1/link_1",
    "2": "iiwa_1/link_2",
    "3": "iiwa_1/link_3",
    "4": "iiwa_1/link_4",
    "5": "iiwa_1/link_5",
    "6": "iiwa_1/link_6",
    "7": "iiwa_1/link_7",
    "ee": "iiwa_1/striker_joint_link",
}


def forward_kinematics(mjx_model: mjx.Model, mjx_data: mjx.Data, q, link="ee"):
    return _mujoco_fk(q, LINK_TO_XML_NAME[link], mjx_model, mjx_data)


def inverse_kinematics(
    mjx_model: mjx.Model,
    mjx_data: mjx.Data,
    desired_position,
    desired_rotation=None,
    initial_q=None,
    link="ee",
):
    q_init = jnp.zeros(mjx_model.nq)
    if initial_q is None:
        q_init = mjx_data.qpos
    else:
        q_init = q_init.at[: initial_q.shape[0]].set(initial_q)

    q_l = mjx_model.jnt_range[:, 0]
    q_h = mjx_model.jnt_range[:, 1]
    lower_limit = (q_l + q_h) / 2 - 0.95 * (q_h - q_l) / 2
    upper_limit = (q_l + q_h) / 2 + 0.95 * (q_h - q_l) / 2

    desired_quat = None
    if desired_rotation is not None:
        desired_quat = R.from_matrix(desired_rotation).as_quat(scalar_first=True)

    return _mujoco_clik(
        desired_position,
        desired_quat,
        q_init,
        LINK_TO_XML_NAME[link],
        mjx_model,
        mjx_data,
        lower_limit,
        upper_limit,
    )


def jacobian(
    mjx_model: mjx.Model, mjx_data: mjx.Data, q: jax.Array, link: str = "ee"
) -> jax.Array:
    jac = _mujoco_jac(q, LINK_TO_XML_NAME[link], mjx_model, mjx_data)
    return jac


def _mujoco_jac(q: jax.Array, name: str, model: mjx.Model, data: mjx.Data) -> jax.Array:
    def f(q):
        new_data = data.replace(qpos=q)
        new_data = mjx.forward(model, new_data)

        body_id = mjx.name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

        # Get body position (world frame)
        pos = new_data.xpos[body_id]  # shape (3,)
        rot = new_data.xmat[body_id]  # shape (3, 3)
        rot = rot.reshape(-1)  # Flatten to (9,)
        return jnp.concatenate([pos, rot])  # shape (12,)

    jac_fn = jax.jacobian(f)
    jac_pos = jac_fn(q)

    return jac_pos


def _mujoco_fk(q: jax.Array, name: str, model: mjx.Model, data: mjx.Data):
    qpos = data.qpos
    qpos = qpos.at[: q.shape[-1]].set(q)
    data = data.replace(qpos=qpos)

    data = mjx.fwd_position(model, data)

    body_id = mjx.name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return data.xpos[body_id].copy(), data.xmat[body_id].copy()

def _mujoco_clik(
    desired_pos, desired_quat, initial_q, name, model, data, lower_limit, upper_limit
):
    body_id = mjx.name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)

    IT_MAX = 1000
    eps = 1e-4
    damp = 1e-3
    progress_thresh = 20.0
    max_update_norm = 0.1
    rot_weight = 1

    # initialize data.qpos
    data0 = data.replace(qpos=initial_q)

    # a little struct to hold loop‐carried values
    State = namedtuple("State", ["data", "i", "error_norm", "progress_criterion"])

    # start with infinite error_norm so we definitely enter once
    init_state = State(data=data0, i=0, error_norm=jnp.inf, progress_criterion=0.0)

    def cond_fn(state):
        # keep looping while all three “continue” conditions hold:
        c1 = state.error_norm > eps
        c2 = state.i < IT_MAX
        c3 = state.progress_criterion <= progress_thresh
        return c1 & c2 & c3

    def body_fn(state):
        data = mjx.fwd_position(model, state.data)
        x_pos = data.xpos[body_id]
        x_quat = data.xquat[body_id]

        # -- compute  err  and  error_norm  --
        error_norm = 0.0
        # reuse a 6-vector for convenience
        err = jnp.zeros(6 if (desired_pos is not None and desired_quat is not None) else 3,
                        dtype=data.qpos.dtype)

        if desired_pos is not None:
            err_pos = desired_pos - x_pos
            error_norm = error_norm + safe_norm(err_pos)
            err = err.at[:3].set(err_pos)

        if desired_quat is not None:
            neg_x_quat = mjx_math.quat_inv(x_quat)
            err_quat = mjx_math.quat_mul(desired_quat, neg_x_quat)
            err_rot = _quat2vel(err_quat)
            error_norm = error_norm + rot_weight * safe_norm(err_rot)
            err = err.at[-3:].set(err_rot)

        # -- compute jacobian and update step  --
        jac_pos, jac_rot = mjx.jac(model, data, x_pos, body_id)
        if desired_quat is None:
            jac = jac_pos.T
        elif desired_pos is None:
            jac = jac_rot.T
        else:
            jac = jnp.concatenate((jac_pos.T, jac_rot.T), axis=0)

        hess_approx = jac.T @ jac
        joint_delta = jac.T @ err
        hess_approx = hess_approx + jnp.eye(hess_approx.shape[0]) * damp

        update_joints = jnp.linalg.solve(hess_approx, joint_delta)
        update_norm = safe_norm(update_joints)

        progress_criterion = error_norm / (update_norm + 1e-16)

        # cap step‐size
        update_joints = jnp.where(update_norm > max_update_norm,
                                  update_joints * (max_update_norm / update_norm),
                                  update_joints)

        # integrate and clamp
        new_qpos = _mjx_integrate_pos(model, data.qpos, update_joints, 1)
        new_qpos = jnp.clip(new_qpos, lower_limit, upper_limit)
        new_data = data.replace(qpos=new_qpos)

        return State(data=new_data,
                     i=state.i + 1,
                     error_norm=error_norm,
                     progress_criterion=progress_criterion)

    # run the loop
    final_state = jax.lax.while_loop(cond_fn, body_fn, init_state)

    # decide success just like your original breaks
    success = final_state.error_norm < eps
    return success, final_state.data.qpos.copy()


# def _mujoco_clik(
#     desired_pos, desired_quat, initial_q, name, model, data, lower_limit, upper_limit
# ):
#     body_id = mjx.name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    
#     IT_MAX = 1000
#     eps = 1e-4
#     damp = 1e-3
#     progress_thresh = 20.0
#     max_update_norm = 0.1
#     rot_weight = 1
#     i = 0

#     dtype = data.qpos.dtype

#     data = data.replace(qpos=initial_q)

#     neg_x_quat = jnp.empty(4, dtype=dtype)
#     error_x_quat = jnp.empty(4, dtype=dtype)

#     if desired_pos is not None and desired_quat is not None:
#         err = jnp.empty(6, dtype=dtype)
#     else:
#         err = jnp.empty(3, dtype=dtype)

#     while True:
#         data = mjx.fwd_position(model, data)

#         x_pos = data.xpos[body_id]
#         x_quat = data.xquat[body_id]

#         error_norm = 0
#         if desired_pos is not None:
#             err_pos = desired_pos - x_pos
#             error_norm += safe_norm(err_pos)
#             err = err.at[:3].set(err_pos)

#         if desired_quat is not None:
#             neg_x_quat = mjx_math.quat_inv(x_quat)
#             error_x_quat = mjx_math.quat_mul(desired_quat, neg_x_quat)
#             err_rot = _quat2vel(error_x_quat)
#             error_norm += safe_norm(err_rot) * rot_weight
#             err = err.at[-3:].set(err_rot)

#         if error_norm < eps:
#             success = True
#             break

#         if i >= IT_MAX:
#             success = False
#             break

#         jac_pos, jac_rot = mjx.jac(model, data, data.xpos[body_id], body_id)
#         jac = jnp.concatenate((jac_pos.T, jac_rot.T), axis=0)

#         hess_approx = jac.T.dot(jac)
#         joint_delta = jac.T.dot(err)

#         hess_approx += jnp.eye(hess_approx.shape[0]) * damp
#         update_joints = jnp.linalg.solve(hess_approx, joint_delta)

#         update_norm = safe_norm(update_joints)

#         progress_criterion = error_norm / update_norm
#         if progress_criterion > progress_thresh:
#             success = False
#             break

#         if update_norm > max_update_norm:
#             update_joints *= max_update_norm / update_norm

#         data = data.replace(qpos=_mjx_integrate_pos(model, data.qpos, update_joints, 1))
#         data = data.replace(qpos=jnp.clip(data.qpos, lower_limit, upper_limit))
#         i += 1

#     return success, data.qpos.copy()


def _quat2vel(quat: jax.Array) -> jax.Array:
    axis, angle = mjx_math.quat_to_axis_angle(quat)
    return axis * angle


def _mjx_integrate_pos(model: mjx.Model, qpos, qvel, dt):
    jnt_types = [model.jnt_type[i] for i in range(model.njnt)]

    return _integrate_pos(jnt_types, qpos, qvel, dt)
