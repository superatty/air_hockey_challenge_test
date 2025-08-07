import jax.numpy as jnp

def robot_to_world(base_frame, translation, rotation=None):
    """
    Transform position or rotation optional from the robot base frame to the world frame

    Args
    ----
    base_frame: jnp.ndarray, (4,4)
        The transformation matrix from the world to robot base frame
    translation: jnp.ndarray, (3,)
        The 3D position to be transformed
    rotation: optional, jnp.ndarray, (3, 3)
        The rotation in the matrix form to be transformed

    Returns
    -------
    position: jnp.ndarray (3,)
        The transformed 3D position
    rotation: jnp.ndarray (3, 3)
        The transformed rotation in the matrix form

    """

    target = jnp.eye(4)
    target = target.at[: len(translation), 3].set(translation)
    if rotation is not None:
        target = target.at[:3, :3].set(rotation)

    target_frame = base_frame @ target

    return target_frame[: len(translation), 3], target_frame[:3, :3]


def world_to_robot(base_frame, translation, rotation=None):
    """
    Transfrom position and rotation (optional) from the world frame to the robot's base frame

    Args
    ----
    base_frame: jnp.ndarray, (4,4)
        The transformation matrix from the world to robot base frame
    translation: jnp.ndarray, (3,)
        The 3D position to be transformed
    rotation: optional, jnp.ndarray, (3, 3)
        The rotation in the matrix form to be tranformed

    Returns
    -------
    position: jnp.ndarray, (3,)
        The transformed 3D position
    rotation: jnp.ndarray, (3, 3)
        The transformed rotation in the matrix form
    """

    target = jnp.eye(4)
    target = target.at[: len(translation), 3].set(translation)
    if rotation is not None:
        target = target.at[:3, :3].set(rotation)

    target_frame = jnp.linalg.inv(base_frame) @ target

    return target_frame[: len(translation), 3], target_frame[:3, :3]
