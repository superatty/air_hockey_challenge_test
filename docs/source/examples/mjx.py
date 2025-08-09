from air_hockey_challenge.environments.brax_envs import AirHockeyPrepare
import os
import jax
from jax import numpy as jnp
import cv2

os.environ["XLA_FLAGS"] = (
    "--xla_gpu_triton_gemm_any=True "
    "--xla_gpu_enable_latency_hiding_scheduler=true "
)

jax.config.update('jax_default_matmul_precision', 'highest')

env = AirHockeyPrepare()
rng = jax.random.PRNGKey(0)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)

while True:
    # Reset the environment
    rng, key = jax.random.split(rng)
    state = jit_reset(key)

    action = jnp.zeros(7)
    states = [state.pipeline_state]

    for i in range(500):
        state = jit_step(state, action)
        states.append(state.pipeline_state)

    images = env.render(states)
    for img in images:
        cv2.imshow("Air Hockey", img[:, :, ::-1])
        cv2.waitKey(round(env.dt * 1000))