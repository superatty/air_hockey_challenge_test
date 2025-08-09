.. _mjx:

============
MJX
============

`MJX <https://mujoco.readthedocs.io/en/stable/mjx.html>`_ is a high-performance physics simulation backend developed 
to leverage GPU acceleration for faster and more efficient training.
It is a reimplementation of the MuJoCo physics engine in `JAX <https://jax.dev>`, a powerful numerical computing library
designed for high-performance machine learning. Please refer to the
`quickstart guide, <https://docs.jax.dev/en/latest/notebooks/thinking_in_jax.html>` for more information on JAX.

We provide the environments additionally in MJX, which allows for faster training of your agents.
They are implemented with the `Brax <https://github.com/google/brax>`_ library, which includes a set of tools for
implementing MJX environments and training agents in JAX. 

.. note::
    Note that the evaluation is done purely on CPU using the MuJoCo physics engine and not on the MJX environments.
    Therefore, you can use the MJX environments for training your agents, but you need to ensure that your agent
    is compatible with the MuJoCo physics engine for evaluation.

Here is a simple example of how to use the MJX environments:

.. literalinclude:: examples/mjx.py

The key things to note are the following:

* In the MuJoCo environments, the puck and the mallets are cylinder objects, while in the MJX environments, they are capsule objects.
  This is due to the limitations of MJX: `Collisions between cylinder and mesh objects are not supported. <https://mujoco.readthedocs.io/en/stable/mjx.html#feature-parity>`_

* To maximize performance with JAX, you should use ``jax.jit`` to compile both ``env.reset`` and ``env.step`` functions. 
    This ensures that environment interactions are efficiently executed on supported hardware.

* Generating pseudorandom numbers in JAX is different from NumPy. Every JAX function that uses random numbers
    takes a random key as an argument. Providing the same key will yield the same output of the function.
    For reproducibility, you create an explicit random state using ``jax.random.PRNGKey`` and use
    ``jax.random.split`` to create a new key for each call and update the random state.
    More information can be found in the `JAX documentation <https://docs.jax.dev/en/latest/random-numbers.html>`_.

* Adding the followng flags tells XLA to use Triton GEMM, which can improve steps per second by ~30% on some GPUs:

    .. code-block:: python
        os.environ["XLA_FLAGS"] = (
            "--xla_gpu_triton_gemm_any=True "
            "--xla_gpu_enable_latency_hiding_scheduler=true "
        )

* If you encounter NaN values during training or evaluation, you can increase the matrix multiplication precision 
    (with the trade-off of performance) by setting the following configuration:

    .. code-block:: python
        jax.config.update('jax_default_matmul_precision', 'highest')

    The possible values are 'default', 'high', and 'highest'. 

* Rendering the MJX environments is not supported, so to render the environments, the MJX state is converted to a 
    MuJoCo state, which takes some time. If you still want to render the MJX environments, it is suggested to save all
    the MJX states and then convert them to MuJoCo states after the simulation is done just like in the example above.