.. _evaluation:

Evaluation
==========

The evaluation is conducted on the Huawei Cloud. The cloud server will regularly check the docker server
and look for update. If an image update is detected, the evaluation will automatically start.
**You can only evaluate your agent once per day**.

The cloud computer currently has 12 vCPUs and 24 GiB RAM. Each agent will be assigned with 4 cores.
Unfortunately, we don't support GPU acceleration this year.


What is evaluated
-----------------
For the warmup phase we will evaluate the agent by running

.. code-block:: python

    python run.py --n_episodes 1 --steps_per_game 45000 -e tournament -g phase-3

That means that we load the default config file ``air_hockey_agent/agent_config.yml`` and overwrite the parameters
related for evaluation. You can append your agent related parameters in the config file.

.. note::

    Please keep in mind that we will restore all all the ``air_hockey_challenge/*`` files as well as ``run.py`` to their
    original state for evaluation on the server.


Get the Dataset
---------------

We provide a script to download the dataset once the evaluation is done. You can download them by running

.. code-block:: python

    python scripts/download_dataset.py


Dataset
-------

The dataset is organized as follows:

::

    dataset_path
    ├── qualifying
    │   ├── eval-{datetime}
    │   │   ├── out.log
    │   │   ├── result_{team}.json
    │   │   ├── Game_0
    │   │   │   ├── computation_time.npy
    │   │   │   ├── ee_constr.npy
    │   │   │   ├── joint_pos_constr.npy
    │   │   │   ├── joint_vel_constr.npy
    │   │   │   ├── dataset.pkl
    │   │   │   └── violations.json
    └── tournament


You can load the files as

.. code-block:: python

    dataset = np.load(path_to_the_file, allow_pickle=True)

The dataset in ``dataset.pkl`` stores a list of tuples at each step. The length of the list is the total number of
steps of the evaluation. Each item in the list contains a tuple:

.. code-block:: python

    (state[array], action[array], reward[float], next_state[array], absorbing[bool], last[bool])

The other ``.npy`` files contain the specific info of the evaluation at each step.

To replay the dataset, you can call the function in the ``air_hockey_challenge/utils/replay_dataset.py``

Metric
------

The agent is evaluated by playing against the baseline agent for 45000 steps. 
We will compute two metrics in the evaluation:

**Success Rate**: A success criterion is defined for each task. We will check if the task
succeed when the episode terminates. Each episode can terminate because of two reasons:

#. Maximum number of steps reached
#. No further interaction can be in the episodes.

**Deployability**: The deployability score will assess your agent in multiple
aspects. Each metric will be assigned one or more penalty points depending on the level
of risk. When a constraint is violated, the deployability penalty points are added up
by the associated penalty points. Same violations will count once per episode.
The following are the constraints considered in the challenge:

#. Violations of the End-Effector's Position Constraints (3):
    The x-y-position of the end-effector should remain within the boundaries of the table.
    The z-position of the end-effector should remain within a range.
    The end-effector's position can be computed :math:`p = \mathrm{ForwardKinematics} (q)_{x}`.
    The constraints can be represented as:

    :math:`l_x < p_x,`

    :math:`l_y < p_y < u_y,`

    :math:`\mathrm{table\,height - tolerance} < p_z < \mathrm{table\, height + tolerance}`

    This constraint is very strict. As we use a high-gain tracking controller to improve the
    tracking performance. An infeasible command could potentially damage the table, end-effector
    , or robot actuator.

#. Violations of the Elbow and Wrist Link Height Constraints (3):
    The elbow and wrist link should remain high enough to avoid collision with the table.
    The height of the elbow and wrist link can be computed as:

    :math:`z_\mathrm{elbow} > 0.25, \quad z_\mathrm{wrist} > 0.25`

#. Violations of the Joint Position Limit Constraints (2):
    The joint position should not exceed the position limits. In the real-robot, violations
    of the position limit will trigger the emergency break.

    :math:`q_l < q_{cmd} < q_u`

#. Violations of the Joint Velocity Limit Constraints (1):
    The velocity should not exceed the velocity limits. The controller in
    the real-robot are set not to exceed the velocity limits.

    :math:`\dot{q}_l < \dot{q}_{cmd} < \dot{q}_u`

#. Computation Time (0.5 - 2):
    The computation time at each step should be smaller than 0.02s.

    * Penalty Points 2:
        maximum computation time > 0.2s or average computation time > 0.02s

    * Penalty Points 1:
        0.2s >= maximum computation time > 0.1s

    * Penalty Points 0.5:
        0.1s >= maximum computation time > 0.02s

Leaderboard
-----------

We will categorize your agent into three categories based on the deployability penalty:

* Deployable
* Improvable
* Non-deployable

The leaderboard is divided into three categories by ``Deployability``. Each
category will be ranked separately according to ``Success Rate``. At each stage, we provide
an overall leaderboard and a task-specific leaderboard. In the overall leaderboard,
deployability is categorized by the maximum penalty score for all tasks; the score for
the ranking is a weighted average of the success rates of all tasks.