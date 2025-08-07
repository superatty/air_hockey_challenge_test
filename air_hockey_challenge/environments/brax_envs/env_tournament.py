import jax
import jax.numpy as jnp
from brax.envs.base import State

from air_hockey_challenge.environments.brax_envs.env_double import AirHockeyDouble


class AirHockeyTournament(AirHockeyDouble):
    """
    Class for the air hockey tournament. Consists of 2 robots which should play against each other.
    When the puck is on one side for more than 15 seconds the puck is reset and the player gets a penalty.
    If a player accumulates 3 penalties his score is reduced by 1.
    """

    def __init__(
        self,
        timestep=1 / 1000,
        n_intermediate_steps=20,
        agent_name="Agent",
        opponent_name="Opponent",
    ):
        super().__init__(
            timestep=timestep,
            n_intermediate_steps=n_intermediate_steps,
        )

        hit_width = (
            self.env_info["table"]["width"] / 2
            - self.env_info["puck"]["radius"]
            - self.env_info["mallet"]["radius"] * 2
        )

        self.hit_range = jnp.array(
            [[-0.7, -0.2], [-hit_width, hit_width]]
        )  # Table Frame

    def _setup(
        self,
        rng_key: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        """
        Initialize the environment state, including puck position and tournament state.
        """
        qpos, qvel = super()._setup(rng_key)

        rng, rng2 = jax.random.split(rng_key, 2)

        # Initialize puck position
        puck_pos = (
            jax.random.uniform(rng, shape=(2))
            * (self.hit_range[:, 1] - self.hit_range[:, 0])
            + self.hit_range[:, 0]
        )

        start_side = jax.random.choice(rng2, jnp.array([1, -1]))
        
        qpos = qpos.at[self.puck_ids[0]].set(puck_pos[0] * start_side)
        qpos = qpos.at[self.puck_ids[1]].set(puck_pos[1])

        return qpos, qvel

    def _reward(self, state, action, next_state, absorbing) -> jax.Array:
        """
        Compute the reward for the tournament task. Currently a placeholder.
        """
        return 0

    def update_timer(self, prev_side, timer, puck_x, dt):
        new_side = jnp.sign(puck_x)
        timer = jnp.where(new_side == prev_side, timer + dt, 0.0)
        prev_side = jnp.where(new_side != prev_side, new_side, prev_side)
        return prev_side, timer

    def handle_timeout(self, prev_side, faults, score, start_side, timer, puck_pos):
        # Agent fault (puck on agent's side, x < -0.15)
        faults_agent = faults.at[0].add(1)
        score_opponent = score.at[1].add(jnp.where(faults_agent[0] % 3 == 0, 1, 0))
        start_side_agent = jnp.array(-1)

        # Opponent fault (puck on opponent's side, x > 0.15)
        faults_opponent = faults.at[1].add(1)
        score_agent = score.at[0].add(jnp.where(faults_opponent[1] % 3 == 0, 1, 0))
        start_side_opponent = jnp.array(1)

        # Apply fault logic based on prev_side
        faults = jnp.where(
            (prev_side == -1) & (jnp.abs(puck_pos[0]) >= 0.15), faults_agent, faults
        )
        faults = jnp.where(
            (prev_side == 1) & (jnp.abs(puck_pos[0]) >= 0.15), faults_opponent, faults
        )
        score = jnp.where(
            (prev_side == -1) & (jnp.abs(puck_pos[0]) >= 0.15), score_opponent, score
        )
        score = jnp.where(
            (prev_side == 1) & (jnp.abs(puck_pos[0]) >= 0.15), score_agent, score
        )
        start_side = jnp.where(
            (prev_side == -1) & (jnp.abs(puck_pos[0]) >= 0.15),
            start_side_agent,
            start_side,
        )
        start_side = jnp.where(
            (prev_side == 1) & (jnp.abs(puck_pos[0]) >= 0.15),
            start_side_opponent,
            start_side,
        )

        return (
            faults,
            score,
            start_side,
            (timer > 15.0) & (jnp.abs(puck_pos[0]) >= 0.15),
        )

    def _is_absorbing(self, state: State) -> tuple[State, jax.Array]:
        """
        Determine if the episode should terminate based on tournament rules.
        Updates scores, faults, and start_side as needed.
        """
        puck_pos, puck_vel = self.get_puck(state.obs)

        prev_side, timer = self.update_timer(
            state.metrics["tournament_prev_side"],
            state.metrics["tournament_timer"],
            puck_pos[0],
            self.dt,
        )

        # Check for puck stuck on one side for more than 15 seconds
        faults, score, start_side, timeout_absorbing = self.handle_timeout(
            prev_side,
            state.metrics["tournament_faults"],
            state.metrics["tournament_score"],
            state.metrics["tournament_start_side"],
            timer,
            puck_pos,
        )

        # Check for puck in goal
        table_length = self.env_info["table"]["length"]
        goal_width = self.env_info["table"]["goal_width"]
        goal_absorbing = (jnp.abs(puck_pos[1]) - goal_width / 2 <= 0) & (
            jnp.abs(puck_pos[0]) > table_length / 2
        )

        # Update score and start_side for goals
        score = score.at[0].add(
            jnp.where(
                (puck_pos[0] > table_length / 2)
                & (jnp.abs(puck_pos[1]) - goal_width / 2 <= 0),
                1,
                0,
            )
        )
        score = score.at[1].add(
            jnp.where(
                (puck_pos[0] < -table_length / 2)
                & (jnp.abs(puck_pos[1]) - goal_width / 2 <= 0),
                1,
                0,
            )
        )
        start_side = jnp.where(
            (puck_pos[0] > table_length / 2)
            & (jnp.abs(puck_pos[1]) - goal_width / 2 <= 0),
            -1,
            start_side,
        )
        start_side = jnp.where(
            (puck_pos[0] < -table_length / 2)
            & (jnp.abs(puck_pos[1]) - goal_width / 2 <= 0),
            1,
            start_side,
        )

        # Check for puck stuck in the middle
        stuck_absorbing = (jnp.abs(puck_pos[0]) < 0.15) & (
            jnp.linalg.norm(puck_vel[0]) < 0.025
        )

        # Combine with base class absorbing condition
        base_absorbing = super()._is_absorbing(state)

        absorbing = (
            timeout_absorbing | goal_absorbing | stuck_absorbing | base_absorbing[1]
        )

        new_metrics = dict(state.metrics)
        new_metrics.update(
            tournament_prev_side=prev_side,
            tournament_timer=timer,
            tournament_faults=faults,
            tournament_score=score,
            tournament_start_side=start_side,
        )

        state = state.replace(metrics=new_metrics)

        return state, absorbing * 1.0


if __name__ == "__main__":
    # from mujoco.viewer import launch_passive
    import tqdm

    env = AirHockeyTournament()

    action = jax.random.uniform(jax.random.key(0), shape=(14), minval=-1, maxval=1) * 8

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    # data = mjx.get_data(env.model, env.mjx_data)
    # viewer = launch_passive(env.model, data[0])
    rng = jax.random.PRNGKey(0)
    for _ in range(10):
        obs = jit_reset(rng)

        # mjx.get_data_into(data, env.model, env.mjx_data)
        # viewer.sync()

        for _ in tqdm.tqdm(range(50)):
            obs = jit_step(obs, action)
            # mjx.get_data_into(data, env.model, env.mjx_data)
            # viewer.sync()
