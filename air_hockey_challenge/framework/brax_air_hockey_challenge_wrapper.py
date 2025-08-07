from copy import deepcopy
import jax
import jax.numpy as jnp

from air_hockey_challenge.constraints.brax_constraints import *
from air_hockey_challenge.environments import brax_position_control_wrapper as position
from air_hockey_challenge.utils.mjx.transformations import robot_to_world

from brax.envs.base import Wrapper


class AirHockeyChallengeWrapper(Wrapper):
    def __init__(
        self, env, custom_reward_function=None, interpolation_order=3, **kwargs
    ):
        """
        Environment Constructor

        Args:
            env [string]:
                The string to specify the running environments. Available environments: [3dof-hit, 3dof-defend].
                [7dof-hit, 7dof-defend, 7dof-prepare, tournament] will be available once the corresponding stage starts.
            custom_reward_function [callable]:
                You can customize your reward function here.
            interpolation_order (int, 3): Type of interpolation used, has to correspond to action shape. Order 1-5 are
                    polynomial interpolation of the degree. Order -1 is linear interpolation of position and velocity.
                    Set Order to None in order to turn off interpolation. In this case the action has to be a trajectory
                    of position, velocity and acceleration of the shape (20, 3, n_joints)
        """

        env_dict = {
            "tournament": position.IiwaPositionTournament,
            "hit": position.IiwaPositionHit,
            "defend": position.IiwaPositionDefend,
            "prepare": position.IiwaPositionPrepare,
            # "3dof-hit": position.PlanarPositionHit,
            # "3dof-defend": position.PlanarPositionDefend,
        }

        if env == "tournament" and type(interpolation_order) != tuple:
            interpolation_order = (interpolation_order, interpolation_order)

        base_env = env_dict[env](interpolation_order=interpolation_order, **kwargs)
        self.env_name = env
        self.env_info = base_env.env_info

        if custom_reward_function:
            base_env._reward = lambda state: custom_reward_function(state)

        constraint_list = ConstraintList()
        constraint_list.add(JointPositionConstraint(self.env_info))
        constraint_list.add(JointVelocityConstraint(self.env_info))
        constraint_list.add(EndEffectorConstraint(self.env_info))
        constraint_list.add(LinkConstraint(self.env_info))

        self.env_info["constraints"] = constraint_list
        self.env_info["env_name"] = self.env_name

        super().__init__(base_env)  # base_env is now self.env

    def reset(self, state):
        state = super().reset(state)

        if "tournament" in self.env_name:
            state.info.update(
                constraints_value=list(), success=jnp.zeros(())
            )
        else:
            state.info.update(
                constraints_value=dict(
                    joint_vel_constr=jnp.zeros(14),
                    joint_pos_constr=jnp.zeros(14),
                    ee_constr=jnp.zeros(5),
                    link_constr=jnp.zeros(2),
                ),
                success=jnp.zeros((), dtype=jnp.int32),
            )

        state.info.update(
            max_j_pos_violation=0,
            max_j_vel_violation=0,
            max_jerk_violation=0,
            max_ee_x_violation=0,
            max_ee_y_violation=0,
            max_ee_z_violation=0,
            max_link_violation=0,
            num_j_pos_violation=0,
            num_j_vel_violation=0,
            num_ee_x_violation=0,
            num_ee_y_violation=0,
            num_ee_z_violation=0,
            num_link_violation=0
        )

        return state

    def step(self, state, action):
        state = super().step(state, action)

        obs, info = state.obs, state.info

        if "tournament" in self.env_name:
            info["constraints_value"] = list()
            for i in range(2):
                obs_agent = obs[i * int(len(obs) / 2) : (i + 1) * int(len(obs) / 2)]
                info["constraints_value"].append(
                    deepcopy(
                        self.env_info["constraints"].fun(
                            obs_agent[jnp.asarray(self.env_info["joint_pos_ids"])],
                            obs_agent[jnp.asarray(self.env_info["joint_vel_ids"])],
                        )
                    )
                )

            info["score"] = state.metrics["tournament_score"]
            info["faults"] = state.metrics["tournament_faults"]

        else:
            info["constraints_value"] = deepcopy(
                self.env_info["constraints"].fun(
                    obs[jnp.asarray(self.env_info["joint_pos_ids"])],
                    obs[jnp.asarray(self.env_info["joint_vel_ids"])],
                )
            )
            info["success"] = self.check_success(obs)

        self._update_constraints_info(info)

        return state.replace(info=info)
    
    def _update_constraints_info(self, info):
        j_pos_constr = info['constraints_value']['joint_pos_constr']
        j_vel_constr = info['constraints_value']['joint_vel_constr']
        ee_constr = info['constraints_value']['ee_constr']
        link_constr = info['constraints_value']['link_constr']

        info["max_j_pos_violation"] = jnp.maximum(info["max_j_pos_violation"], jnp.max(j_pos_constr))
        info["max_j_vel_violation"] = jnp.maximum(info["max_j_vel_violation"], jnp.max(j_vel_constr))
        info["max_ee_x_violation"] = jnp.maximum(info["max_ee_x_violation"], jnp.max(ee_constr[0]))
        info["max_ee_y_violation"] = jnp.maximum(info["max_ee_y_violation"], jnp.max(ee_constr[1:3]))
        info["max_ee_z_violation"] = jnp.maximum(info["max_ee_z_violation"], jnp.max(ee_constr[3:5]))
        info["max_link_violation"] = jnp.maximum(info["max_link_violation"], jnp.max(link_constr))
        info["num_j_pos_violation"] = info["num_j_pos_violation"] + jnp.where(jnp.any(j_pos_constr > 0), 1, 0)
        info["num_j_vel_violation"] = info["num_j_vel_violation"] + jnp.where(jnp.any(j_vel_constr > 0), 1, 0)
        info["num_ee_x_violation"] = info["num_ee_x_violation"] + jnp.where(jnp.any(ee_constr[0] > 0), 1, 0)
        info["num_ee_y_violation"] = info["num_ee_y_violation"] + jnp.where(jnp.any(ee_constr[1:3] > 0), 1, 0)
        info["num_ee_z_violation"] = info["num_ee_z_violation"] + jnp.where(jnp.any(ee_constr[3:5] > 0), 1, 0)
        info["num_link_violation"] = info["num_link_violation"] + jnp.where(jnp.any(link_constr > 0), 1, 0)

    def check_success(self, obs):
        puck_pos, puck_vel = self.env.get_puck(obs)

        puck_pos, _ = robot_to_world(
            self.env_info["robot"]["base_frame"][0], translation=puck_pos
        )
        success = 0

        def hit_case():
            cond_x = puck_pos[0] - self.env_info["table"]["length"] / 2 > 0
            cond_y = jnp.abs(puck_pos[1]) - self.env_info["table"]["goal_width"] / 2 < 0
            return jnp.logical_and(cond_x, cond_y)

        def defend_case():
            cond_x = jnp.logical_and(puck_pos[0] > -0.8, puck_pos[0] <= -0.2)
            cond_vel = puck_vel[0] < 0.1
            return jnp.logical_and(cond_x, cond_vel)

        def prepare_case():
            cond_x = jnp.logical_and(puck_pos[0] > -0.8, puck_pos[0] <= -0.2)
            cond_y = jnp.abs(puck_pos[1]) < 0.39105
            cond_vel = puck_vel[0] < 0.1
            return jnp.logical_and(jnp.logical_and(cond_x, cond_y), cond_vel)

        success = 0

        if "hit" in self.env_name:
            success = jnp.where(hit_case(), 1, 0)
        elif "defend" in self.env_name:
            success = jnp.where(defend_case(), 1, 0)
        elif "prepare" in self.env_name:
            success = jnp.where(prepare_case(), 1, 0)

        return success


if __name__ == "__main__":
    import tqdm

    import os

    os.environ["XLA_FLAGS"] = (
        "--xla_gpu_triton_gemm_any=True "
        "--xla_gpu_enable_latency_hiding_scheduler=true "
    )

    # jax.config.update("jax_enable_x64", True)

    import time

    t = time.time()

    env = AirHockeyChallengeWrapper(
        env="defend",
        interpolation_order=3,
    )
    rng = jax.random.PRNGKey(0)

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)

    state = jit_reset(rng)
    action = jnp.zeros((2, env.env_info["robot"]["n_joints"]))

    # from etils import epy

    for i in range(100):
        state = jit_step(state, action)
        # print("Step:", i)
        # epy.pprint(state.info['constraints_value'])

    jax.block_until_ready(state)
    print("Time taken for 100 steps:", time.time() - t)
