from air_hockey_challenge.environments.brax_envs.env_base import AirHockeyBase
from mujoco import mjx
from jax import numpy as jnp

class AirHockeyDouble(AirHockeyBase):
    def __init__(self, timestep=1/1000, n_intermediate_steps=20):
        super().__init__(n_agents=2, timestep=timestep, n_intermediate_steps=n_intermediate_steps)

    def get_ee(self, data: mjx.Data, robot=1):
        ee_body_id = self.sys.mj_model.body(f"iiwa_{robot}/striker_mallet").id

        ee_pos = data.xpos[ee_body_id]
        ee_vel = data.cvel[ee_body_id]

        return ee_pos, ee_vel
    
    def get_joints(self, obs, robot=None):
        if robot == 1:
            q_pos = obs[jnp.asarray(self.env_info["joint_pos_ids"])]
            q_vel = obs[jnp.asarray(self.env_info["joint_vel_ids"])]
        elif robot == 2:
            q_pos = obs[23 + jnp.asarray(self.env_info["joint_pos_ids"])]
            q_vel = obs[23 + jnp.asarray(self.env_info["joint_vel_ids"])]
        else:
            assert robot is None

            q_pos = jnp.concatenate(
                [
                    obs[jnp.asarray(self.env_info["joint_pos_ids"])],
                    obs[23 + jnp.asarray(self.env_info["joint_pos_ids"])],
                ],
            )
            q_vel = jnp.concatenate(
                [
                    obs[jnp.asarray(self.env_info["joint_vel_ids"])],
                    obs[23 + jnp.asarray(self.env_info["joint_vel_ids"])],
                ],
            )

        return q_pos, q_vel
    
    def _create_observation(self, state, obs):
        obs = super()._create_observation(state, obs)
        
        # Filter the joint velocity
        q_pos, q_vel = self.get_joints(obs)
        q_vel_filter = self.filter_ratio * q_vel + (1 - self.filter_ratio) * state.info["q_vel_prev"]

        state.info.update(q_pos_prev=q_pos, q_vel_prev=q_vel_filter)
        obs = obs.at[jnp.asarray(self.env_info["joint_vel_ids"])].set(q_vel_filter[:7])
        obs = obs.at[23 + jnp.asarray(self.env_info["joint_vel_ids"])].set(q_vel_filter[7:])
        
        yaw_angle = obs[self.env_info["puck_pos_ids"][2]]
        obs = obs.at[self.env_info["puck_pos_ids"][2]].set((yaw_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi)
        
        return obs
    
    def _modify_observation(self, obs):
        obs = super()._modify_observation(obs)
        
        # Convert observation from world frame to robot frame for both robots
        
        puck_pos, puck_vel = self.get_puck(obs)
        
        # Modify observation for first robot 
        robot_frame = self.env_info["robot"]["base_frame"][0]
        
        puck_pos = self._puck_pose_2d_in_robot_frame(puck_pos, robot_frame)
        puck_vel = self._puck_vel_2d_in_robot_frame(puck_vel, robot_frame)

        obs = obs.at[jnp.asarray(self.env_info["puck_pos_ids"])].set(puck_pos)
        obs = obs.at[jnp.asarray(self.env_info["puck_vel_ids"])].set(puck_vel)

        opponent_ee_pos = obs[jnp.asarray(self.env_info["opponent_ee_ids"])]
        opponent_ee_pos = (jnp.linalg.inv(robot_frame) @ jnp.concatenate([opponent_ee_pos, jnp.ones(1)]))[:3]
        obs = obs.at[jnp.asarray(self.env_info["opponent_ee_ids"])].set(opponent_ee_pos)

        # Modify observation for second robot
        robot_frame = self.env_info["robot"]["base_frame"][1]
        
        puck_pos = self._puck_pose_2d_in_robot_frame(puck_pos, robot_frame)
        puck_vel = self._puck_vel_2d_in_robot_frame(puck_vel, robot_frame)
        
        obs = obs.at[23 + jnp.asarray(self.env_info["puck_pos_ids"])].set(puck_pos)
        obs = obs.at[23 + jnp.asarray(self.env_info["puck_vel_ids"])].set(puck_vel)
        
        opponent_ee_pos = obs[23 + jnp.asarray(self.env_info["opponent_ee_ids"])]
        opponent_ee_pos = (jnp.linalg.inv(robot_frame) @ jnp.concatenate([opponent_ee_pos, jnp.ones(1)]))[:3]
        obs = obs.at[23 + jnp.asarray(self.env_info["opponent_ee_ids"])].set(opponent_ee_pos)

        return obs
