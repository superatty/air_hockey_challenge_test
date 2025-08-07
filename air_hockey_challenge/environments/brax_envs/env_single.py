from air_hockey_challenge.environments.brax_envs.env_base import AirHockeyBase
from mujoco import mjx
from jax import numpy as jnp


class AirHockeySingle(AirHockeyBase):
    def __init__(self, timestep=1/1000, n_intermediate_steps=20):
        super().__init__(n_agents=1, timestep=timestep, n_intermediate_steps=n_intermediate_steps)
        
    def get_ee(self, data: mjx.Data):
        ee_pos = data.xpos[self.ee_pos_id]
        ee_vel = data.cvel[self.ee_pos_id]
        
        return ee_pos, ee_vel
    
    def get_joints(self, obs):
        qpos = obs[jnp.asarray(self.env_info["joint_pos_ids"])]
        qvel = obs[jnp.asarray(self.env_info["joint_vel_ids"])]
        
        return qpos, qvel
    
    def _create_observation(self, state, obs):
        obs = super()._create_observation(state, obs)
        
        # Filter the joint velocity
        q_pos, q_vel = self.get_joints(obs)
        q_vel_filter = self.filter_ratio * q_vel + (1 - self.filter_ratio) * state.info["q_vel_prev"]
        
        state.info.update(q_pos_prev=q_pos, q_vel_prev=q_vel_filter)
        obs = obs.at[jnp.asarray(self.env_info["joint_vel_ids"])].set(q_vel_filter)
        
        yaw_angle = obs[self.env_info["puck_pos_ids"][2]]
        obs = obs.at[self.env_info["puck_pos_ids"][2]].set((yaw_angle + jnp.pi) % (2 * jnp.pi) - jnp.pi)
        
        return obs
    
    def _modify_observation(self, obs):
        obs = super()._modify_observation(obs)
        
        # Convert observation from world frame to robot frame
        
        puck_pos, puck_vel = self.get_puck(obs)
        robot_frame = self.env_info["robot"]["base_frame"][0]
        
        puck_pos = self._puck_pose_2d_in_robot_frame(puck_pos, robot_frame)
        puck_vel = self._puck_vel_2d_in_robot_frame(puck_vel, robot_frame)

        obs = obs.at[jnp.asarray(self.env_info["puck_pos_ids"])].set(puck_pos)
        obs = obs.at[jnp.asarray(self.env_info["puck_vel_ids"])].set(puck_vel)

        return obs