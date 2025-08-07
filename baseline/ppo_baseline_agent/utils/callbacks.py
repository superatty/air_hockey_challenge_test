import json

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, EventCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import sync_envs_normalization

best_model_file_name = "best_model.zip"
vecnormalize_file_name = "vecnormalize.pkl"
custom_log_file_name = "custom_log.json"
checkpoint_dir = "checkpoint"


class SaveBestModel(BaseCallback):
    def __init__(self, save_dir, custom_log):
        super().__init__(verbose=1)
        self.save_dir = save_dir
        self.custom_log = custom_log

    def _on_step(self) -> bool:
        self.model.save(self.save_dir / best_model_file_name)
        self.training_env.save(self.save_dir / vecnormalize_file_name)
        self.custom_log["best_mean_reward"] = float(self.parent.best_mean_reward)
        return True

class CheckpointLog(BaseCallback):
    def __init__(self, save_dir, custom_log, save_freq):
        super().__init__(verbose=1)
        self.save_dir = save_dir
        self.custom_log = custom_log
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls == 0 or self.n_calls % self.save_freq:
            return True
        self.custom_log["num_timesteps"] = self.model.num_timesteps
        with open(self.save_dir / custom_log_file_name, "w") as f:
            json.dump(self.custom_log, f)
        
        self.model.save(self.save_dir / checkpoint_dir / (str(self.model.num_timesteps) + ".zip"))
        self.training_env.save(self.save_dir / checkpoint_dir / (str(self.model.num_timesteps) + ".pkl"))
        with open(self.save_dir / checkpoint_dir / (str(self.model.num_timesteps) + ".json"), "w") as f:
            json.dump(self.custom_log, f)

        return True

class CustomEvalCallback(EventCallback):
    def __init__(self, eval_env, callback_on_new_best, n_eval_episodes, eval_freq, best_mean_reward = -np.inf):
        super().__init__(callback_on_new_best, verbose=1)
        self.eval_env = eval_env
        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = best_mean_reward
    
    def _log_info_callback(self, locals, globals):
        self.all_infos.append(locals["info"])
        if locals["done"]:
            self.done_infos.append(locals["info"])

    def _on_step(self):
        if self.n_calls == 0 or self.n_calls % self.eval_freq:
            return True
        
        self.all_infos = []
        self.done_infos = []
        sync_envs_normalization(self.training_env, self.eval_env)
        episode_rewards, episode_lengths = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            return_episode_rewards=True,
            warn=True,
            callback=self._log_info_callback,
        )
        mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
        mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
        print(f"Eval num_timesteps={self.num_timesteps}, " f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}")
        print(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")

        if mean_reward > self.best_mean_reward:
            print("New best mean reward!")
            self.best_mean_reward = mean_reward
            self.callback.on_step()

        self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        self.logger.dump(self.num_timesteps)

        # if self.done_infos[0]["task"] == "hit":
        #     # Remap
        #     for i in self.done_infos:
        #         i["reward_sparse"] = mean_reward
        #         i["episode_length"] = mean_ep_length
        #         i["max_puck_velocity_after_hit"] = i["max_puck_vel_after_hit"]
        #         i["mean_puck_velocity_after_hit"] = i["mean_puck_vel_after_hit"]
        #         i["mean_compute_time"] = i["mean_compute_time_ms"]
        #         i["max_compute_time"] = i["max_compute_time_ms"]

        #     evals_in_info = ["reward_sparse", "episode_length", "has_hit", "has_hit_step", "has_scored", "has_scored_step", "min_dist_ee_puck", "min_dist_puck_goal", "max_puck_velocity_after_hit", "mean_puck_velocity_after_hit", ]
                
        # elif self.done_infos[0]["task"] == "prepare":
        #     # Remap
        #     for i in self.done_infos:
        #         i["reward_sparse"] = mean_reward
        #         i["episode_length"] = mean_ep_length
        #         i["mean_compute_time"] = i["mean_compute_time_ms"]
        #         i["max_compute_time"] = i["max_compute_time_ms"]
            
        #     evals_in_info = ["reward_sparse", "episode_length", "success", "has_hit", "has_hit_step"]
        # # TODO temporary for sure xdxd
        # elif self.done_infos[0]["task"] == "defend":
        #     return True
        # elif self.done_infos[0]["task"] == "tournament":
        #     return True

        # for eval in evals_in_info:
        #     val = np.mean([i[eval] for i in self.done_infos])
        #     self.logger.record(f"eval/{eval}", val)

        # constraints_in_info = ["max_j_pos_violation", "max_j_vel_violation", "max_ee_x_violation", "max_ee_y_violation", "max_ee_z_violation", "num_j_pos_violation", "num_j_vel_violation", "num_ee_x_violation", "num_ee_y_violation", "num_ee_z_violation", "max_compute_time", "mean_compute_time"]

        # for constraint in constraints_in_info:
        #     val = np.mean([i[constraint] for i in self.done_infos])
        #     self.logger.record(f"constraint/{constraint}", val)

        return True