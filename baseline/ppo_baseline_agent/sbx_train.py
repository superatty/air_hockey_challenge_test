from pathlib import Path
from sbx import PPO
from gymnasium import spaces
import numpy as np

from baseline.ppo_baseline_agent import train
from baseline.ppo_baseline_agent.utils.brax_wrapper import BraxSB3Wrapper
from baseline.ppo_baseline_agent.modified_envs.env_defend import EnvDefend
from brax.envs.wrappers import training
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.vec_env import sync_envs_normalization

from utils.callbacks import SaveBestModel, CheckpointLog, CustomEvalCallback

import wandb

ENVS = {
    "defend": EnvDefend,
}


def make_environments(env_name: str, num_envs: int, gamma: float, horizon: int):
    train_env = ENVS[env_name]()

    min_action = np.array(train_env.act_low)
    max_action = np.array(train_env.act_high)
    action_space = spaces.Box(min_action, max_action, dtype=np.float32)

    train_env = training.EpisodeWrapper(train_env, horizon, 1)
    train_env = training.VmapWrapper(train_env, num_envs)
    train_env = training.AutoResetWrapper(train_env)

    train_env = BraxSB3Wrapper(train_env, seed=0, keep_infos=False)
    train_env.action_space = action_space
    train_env = VecMonitor(train_env)
    train_env = VecNormalize(train_env, gamma=gamma)

    eval_env = ENVS[env_name]()
    eval_env = training.EpisodeWrapper(eval_env, horizon, 1)
    eval_env = training.VmapWrapper(eval_env, 1)
    eval_env = training.AutoResetWrapper(eval_env)

    eval_env = BraxSB3Wrapper(eval_env, seed=1, keep_infos=False)
    eval_env.action_space = action_space
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, gamma=gamma)

    return train_env, eval_env

def setup_wandb(train_dir: Path, env_name: str):
    wandb.init(
        entity="atalaydonat",
        project="air_hockey_challenge",
        group=env_name,
        mode="online",
        dir=train_dir,
        name=train_dir.name,
        sync_tensorboard=True,
    )

if __name__ == "__main__":
    # Set the environment name and parameters
    env_name = "defend"
    num_envs = 4096
    gamma = 1.0
    horizon = 150
    run_name = "test_9"
    # Define the directory to save the model and normalization stats
    train_dir = Path(
        "/home/donat/projects/air_hockey_challenge/baseline/ppo_baseline_agent/sbx_checkpoints/defend"
    ) / run_name

    if train_dir.exists():
        raise FileExistsError(f"Directory {train_dir} already exists. Please choose a different run name.")

    setup_wandb(train_dir, env_name)

    # Create training and evaluation environments
    train_env, eval_env = make_environments(env_name, num_envs, gamma, horizon)

    # Initialize the PPO model
    model = PPO(
        "MlpPolicy",
        train_env,
        n_steps=10,
        batch_size=512,
        learning_rate=5e-5,
        gamma=gamma,
        n_epochs=10,
        seed=0,
        verbose=1,
        tensorboard_log=train_dir
    )


    custom_log = {
        "best_mean_reward": -1e10,
    }

    sync_envs_normalization(train_env, eval_env)
    checkpoint_callback = CheckpointLog(save_dir=train_dir, custom_log=custom_log, save_freq=int(1e7 / num_envs))
    save_best_model_callback = SaveBestModel(train_dir, custom_log)
    custom_eval_callback = CustomEvalCallback(eval_env, save_best_model_callback, 30, int(1e5 / num_envs), custom_log["best_mean_reward"])
    callbacks = CallbackList([checkpoint_callback, custom_eval_callback])

    # Start training the model
    model.learn(total_timesteps=int(2e8), progress_bar=True, tb_log_name="run", callback=callbacks)