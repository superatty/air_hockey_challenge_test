
from modified_envs.env_defend import EnvDefend
import functools
from brax.training.agents.ppo import train as ppo
import os
import jax
import cv2

if __name__ == "__main__":
    # # Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
    # xla_flags = os.environ.get('XLA_FLAGS', '')
    # xla_flags += ' --xla_gpu_triton_gemm_any=True'
    # os.environ['XLA_FLAGS'] = xla_flags

    env = EnvDefend()

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=0,
        episode_length=150,
        normalize_observations=True,
        # discounting=1,
        # learning_rate=5e-5,
        # num_envs=4096,
        # batch_size=256,
        # seed=0,
        # log_training_metrics=True,
        # run_evals=True,
        # num_evals=5,
        # unroll_length=512,
        # save_checkpoint_path="/home/donat/projects/air_hockey_challenge/baseline/ppo_baseline_agent/checkpoints",
        restore_checkpoint_path="/home/donat/projects/air_hockey_challenge/baseline/ppo_baseline_agent/checkpoints/000201326592",
    )

    # def progress(num_steps, metrics):
    #     print(f"Step {num_steps}")
            
    #     if 'eval/episode_reward' in metrics:
    #         print(f"Reward: {metrics['eval/episode_reward']:.2f}, Episode Length: {metrics['eval/avg_episode_length']:.2f}")

    # print("Starting training...")

    make_inference_fn, params, _ = train_fn(environment=env)
    # make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

    # print("Training completed.")

    inference_fn = make_inference_fn(params)
    jit_inference_fn = jax.jit(inference_fn)

    eval_env = EnvDefend()
    jit_reset = jax.jit(eval_env.reset)
    jit_step = jax.jit(eval_env.step)
    rng = jax.random.PRNGKey(0)

    state = jit_reset(rng)
    rollout = [state.pipeline_state]

    n_steps = 500

    for i in range(n_steps):
        act_rng, rng = jax.random.split(rng)
        ctrl, _ = jit_inference_fn(state.obs, act_rng)
        print(f"Action: {ctrl}")
        state = jit_step(state, ctrl)
        rollout.append(state.pipeline_state)
        
        print(state.reward)

        if state.done:
            break

    imgs = env.render(rollout)

    input("Press Enter to watch the rollout...")

    for img in imgs:
        cv2.imshow("Air Hockey", img[:, :, ::-1])
        cv2.waitKey(0)
    cv2.destroyAllWindows()
