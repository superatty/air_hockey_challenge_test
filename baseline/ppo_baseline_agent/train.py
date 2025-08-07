
from modified_envs.env_defend import EnvDefend
import functools
from brax.training.agents.ppo import train as ppo

if __name__ == "__main__":
    # os.environ['XLA_FLAGS'] = (
    #     '--xla_gpu_triton_gemm_any=True '
    #     '--xla_gpu_enable_latency_hiding_scheduler=true '
    # )

    env = EnvDefend()
    
    train_fn = functools.partial(
        ppo.train,
        num_timesteps=2e7,
        episode_length=150,
        normalize_observations=True,
        discounting=1,
        learning_rate=5e-5,
        num_envs=2048,
        batch_size=1024,
        seed=0,
        log_training_metrics=True,
        run_evals=True,
        num_evals=5,
        unroll_length=10,
        num_minibatches=24,
        num_updates_per_batch=8,
        max_grad_norm=0.5,
        
        # max_grad_norm=0.5,
        # entropy_cost=0,
        save_checkpoint_path="/home/donat/projects/air_hockey_challenge/baseline/ppo_baseline_agent/checkpoints",
        # restore_checkpoint_path="/home/donat/projects/air_hockey_challenge/baseline/ppo_baseline_agent/checkpoints",
    )

    def progress(num_steps, metrics):
        print(f"Step {num_steps}")
            
        if 'eval/episode_reward' in metrics:
            print(f"Reward: {metrics['eval/episode_reward']:.2f}, Episode Length: {metrics['eval/avg_episode_length']:.2f}")

    print("Starting training...")

    # make_inference_fn, params, _ = train_fn(environment=env)
    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

    print("Training completed.")

    # inference_fn = make_inference_fn(params)
    # jit_inference_fn = jax.jit(inference_fn)

    # eval_env = EnvDefend()
    # jit_reset = jax.jit(eval_env.reset)
    # jit_step = jax.jit(eval_env.step)
    # rng = jax.random.PRNGKey(0)

    # state = jit_reset(rng)
    # rollout = [state.pipeline_state]

    # n_steps = 500

    # for i in range(n_steps):
    #     act_rng, rng = jax.random.split(rng)
    #     ctrl, _ = jit_inference_fn(state.obs, act_rng)
    #     state = jit_step(state, ctrl)
    #     rollout.append(state.pipeline_state)

    #     if state.done:
    #         break

    # imgs = env.render(rollout)

    # input("Press Enter to watch the rollout...")

    # for img in imgs:
    #     cv2.imshow("Air Hockey", img[:, :, ::-1])
    #     cv2.waitKey(0)
    # cv2.destroyAllWindows()
