import argparse

from d2ac import configs, set_up_training_env
from d2ac.agent.d2ac_agent import GoalConditionedD2AC as Agent
from d2ac.runner.d2ac_runner import D2ACRunner as Runner


def launch(args, model_class: str = "base"):
    print("env name:", args.env)

    training_dict = set_up_training_env(args)
    env = training_dict["env"]
    env_params = training_dict["env_params"]
    reward_func = training_dict["reward_func"]
    render_env = training_dict["render_env"]
    save_path = training_dict["save_path"]
    resume_path = training_dict["resume_path"]
    video_path = training_dict["video_path"]

    agent = Agent(env_params, args)
    runner = Runner(
        env=env,
        env_params=env_params,
        render_env=render_env,
        args=args,
        agent=agent,
        reward_func=reward_func,
        resume_path=resume_path,
        video_path=video_path,
        save_path=save_path,
    )
    return runner


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    configs.env_config.add_config(parser)
    configs.algo_config.add_config(parser)

    parser.add_argument("--exp_name", type=str, default="d2ac_sparse")
    parser.add_argument("--model_class", type=str, default="base")

    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_models", type=int, default=2)

    parser.add_argument("--noise_eps", type=float, default=0.1)
    parser.add_argument("--random_eps", type=float, default=0.2)

    parser.add_argument("--future_p", type=float, default=0.8)
    parser.add_argument("--batch_size", type=int, default=1024)

    parser.add_argument("--clip_inputs", action="store_true")
    parser.add_argument("--clip_obs", type=float, default=200)

    parser.add_argument("--normalize_inputs", action="store_true")
    parser.add_argument("--clip_range", type=float, default=5)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--polyak", type=float, default=0.995)
    parser.add_argument("--polyak_pi", type=float, default=0.995)
    parser.add_argument("--target_update_freq", type=int, default=1)
    parser.add_argument("--alpha_init", type=float, default=0.2)
    parser.add_argument("--targ_entropy_coef", type=float, default=0.0)

    parser.add_argument("--num_bins", type=int, default=101)
    parser.add_argument("--vmin", type=float, default=-50)
    parser.add_argument("--vmax", type=float, default=0)

    parser.add_argument(
        "--weight_schedule",
        type=str,
        default="uniform",
        choices=[
            "uniform",
            "edm",
        ],
    )
    parser.add_argument("--sigma_data", type=float, default=1.0)
    parser.add_argument("--sigma_min", type=float, default=0.05)
    parser.add_argument("--sigma_max", type=float, default=2.0)
    parser.add_argument("--n_time_embed", type=int, default=32)

    parser.add_argument("--n_sampling_steps_train", type=int, default=5)
    parser.add_argument("--n_sampling_steps_inference", type=int, default=2)
    parser.add_argument("--n_sampling_steps_planning", type=int, default=5)

    parser.add_argument("--adamw", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--pi_iters", type=int, default=2)

    parser.add_argument("--q_entropy_loss_coef", type=float, default=0.0)
    parser.add_argument("--z_loss_coef", type=float, default=0.0)

    parser.add_argument("--action_l2", type=float, default=0.00)
    parser.add_argument("--lr_actor", type=float, default=0.001)
    parser.add_argument("--lr_critic", type=float, default=0.001)

    args = parser.parse_args()

    args.bin_size = (args.vmax - args.vmin) / (args.num_bins - 1)
    args.backup_method = "distributional"
    args.action_sampling_mode = "one_step"
    args.scalings = "edm"

    runner = launch(args, model_class=args.model_class)
    runner.run()
