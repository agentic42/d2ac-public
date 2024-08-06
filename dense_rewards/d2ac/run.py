import argparse

from d2ac import configs, set_up_training_env
from d2ac.algo.d2ac_algo import TrainLoop

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    configs.env_config.add_config(parser)
    configs.algo_config.add_config(parser)

    parser.add_argument("--exp_name", type=str, default="d2ac_dense")
    parser.add_argument("--hid", type=int, default=256)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--n_models", type=int, default=2)

    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--polyak", type=float, default=0.995)
    parser.add_argument("--polyak_pi", type=float, default=0.995)
    parser.add_argument("--alpha_init", type=float, default=0.2)
    parser.add_argument("--targ_entropy_coef", type=float, default=0.0)

    parser.add_argument("--num_bins", type=int, default=101)
    parser.add_argument("--vmin", type=float, default=-10)
    parser.add_argument("--vmax", type=float, default=10)
    parser.add_argument(
        "--backup_method",
        type=str,
        default="distributional",
        choices=["two_hot", "distributional"],
    )
    parser.add_argument(
        "--discrete_mode", type=str, default="linear", choices=["log", "linear"]
    )

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
    parser.add_argument("--n_time_embed", type=int, default=0)
    parser.add_argument(
        "--scalings",
        type=str,
        default="uniform",
        choices=[
            "uniform",
            "edm",
        ],
    )
    parser.add_argument(
        "--action_sampling_mode",
        type=str,
        default="one_step",
        choices=["sde", "one_step"],
    )

    parser.add_argument("--consistency_model", action="store_true")

    parser.add_argument("--n_sampling_steps_train", type=int, default=5)
    parser.add_argument("--n_sampling_steps_inference", type=int, default=1)
    parser.add_argument("--n_sampling_steps_planning", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--adamw", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--pi_iters", type=int, default=1)

    parser.add_argument("--q_entropy_loss_coef", type=float, default=0.0)
    parser.add_argument("--z_loss_coef", type=float, default=0.0)

    args = parser.parse_args()

    # add derived args
    args.bin_size = (args.vmax - args.vmin) / (args.num_bins - 1)

    env, test_env, resume_path, save_path = set_up_training_env(args)

    algo = TrainLoop(
        env,
        test_env,
        args,
        resume_path=resume_path,
        save_path=save_path,
        algo_device="cpu",
    )
    algo.start()
