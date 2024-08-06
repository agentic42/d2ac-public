import random


def add_config(parser):
    parser.add_argument("--seed", type=int, default=random.randint(0, 10003))
    parser.add_argument("--epochs", type=int, default=5000)
    parser.add_argument("--steps_per_epoch", type=int, default=4000)
    parser.add_argument("--start_steps", type=int, default=10000)
    parser.add_argument("--update_after", type=int, default=1000)
    parser.add_argument("--update_ratio", type=float, default=1.0)

    parser.add_argument("--update_every", type=int, default=1)
    parser.add_argument("--replay_size", type=int, default=int(1e6))
    parser.add_argument("--resume_path", type=str, default="")
    parser.add_argument("--save_freq", type=int, default=1)

    parser.add_argument("--method_name", type=str, default="not_specified")
    parser.add_argument("--run_name", type=str, default="not_specified")
