import random


def add_config(parser):
    parser.add_argument("--seed", type=int, default=random.randint(0, 10003))
    parser.add_argument("--n_initial_rollouts", type=int, default=100)

    parser.add_argument("--n_epochs", type=int, default=5000)
    parser.add_argument("--n_cycles", type=int, default=10)
    parser.add_argument("--optimize_every", type=int, default=2)
    parser.add_argument("--n_batches", type=int, default=1)

    parser.add_argument("--replay_size", type=int, default=2500000)
    parser.add_argument("--resume_path", type=str, default="")

    parser.add_argument("--demo_length", type=int, default=10)
    parser.add_argument("--play", action="store_true")
    parser.add_argument("--log_videos", action="store_true")

    parser.add_argument("--method_name", type=str, default="not_specified")
    parser.add_argument("--run_name", type=str, default="not_specified")
