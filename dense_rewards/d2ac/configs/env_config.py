from brax import envs
from brax.envs.wrappers import gym as gym_wrapper
from brax.envs.wrappers import torch as torch_wrapper

from d2ac.utils import torch_utils
from d2ac.utils.dm_utils import TorchDMControlEnv


def add_config(parser):
    parser.add_argument(
        "--env",
        type=str,
        default="brax:halfcheetah",
    )
    parser.add_argument(
        "--env_backend",
        type=str,
        default="positional",
        choices=["generalized", "positional", "spring"],
    )
    parser.add_argument("--episode_length", type=int, default=1000)
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--num_test_envs", type=int, default=4)
    parser.add_argument("--num_test_episodes", type=int, default=3)
    parser.add_argument("--max_env_steps", type=int, default=int(5e6))
    parser.add_argument("--envs_on_cuda", action="store_true")


def create_train_test_envs(args):
    if args.env.startswith("brax"):
        auto_reset = False
        action_repeat = 1
        assert len(args.env.split(":")) == 2, "Invalid environment name"
        domain_name = args.env.split(":")[1]
        env = gym_wrapper.VectorGymWrapper(
            envs.create(
                domain_name,
                batch_size=args.num_envs,
                episode_length=args.episode_length,
                backend=args.env_backend,
                auto_reset=auto_reset,
                action_repeat=action_repeat,
            )  # type: ignore
        )
        if args.envs_on_cuda:
            env = torch_wrapper.TorchWrapper(env, device=torch_utils.device)
        else:
            env = torch_wrapper.TorchWrapper(env)
        test_env = gym_wrapper.VectorGymWrapper(
            envs.create(
                domain_name,
                batch_size=args.num_test_envs,
                episode_length=args.episode_length,
                backend=args.env_backend,
                auto_reset=auto_reset,
                action_repeat=action_repeat,
            )  # type: ignore
        )
        if args.envs_on_cuda:
            test_env = torch_wrapper.TorchWrapper(test_env, device=torch_utils.device)
        else:
            test_env = torch_wrapper.TorchWrapper(test_env)
    elif args.env.startswith("dm"):
        assert len(args.env.split(":")) == 3, "Invalid environment name"
        domain_name = args.env.split(":")[1]
        task_name = args.env.split(":")[2]
        print(f"creating env {domain_name} {task_name} ...")
        env = TorchDMControlEnv(domain_name, task_name, num_envs=args.num_envs)
        print("created train env")
        test_env = TorchDMControlEnv(
            domain_name, task_name, num_envs=args.num_test_envs
        )
        print("created test env")
    else:
        raise NotImplementedError

    return env, test_env
