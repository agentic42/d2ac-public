import gymnasium as gym

from d2ac.utils import vec_env


def add_config(parser):
    parser.add_argument("--env", type=str, default="FetchPush-v2")

    parser.add_argument("--n_workers", type=int, default=12)
    parser.add_argument("--n_test_rollouts", type=int, default=15)
    parser.add_argument("--max_env_steps", type=int, default=int(5e6))


def get_env_params(env):
    # obs, _ = env.reset()
    obs = env.reset()
    obs = obs[0] if type(obs) == tuple else obs
    params = {
        "obs": obs["observation"].shape[0],
        "goal": obs["desired_goal"].shape[0],
        "action": env.action_space.shape[0],
        "action_max": env.action_space.high[0],
        "max_timesteps": env._max_episode_steps,
    }
    return params


def get_env_with_id(num_envs, env_id):
    vec_fn = vec_env.SubprocVecEnv
    return vec_fn([lambda: gym.make(env_id) for _ in range(num_envs)])


def get_env_with_fn(num_envs, env_fn, *args, **kwargs):
    vec_fn = vec_env.SubprocVecEnv
    return vec_fn([lambda: env_fn(*args, **kwargs) for _ in range(num_envs)])


ENV_MATCHING = {
    "gym:fetch:push": "FetchPush-v2",
    "gym:fetch:slide": "FetchSlide-v2",
    "gym:fetch:pick_and_place": "FetchPickAndPlace-v2",
    "gym:fetch:reach": "FetchReach-v2",
    "gym:hand:reach": "HandReach-v1",
    "gym:hand:manipulate_block": "HandManipulateBlock-v1",
    "gym:hand:manipulate_blockz": "HandManipulateBlockRotateZ-v1",
    "gym:hand:manipulate_block_full": "HandManipulateBlockFull-v1",
    "gym:hand:manipulate_blockz_sensor": "HandManipulateBlockRotateZ_BooleanTouchSensors-v1",
    "gym:hand:manipulate_block_full_sensor": "HandManipulateBlockFull_BooleanTouchSensors-v1",
    "gym:hand:manipulate_egg": "HandManipulateEgg-v1",
    "gym:hand:manipulate_egg_rotate": "HandManipulateEggRotate-v1",
    "gym:hand:manipulate_egg_full": "HandManipulateEggFull-v1",
    "gym:hand:manipulate_egg_sensor": "HandManipulateEgg_BooleanTouchSensors-v1",
    "gym:hand:manipulate_egg_rotate_sensor": "HandManipulateEggRotate_BooleanTouchSensors-v1",
    "gym:hand:manipulate_egg_full_sensor": "HandManipulateEggFull_BooleanTouchSensors-v1",
    "gym:hand:manipulate_pen": "HandManipulatePen-v1",
    "gym:hand:manipulate_pen_rotate": "HandManipulatePenRotate-v1",
    "gym:hand:manipulate_pen_full": "HandManipulatePenFull-v1",
    "gym:hand:manipulate_pen_sensor": "HandManipulatePen_BooleanTouchSensors-v1",
    "gym:hand:manipulate_pen_rotate_sensor": "HandManipulatePenRotate_BooleanTouchSensors-v1",
    "gym:hand:manipulate_pen_full_sensor": "HandManipulatePenFull_BooleanTouchSensors-v1",
}


def create_envs(args):
    assert args.env in ENV_MATCHING.keys(), f"Unknown environment {args.env}"
    matched_env_name = ENV_MATCHING[args.env]
    print(f"matched env {args.env} to {matched_env_name}")

    env = gym.make(matched_env_name)
    env_params = get_env_params(env)
    assert hasattr(env, "compute_reward")
    reward_func = env.compute_reward

    if args.n_workers > 1:
        env = get_env_with_id(num_envs=args.n_workers, env_id=matched_env_name)

    render_env = gym.make(matched_env_name, render_mode="rgb_array")
    print("Successfully created environment")

    return env, env_params, reward_func, render_env
