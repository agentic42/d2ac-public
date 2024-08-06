from d2ac.replay.core import Replay


def build_from_args(env_params, args, reward_func, model_class: str = "base"):
    if model_class in ["base", "bilinear"]:
        return Replay(env_params, args, reward_func)
    else:
        raise NotImplementedError
