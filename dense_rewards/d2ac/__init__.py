import os
import os.path as osp
import random

import numpy as np
import torch

from d2ac import configs
from d2ac.utils import logger, torch_utils
from d2ac.utils.run_utils import dump_config, get_exp_name, setup_logging

LOCAL_BASE_PATH = os.path.join(os.getcwd(), "exp")
LOG_FOLDER_NAME = "state"


def set_up_training_env(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch_utils.use_cuda:
        torch.cuda.manual_seed_all(seed)

    env_name = args.env
    exp_name = get_exp_name(env_name.replace(":", "-") + "-" + args.exp_name, seed)
    # override exp_name
    args.exp_name = exp_name

    # log some useful info
    args.device = str(torch_utils.device)
    output_dir = os.path.join(LOCAL_BASE_PATH, exp_name)
    os.makedirs(output_dir, exist_ok=True)

    logger.configure(
        dir=output_dir,
        format_strs=[
            "csv",
            "stdout",
            "tensorboard",
        ],
    )

    resume_folder = None
    resume_path = None
    if len(args.resume_path) > 0:
        resume_folder = osp.join(LOCAL_BASE_PATH, args.resume_path)
        print(f"Resuming from local folder: {resume_folder}")
        resume_path = osp.join(resume_folder, LOG_FOLDER_NAME)

    save_path = osp.join(output_dir, LOG_FOLDER_NAME)
    os.makedirs(save_path, exist_ok=True)
    print("save_path = %s" % save_path)

    setup_logging(output_dir)
    print("Set up logging at %s" % output_dir)

    env, test_env = configs.env_config.create_train_test_envs(args)
    print("Created envs")

    dump_config(config=dict(vars(args)), output_dir=output_dir)

    if resume_folder is not None:
        configs.resume_config.check_resume_config(resume_folder, args)

    return env, test_env, resume_path, save_path
