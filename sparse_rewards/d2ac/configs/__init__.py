import os

from d2ac.configs import algo_config, env_config, resume_config

NUM_THREADS = 8

os.environ["OMP_NUM_THREADS"] = str(NUM_THREADS)
os.environ["MKL_NUM_THREADS"] = str(NUM_THREADS)
os.environ["IN_MPI"] = str(NUM_THREADS)
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"
