# D2AC on Dense-Reward Environments 

## Development Environment

The complete list of dependencies is in `environment_linux.yml`.
```
conda env create --file environment_linux.yml
```

### Trouble shooting

If the above fails, then you need to manually build your environment following the guide below.
```
conda create -n d2ac python=3.10
conda activate d2ac
```

### PyTorch

To install a version of PyTorch:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
The installed versions are: `torch-2.1.2+cu118 torchaudio-2.1.2+cu118 torchvision-0.16.2+cu118 triton-2.1.0`.

### DeepMind Control Suite

```
pip install --upgrade mujoco
pip install dm_control
pip install dm-env
pip install brax
```

### Jax

```
pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### Additional Packages

```
pip install black
pip install pandas
```

## Example Commands

To train a model on Dog-Run environment:
```
python -m d2ac.run --env dm:dog:run --num_envs 4 --num_test_envs 4 --max_env_steps 3000000 --targ_entropy_coef 0.0 --backup_method distributional --discrete_mode linear --vmin -1000.0 --vmax 1000. --num_bins 201 --n_sampling_steps_train 5 --n_sampling_steps_inference 2 --n_sampling_steps_planning 5 --action_sampling_mode one_step --scalings edm --n_time_embed 32 --pi_iters 2 --adamw
```

You can train models on other environments by simply changing the `--env` flag.