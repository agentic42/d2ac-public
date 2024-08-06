# D2AC for Sparse Reward Environments 

## Development Environment

The complete list of dependencies is in `environment_linux.yml`.
```
conda env create --file environment_linux.yml
```

### Trouble shooting

If the above fails, then you need to manually build your environment following the guide below.

```
conda create -n d2ac_sparse python=3.10
conda activate d2ac_sparse
```

First, install PyTorch:
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

The installed versions should be: `torch-2.1.2+cu118 torchaudio-2.1.2+cu118 torchvision-0.16.2+cu118 triton-2.1.0.`

```
pip3 install black
pip install gymnasium-robotics
pip install tensorboardX
pip3 install pandas
pip3 install opencv-python
```

## Example Commands

To train a Fetch Pick-and-Place policy:
```
python -m d2ac.run --env gym:fetch:pick_and_place --clip_inputs --normalize_inputs --adamw --target_update_freq 10 --max_env_steps 2000000
```

To train a Shadow Hand Manipulate model to rotate an Egg:
```
python -m d2ac.run --env gym:hand:manipulate_egg_rotate --n_workers 20 --gamma 0.99 --clip_inputs --normalize_inputs --adamw --target_update_freq 10 --max_env_steps 5000000
```