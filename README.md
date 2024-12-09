# 0. If you just want the glassDBN model source code
Go to `src/models/DBNglassFIX.py`.
`glassDBN`, `default_HPs`, and pretrained weights in `assets/model_weights/DBNglassFIX_ukb.pt` is what you need.

# 1. Requirements
```bash
conda create -n glass python=3.12
conda activate glass
conda install pytorch torchvision torchaudio pytorch-cuda=12.4 -c pytorch -c nvidia
pip install -r requirements.txt
```

## Optional
- `prefix`: custom prefix for the project
    - default prefix is UTC time
    - appears in the name of logs directory and the name of WandB project
    - `exp` mode runs with custom prefix will use HPs from `tune` mode runs with the same prefix
        - unless model.default_HP is set to `True`
- `permute`: whether TS models should be trained on time-reshuffled data
    - set to `permute=Multiple` to permute
- `HP_path`: path to custom hyperparams to load
- `follow_splits`: path to an experiment which splist you want to replicate


# `scripts/run_experiments.py` options:
## Required:
- `mode`: 
    - `tune` - tune mode: run multiple experiments with different hyperparams
    - `exp` - experiment mode: run experiments with the best hyperparams found in the `tune` mode, or with default hyperparams `default_HPs` is set to `True`

- `model`: model for the experiment. Models' config files can be found at `src/conf/model`, and their sourse code is located at `src/models`

- `dataset`: dataset for the experiments. Datasets' config files can be found at `src/conf/dataset`, and their loading scripts are located at `src/datasets`.
