# Accelerating 3D Medical Image Segmentation by Adaptive Small-Scale Target Localization
Code release

## Table of Contents
* [Repository Structure](#repository-structure)
* [Experiment Reproduction](#experiment-reproduction)

## Repository Structure
```
├── config
│   ├── assets
│   └── exp_holdout
├── lowres
│   ├── benchmark
│   │   ├── benchmark_time.sh
│   │   └── model_predict.py
│   ├── dataset
│   │   └── luna.py
│   ├── model
│   ├── path.py
│   └── ...
├── model
├── notebook
│   ├── data_preprocessing
│   │   ├── LUNA16_download.ipynb
│   │   └── LUNA16_preprocessing.ipynb
│   └── time_performance.ipynb
└── README.md
```
Download and preprocessing for the LUNA16 dataset can be done via IPython notebooks located at `notebook/data_preprocessing`. Also, the time performance diagrams could be built `notebook/results`.

The pre-trained models' weights can be found in the `model` folder. She source code for each of them is located at the `lowres/model` folder. The hyperparameters for these models (e.g., patch_size, batch_size, etc.) are stored in `*.config` files at `config/assets/model`.

All the necessary paths should be specified inside `lowres/path.py`. These are:
- `luna_raw_path` -- where to download the raw LUNA16 files
- `luna_data_path` -- where the preprocessed, structured files will be saved
- `path_to_pretrained_model_x8` -- where the ModelX8 weights are located

Alternatively pretrained ModelX8 (for LUNA16) could be found in 
`~/low-resolution/model/model_x8.pth`.

Finally, the script `lowres/benchmark/benchmark_time.sh` can estimate the time, the given model spend to process the 
chosen amount of scans. By default the whole dataset used, so the desired number can be specified inside `lowres/benchmark/model_predict.sh`. The single argument of the script -- the desired number of threads (e.g., `8`).

## Installation
Execute from the directory you want the repo to be installed:

```
git clone https://github.com/neuro-ml/low-resolution
cd low-resolution
pip install -e .
```

## Experiment Reproduction
To run a single experiment please follow the steps below:

First, the experiment structure must be created:
```
python -m dpipe build_experiment --config_path "$1" --experiment_path "$2"
```

where the first argument is a path to the `.config` file e.g. 
`"~/low-resolution/config/exp_holdout/unet3d.config"`
and the second is a path to the folder, where the experiment structure will be organized e.g.
`"~/unet3d_experiment/"`.

Then, to run an experiment please go to the experiment folder inside the created structure:
```
cd ~/unet3d_experiment/experiment_0/
```
and call the following command to start the experiment:
```
python -m dpipe run_experiment --config_path "../resources.config"
```
where `resources.config` is the general `.config` file of the experiment.


