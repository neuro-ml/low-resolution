# Accelerating 3D Medical Image Segmentation by Adaptive Small-Scale Target Localization
Code release

## Table of Contents
* [Requirements](#requirements)
* [Repository Structure](#repository-structure)
* [Experiment Reproduction](#experiment-reproduction)


## Requirements
- [Python](https://www.python.org) (v3.6 or later)
- [Deep Pipe](https://github.com/neuro-ml/deep_pipe) (commit: [4383211ea312c098d710fbeacc05151e10a27e80](https://github.com/neuro-ml/deep_pipe/tree/4383211ea312c098d710fbeacc05151e10a27e80))
- [imageio](https://pypi.org/project/imageio/) (v 2.8.0)
- [NiBabel](https://pypi.org/project/nibabel/) (v3.0.2)
- [NumPy](http://numpy.org/) (v1.17.0 or later)
- [OpenCV python](https://pypi.org/project/opencv-python/) (v4.2.0.32)
- [Pandas](https://pandas.pydata.org/) (v1.0.1 or later)
- [pdp](https://pypi.org/project/pdp/) (v 0.3.0)
- [pydicom](https://pypi.org/project/pydicom/) (v 1.4.2)
- [resource-manager](https://pypi.org/project/resource-manager/) (v 0.11.1)
- [SciPy library](https://www.scipy.org/scipylib/index.html) (v0.19.0 or later)
- [scikit-image](https://scikit-image.org) (v0.15.0 or later)
- [Simple ITK](http://www.simpleitk.org/) (v1.2.4)
- [torch](https://pypi.org/project/torch/) (v1.1.0 or later)
- [tqdm](https://tqdm.github.io) (v4.32.0 or later)

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


