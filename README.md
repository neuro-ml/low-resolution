# Accelerating 3D Medical Image Segmentation by Adaptive Small-Scale Target Localization
Code release for the [paper](https://www.mdpi.com/2313-433X/7/2/35)

<div align="center">
<img src="https://user-images.githubusercontent.com/25771270/139200236-e20097b7-7a94-49b5-9be3-3af0411038b7.png" width="800">
</div>

## Table of Contents
* [Repository Structure](#repository-structure)
* [Installation](#installation)
* [Experiment Reproduction](#experiment-reproduction)
* [References](#references)

## Repository Structure

[tbd]

Time performance diagrams could be built `notebook/results`.

The pre-trained models' weights can be found in the `model` folder. She source code for each of them is located at the `lowres/model` folder. The hyperparameters for these models (e.g., patch_size, batch_size, etc.) are stored in `*.config` files at `config/assets/model`.

Pre-trained ModelX8 (for LUNA16) could be found in 
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

Later we assume that the directory is `~/`

### LUNA16 Challenge Dataset

One of the datasets we used for models training and comparison is a LUNA16 Challenge dataset [[1]](#1). If you want to reproduce the results you need to download and preprocess it in a certain way.

#### Downloading

To download it use the [script](lowres/dataset/luna/downloader.py) that will get not only the data from the challenge, but also additional lung nodules annotaions from the LIDC-IDRI database [[2]](#2), that contains all the series from the LUNA16 Challenge dataset.

The only argument must be given is the absoulte path to the folder, the data will be downloaded, e.g. `/mount/hdd/LUNA16_raw`.

```
python ~/low-resolution/lowres/dataset/luna/downloader.py /mount/hdd/LUNA16_raw
```

If there were some problems during download, the script can be run one more time to get the missing or corrupted archives.


#### Extraction

After all the archives were succesfully downloaded, they need to be unpacked inside the same folder, uisng e.g. 7za:

```
7za x *.zip
```

### Arranging

Experimention with the raw data is not so handy, so we will use [bev](https://github.com/neuro-ml/bev) library in order to prepare it. The upsides of using the library are given in the README.

First you need to setup bev storage and repository ("Creating a repository" page in the [wiki](https://github.com/neuro-ml/bev/wiki)). After that you can execute the prepared scripts for raw data processing:

```
python ~/low-resolution/lowres/dataset/luna/images.py /mount/hdd/LUNA16_raw
python ~/low-resolution/lowres/dataset/luna/lungs_mask.py /mount/hdd/LUNA16_raw
python ~/low-resolution/lowres/dataset/luna/lung_nodules_mask.py /mount/hdd/LUNA16_raw
```

Every script will process the corresponding instances and put the result in the folder `LUNA16_processed` near the folder with the raw files (in our example it is `/mount/hdd/LUNA16_raw`)

Finally, we need to add the procesed data to the created storage:

```
bev /mount/hdd/LUNA16_processed/image ~/low-resolution/assets
bev /mount/hdd/LUNA16_processed/lungs_mask ~/low-resolution/assets
bev /mount/hdd/LUNA16_processed/lung_nodules_mask ~/low-resolution/assets
```

After that you will see modified `.hash` files that points to your local storage. At this poit all the datawork is finished.

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

## References
<a id="1">[1]</a> Setio A. A. A. et al. Validation, comparison, and combination of algorithms for automatic detection of pulmonary nodules in computed tomography images: the LUNA16 challenge //Medical image analysis. – 2017. – Т. 42. – С. 1-13.

<a id="2">[2]</a> Armato III, S. G., McLennan, G., Bidaut, L., McNitt-Gray, M. F., Meyer, C. R., Reeves, A. P., Zhao, B., Aberle, D. R., Henschke, C. I., Hoffman, E. A., Kazerooni, E. A., MacMahon, H., Van Beek, E. J. R., Yankelevitz, D., Biancardi, A. M., Bland, P. H., Brown, M. S., Engelmann, R. M., Laderach, G. E., Max, D., Pais, R. C. , Qing, D. P. Y. , Roberts, R. Y., Smith, A. R., Starkey, A., Batra, P., Caligiuri, P., Farooqi, A., Gladish, G. W., Jude, C. M., Munden, R. F., Petkovska, I., Quint, L. E., Schwartz, L. H., Sundaram, B., Dodd, L. E., Fenimore, C., Gur, D., Petrick, N., Freymann, J., Kirby, J., Hughes, B., Casteele, A. V., Gupte, S., Sallam, M., Heath, M. D., Kuhn, M. H., Dharaiya, E., Burns, R., Fryd, D. S., Salganicoff, M., Anand, V., Shreter, U., Vastagh, S., Croft, B. Y., Clarke, L. P. (2015). Data From LIDC-IDRI [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX
