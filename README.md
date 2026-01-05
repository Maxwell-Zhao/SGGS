# SG-GS: Photo-realistic Animatable Human Avatars with Semantically-Guided Gaussian Splatting
## [Paper](https://arxiv.org/pdf/2408.09665) | [Project Page](https://sggs-projectpage.github.io/)

You can find detailed usage instructions for using pretrained models and training your own models below.

If you find our code useful, please cite:

```bibtex
@article{zhao2024sg,
  title={SG-GS: Photo-realistic Animatable Human Avatars with Semantically-Guided Gaussian Splatting},
  author={Zhao, Haoyu and Yang, Chen and Wang, Hao and Zhao, Xingyue and Shen, Wei},
  journal={arXiv preprint arXiv:2408.09665},
  year={2024}
}
```

## Installation
### Environment Setup
This repository has been tested on the following platform:
1) Python 3.7.13, PyTorch 1.12.1 with CUDA 11.6 and cuDNN 8.3.2, Ubuntu 22.04/CentOS 7.9.2009

To clone the repo, run either:
```
git clone --recursive https://github.com/mikeqzy/3dgs-avatar-release.git
```
or
```
git clone https://github.com/mikeqzy/3dgs-avatar-release.git
cd 3dgs-avatar-release
git submodule update --init --recursive
```

Next, you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `3dgs-avatar` using
```
conda env create -f environment.yml
conda activate 3dgs-avatar
# install tinycudann
pip install git+https://github.com/NVlabs/tiny-cuda-nn/#subdirectory=bindings/torch
```

### SMPL Setup
Download `SMPL v1.0 for Python 2.7` from [SMPL website](https://smpl.is.tue.mpg.de/) (for male and female models), and `SMPLIFY_CODE_V2.ZIP` from [SMPLify website](https://smplify.is.tue.mpg.de/) (for the neutral model). After downloading, inside `SMPL_python_v.1.0.0.zip`, male and female models are `smpl/models/basicmodel_m_lbs_10_207_0_v1.0.0.pkl` and `smpl/models/basicModel_f_lbs_10_207_0_v1.0.0.pkl`, respectively. Inside `mpips_smplify_public_v2.zip`, the neutral model is `smplify_public/code/models/basicModel_neutral_lbs_10_207_0_v1.0.0.pkl`. Remove the chumpy objects in these .pkl models using [this code](https://github.com/vchoutas/smplx/tree/master/tools) under a Python 2 environment (you can create such an environment with conda). Finally, rename the newly generated .pkl files and copy them to subdirectories under `./body_models/smpl/`. Eventually, the `./body_models` folder should have the following structure:
```
body_models
 └-- smpl
    ├-- male
    |   └-- model.pkl
    ├-- female
    |   └-- model.pkl
    └-- neutral
        └-- model.pkl
```

Then, run the following script to extract necessary SMPL parameters used in our code:
```
python extract_smpl_parameters.py
```
The extracted SMPL parameters will be saved into `./body_models/misc/`.

## Dataset preparation
Due to license issues, we cannot publicly distribute our preprocessed ZJU-MoCap and PeopleSnapshot data. 
Please follow the instructions of [ARAH](https://github.com/taconite/arah-release) to download and preprocess the datasets.
For PeopleSnapshot, we use the optimized SMPL parameters from Anim-NeRF [here](https://drive.google.com/drive/folders/1tbBJYstNfFaIpG-WBT6BnOOErqYUjn6V?usp=drive_link).


## Training
To train new networks from scratch, run
```shell
# ZJU-MoCap
python train.py dataset=zjumocap_377_mono
# PeopleSnapshot
python train.py dataset=ps_female_3 option=iter30k pose_correction=none 
```
To train on a different subject, simply choose from the configs in `configs/dataset/`.

We use [wandb](https://wandb.ai) for online logging, which is free of charge but needs online registration.

## Evaluation
To evaluate the method for a specified subject, run
```shell
# ZJU-MoCap
python render.py mode=test dataset.test_mode=view dataset=zjumocap_377_mono
# PeopleSnapshot
python render.py mode=test dataset.test_mode=pose pose_correction=none dataset=ps_female_3
```

## Test on out-of-distribution poses
First, please download the preprocessed AIST++ and AMASS sequence for subjects in ZJU-MoCap [here](https://drive.google.com/drive/folders/17vGpq6XGa7YYQKU4O1pI4jCMbcEXJjOI?usp=drive_link)
and extract under the corresponding subject folder `${ZJU_ROOT}/CoreView_${SUBJECT}`.

To animate the subject under out-of-distribution poses, run
```shell
python render.py mode=predict dataset.predict_seq=0 dataset=zjumocap_377_mono
```

We provide four preprocessed sequences for each subject of ZJU-MoCap, 
which can be specified by setting `dataset.predict_seq` to 0,1,2,3, 
where `dataset.predict_seq=3` corresponds to the canonical rendering.

Currently, the code only supports animating ZJU-MoCap models for out-of-distribution models.

## Acknowledgement
This project is built on source codes from [3DGS](https://github.com/graphdeco-inria/gaussian-splatting) and [3DGS-Avatar](https://github.com/mikeqzy/3dgs-avatar-release). 
We also use the data preprocessing script and part of the network implementations from [ARAH](https://github.com/taconite/arah-release).
We sincerely thank these authors for their awesome work.

