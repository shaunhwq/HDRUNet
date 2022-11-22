# HDRUNet [[Paper Link]](http://arxiv.org/abs/2105.13084)

### HDRUNet: Single Image HDR Reconstruction with Denoising and Dequantization
Xiangyu Chen, [Yihao Liu](https://scholar.google.com.hk/citations?user=WRIYcNwAAAAJ&hl=zh-CN), Zhengwen Zhang, [Yu Qiao](https://scholar.google.com/citations?user=gFtI-8QAAAAJ&hl=zh-CN) and [Chao Dong](https://scholar.google.com.hk/citations?user=OSDCB0UAAAAJ&hl=zh-CN)

#### We won the second place in [NTIRE2021 HDR Challenge](https://data.vision.ee.ethz.ch/cvl/ntire21/) ([Track1: Single Frame](https://competitions.codalab.org/competitions/28161)). The paper is accepted to CVPR2021 Workshop.

<img src="https://raw.githubusercontent.com/chxy95/HDRUNet/master/images/introduction.jpg"/>

#### BibTeX

    @InProceedings{chen2021hdrunet,
        author    = {Chen, Xiangyu and Liu, Yihao and Zhang, Zhengwen and Qiao, Yu and Dong, Chao},
        title     = {HDRUNet: Single Image HDR Reconstruction With Denoising and Dequantization},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
        month     = {June},
        year      = {2021},
        pages     = {354-363}
    }

## Overview
Overview of the network:

<img src="https://raw.githubusercontent.com/chxy95/HDRUNet/master/images/Network_Structure.png" width="600"/>

Overview of the loss function:

```
Tanh_L1(Y, H) = |Tanh(Y) - Tanh(H)|
```

## Getting Started

1. [Dataset](#dataset)
2. [Configuration](#configuration)
3. [How to test](#how-to-test)
4. [How to train](#how-to-train)
5. [Visualization](#visualization)

### Dataset
Register a codalab account and log in, then find the download link on this page:
```
https://competitions.codalab.org/competitions/28161#participate-get-data
```
#### It is strongly recommended to use the data provided by the competition organizer for training and testing, or you need at least a basic understanding of the competition data. Otherwise, you may not get the desired result.

### Configuration
```
pip install -r requirements.txt
```

### Testing (on external images, for cuda-10.2)

Installation
```
conda create -n HDRUNet python=3.8.0
conda activate HDRUNet
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
pip3 install opencv-python tqdm scipy pyyaml
```

Running
```
export LD_LIBRARY_PATH=/data2/shaun/cuda-10.2/lib64:$LD_LIBRARY_PATH
export CUDA_VISIBLE_DEVICES=3
python3 codes/test_demo.py --input_dir path_to_input_folder --output_dir path_to_output_folder
```

### How to test

- Modify `dataroot_LQ` and `pretrain_model_G` (you can also use the pretrained model which is provided in the `./pretrained_model`) in `./codes/options/test/test_HDRUNet.yml`, then run
```
cd codes
python test.py -opt options/test/test_HDRUNet.yml
```
The test results will be saved to `./results/testset_name`.

### How to train

- Prepare the data. Modify `input_folder` and `save_folder` in `./scripts/extract_subimgs_single.py`, then run
```
cd scripts
python extract_subimgs_single.py
```

- Modify `dataroot_LQ` and `dataroot_GT` in `./codes/options/train/train_HDRUNet.yml`, then run
```
cd codes
python train.py -opt options/train/train_HDRUNet.yml
```
The models and training states will be saved to `./experiments/name`.

### Visualization

In `./scripts`, several scripts are available. `data_io.py` and `metrics.py` are provided by the competition organizer for reading/writing data and evaluation. Based on these codes, I provide a script for visualization by using the tone-mapping provided in `metrics.py`. Modify paths of the data in `./scripts/tonemapped_visualization.py` and run
```
cd scripts
python tonemapped_visualization.py
```
to visualize the images.

## Acknowledgment
The code is inspired by [BasicSR](https://github.com/xinntao/BasicSR).
