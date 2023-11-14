# ASIC: Aligning Sparse Image Collections (ICCV 2023 Oral)

### [Project Page](https://kampta.github.io/asic/) | [Video](https://www.youtube.com/watch?v=fLjkkMriuoY) | [Paper](https://arxiv.org/abs/2303.16201)

Re-implementation of ASIC - Please note that this code base has been re-implemented from scratch, cleaned up and includes bugfixes. The default hyperparameters/metrics in the eval may differ from those in the paper. An updated version on the arXiv will be uploaded shortly.

[Kamal Gupta](https://kampta.github.io/)<sup>1</sup>, [Varun Jampani](https://varunjampani.github.io/)<sup>2</sup>, [Carlos Esteves](https://machc.github.io/)<sup>2</sup>, [Abhinav Shrivastava](https://www.cs.umd.edu/~abhinav)<sup>1</sup>, [Ameesh Makadia](https://www.ameeshmakadia.com/)<sup>2</sup>, [Noah Snavely](https://www.cs.cornell.edu/~snavely/)<sup>2</sup>, [Abhishek Kar](https://abhishekkar.info)<sup>2</sup><br>
<sup>1</sup>University of Maryland, College Park, <sup>2</sup>Google

![ASIC](https://kampta.github.io/asic/static/images/cow.jpg)


## Set up

This project uses Anaconda (or Miniconda) for dependency management. All the requirements can be install with the following commands.

```
conda env create -f environment.yaml
conda activate asic
pip install --force-reinstall cython==0.29.36
pip install --no-build-isolation git+https://github.com/lucasb-eyer/pydensecrf.git
conda activate asic
```

## Prepare datasets

We use [CUB](https://www.vision.caltech.edu/datasets/cub_200_2011/) and [SPair-71k](https://cvlab.postech.ac.kr/research/SPair-71k/) datasets for most of our experiments. 

**CUB** 
This script downloads and preprocesses the CUB dataset (computes the DINO features, nearest neighbors, etc.)

```
python prepare_data.py \
    --dset cub \
    --cub_idx 1 \
    --img_dir raw_data \
    --out_dir processed_data
```

**SPair-71k** 
This script downloads and preprocesses the SPair-71k dataset

```
python prepare_data.py \
    --dset spair \
    --spair_cat cat \
    --img_dir raw_data \
    --out_dir processed_data
```

**Custom Dataset**
You can prepare any *folder* of images in-the-wild using

```
python prepare_data.py \
    --dset folder
    --img_dir <path/to/image/folder> \
    --out_dir processed_data
```

## Train model

We recommend using logging with [wandb](https://wandb.ai) for tracking experiments.

**CUB**
```
python train.py \
    --exp-name cub_idx_1_test \
    --flow_dir processed_data \
    --dset cub \
    --cub_idx 1 \
    --img_dir raw_data
```

**SPair**
```

python train.py \
    --exp-name spair_cat_test \
    --flow_dir processed_data \
    --dset spair \
    --spair_cat aeroplane \
    --img_dir raw_data
```

## Evaluate

Evaluate nearest neighbors [dino-vit-features](https://dino-vit-features.github.io)
```
python evaluate.py \
    --dset spair \
    --spair_cat aeroplane \
    --img_dir raw_data \
    --flow_dir processed_data \
    --stride 2
```

Evaluate ASIC
```
python evaluate.py \
    --dset spair \
    --spair_cat aeroplane \
    --img_dir raw_data \
    --flow_dir processed_data \
    --stride 2 --k 2 3 \
    --ckpt logs/spair_aeroplane/checkpoint.pt
```

Metrics (reproduced using this codebase, numbers here may differ a bit from the ones in the paper because of randomness in the runs). Mean and standard deviation of 3 runs also provided

| Category    | PCK@0.10    | 2-cycle PCK@0.10 | 3-cycle PCK@0.10 |
| ----------- | ----------- | ---------------- | ---------------- |
| aeroplane   |  57.9 (0.48)|     68.4         |       62.1       |
| bicycle     |  21.6 (0.31)|     53.9         |       40.0       |
| bird        |  76.6 (6.23)|     80.7         |       78.0       |
| boat        |  20.5 (0.35)|     49.4         |       35.7       |
| bottle      |  37.2 (0.71)|     57.4         |       47.0       |
| bus         |  28.1 (0.92)|     52.5         |       38.5       |
| car         |  25.3 (0.21)|     50.3         |       36.5       |
| cat         |  53.1 (0.34)|     65.6         |       61.5       |
| chair       |  20.6 (1.11)|     47.6         |       33.9       |
| cow         |  44.2 (1.08)|     57.2         |       50.4       |
| dog         |  46.3 (1.11)|     62.5         |       53.6       |
| horse       |  36.9 (1.08)|     55.8         |       47.9       |
| motorbike   |  26.5 (0.66)|     42.3         |       35.7       |
| person      |  44.9 (3.07)|     60.8         |       52.7       |
| pottedplant |  16.5 (0.49)|     36.4         |       26.9       |
| sheep       |  24.6 (0.64)|     46.5         |       36.8       |
| train       |  49.1 (0.84)|     55.2         |       51.2       |
| tvmonitor   |  27.2 (0.43)|     44.8         |       33.9       |
| Average     |  36.5 (1.06)|     54.8         |       45.7       |

## Cite
```
@inproceedings{gupta2023asic,
  title         = {ASIC: Aligning Sparse Image Collections},
  author        = {Gupta, Kamal and Jampani, Varun and Esteves, Carlos and Shrivastava, Abhinav and Makadia, Abhinav and Snavely, Noah and Kar, Abhishek},
  booktitle     = {ICCV},
  year          = {2023},
}
```