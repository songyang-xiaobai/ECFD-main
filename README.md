# Efficient Cross-Information Fusion Decoder for Semantic Segmentation

This repo contains the supported code and configuration files to reproduce semantic segmentaion. It is based on [mmsegmentaion](https://github.com/open-mmlab/mmsegmentation/tree/v0.11.0).

This repository contains the official Pytorch implementation of training & evaluation code and the pretrained models for ECFD

## Updates

**9/10/2023** Initial commits

## Results and Models

### Cityscapes

| Backbone   | Method     | Crop Size | Lr Schd | mIoU  | mIoU (ms+flip) | \#params | FLOPs | config                                                                                       | log                                                                                                  | model |
|------------|------------|-----------|---------|-------|----------------|----------|-------|----------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------|-------|
| R50        | ECFD-tiny  | 512x1024  | 80K     | 79.91 | 81.18          | 41M      | 206G  | [config](configs/ecfd_final/ablation/ecfd_tiny_r50_512x1024_80k_cityscapes.py)               | [Log](logs/ablation/ecfd_tiny_r50_512x1024_80k_cityscapes_L_3/20230808_013209.log.json)              |       |
| R50        | ECFD-small | 512x1024  | 80K     | 80.14 | 81.32          | 51M      | 222G  | [config](configs/ecfd_final/mode/ecfd_small_r50_512x1024_80k_cityscapes.py)                  | [Log](logs/mode/ecfd_small_r50_512x1024_80k_cityscapes/20230813_104518.log.json)                     |       |
| R101       | ECFD-tiny  | 512x1024  | 80K     | 80.50 | 81.48          | 60M      | 245G  | [config](configs/ecfd_final/cityscapes/ecfd_tiny_r101_512x1024_80k_cityscapes.py)            | [Log](logs/cityscapes/ecfd_tiny_r101_512x1024_80k_cityscapes)                                        |       |
| R101       | ECFD-small | 512x1024  | 80K     | 80.74 | 82.00          | 70M      | 261G  | [config](configs/ecfd_final/cityscapes/ecfd_small_r101_512x1024_80k_cityscapes.py)           | [Log](logs/cityscapes/ecfd_small_r101_512x1024_80k_cityscapes/20230813_104706.log.json)              |       |
| Swin-Large | ECFD-tiny  | 512x1024  | 80K     | 82.67 | 83.41          | 209M     | 473G  | [config](configs/ecfd_final/cityscapes/ecfd_tiny_swin_large_512x1024_80k_cityscapes.py)      | [Log](logs/cityscapes/ecfd_tiny_swin_large_512x1024_80k_cityscapes/20230813_002424.log.json)         |       |
| Swin-Large | ECFD-small | 512x1024  | 80K     | 83.10 | 83.61          | 218M     | 488G  | [config](configs/ecfd_final/cityscapes/ecfd_small_swin_large_512x1024_80k_cityscapes.py)     | [Log](logs/cityscapes/ecfd_small_swin_large_512x1024_80k_cityscapes/20230829_103558.log.json)        |       |
| ConvNeXt   | ECFD-tiny  | 512x1024  | 80K     | 83.70 | 84.44          | 210M     | 453G  | [config](configs/ecfd_final/cityscapes/ecfd_tiny_convnext_large_512x1024_80k_cityscapes.py)  | [Log](logs/cityscapes/ecfd_tiny_convnext_large_512x1024_80k_cityscapes_ss/20230826_193959.log.json)  |       |
| ConvNeXt   | ECFD-small | 512x1024  | 80K     | 83.55 | 84.42          | 219M     | 468G  | [config](configs/ecfd_final/cityscapes/ecfd_small_convnext_large_512x1024_80k_cityscapes.py) | [Log](logs/cityscapes/ecfd_small_convnext_large_512x1024_80k_cityscapes_ss/20230826_203843.log.json) |[model](https://pan.baidu.com/s/1PV1UuAfMQ-HnZihWGQxXfg)       |

### Pascal Context:

Due to hardware equipment reasons, if there is too much data involved in validation during the training process, the process may be killed. Therefore, in subsequent experiments, we use the first 1000 pieces of Pascal Context dataset as the validation set during the training process (Divide the dataset into 5 parts). When conducting multi-scale testing, divide the dataset into 3 parts and use the - out option in tool/test.py to generate and save the model prediction results. val\_ 0. pkl, val\_ 1. pkl and val\_ 2. pkl, finally through tools/all\_ Evaluate. py statistically calculates the accuracy of the entire validation set.

```
CONFIG=$1
CHECKPOINT=$2
GPUS=$3
OUT=$4
PORT=$5
# Group testing in multi-scale testing
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT tools/test.py $CONFIG $CHECKPOINT --launcher pytorch --eval mIoU  --aug-test  --out=$OUT
# Calculate the prediction accuracy of all data in the dataset
python tools/all_evaluate.py CONFIG --out PKL_PATH --eval mIoU
```

| Backbone   | Method     | Crop Size | Lr Schd | mIoU  | mIoU (ms+flip) | \#params | FLOPs | config                                                                                              | log                                                                                                      | model |
|------------|------------|-----------|---------|-------|----------------|----------|-------|-----------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------|-------|
| R50        | ECFD-tiny  | 480x480   | 80K     | 52.13 | 54.10          | 41M      | 91G   | [config](configs/ecfd_final/pascal_context/ecfd_tiny_r50_480x480_80k_pascal_context.py)             | [Log](logs/pascal-context/ecfd_tiny_r50_480x480_80k_pascal_context/20230813_110953.log.json)             |       |
| R50        | ECFD-small | 480x480   | 80K     | 52.66 | 54.27          | 43M      | 98G   | [config](configs/ecfd_final/pascal_context/ecfd_small_r50_480x480_80k_pascal_context.py)            | [Log](logs/pascal-context/ecfd_small_r50_480x480_80k_pascal_context/20230813_110905.log.json)            |       |
| R101       | ECFD-tiny  | 480x480   | 80K     | 53.80 | 55.75          | 57M      | 108G  | [config](configs/ecfd_final/pascal_context/ecfd_tiny_r101_480x480_80k_pascal_context.py)            | [Log](logs/pascal-context/ecfd_tiny_r101_480x480_80k_pascal_context/20230813_110719.log.json)            |       |
| R101       | ECFD-small | 480x480   | 80K     | 54.3  | 56.09          | 63M      | 115G  | [config](configs/ecfd_final/pascal_context/ecfd_small_r101_480x480_80k_pascal_context.py)           | [Log](logs/pascal-context/ecfd_small_r101_480x480_80k_pascal_context/20230813_110813.log.json)           |       |
| Swin-Large | ECFD-tiny  | 480x480   | 80K     | 63.73 | 64.94          | 205M     | 217G  | [config](configs/ecfd_final/pascal_context/ecfd_tiny_swin_large_480x480_80k_pascal_context.py)      | [Log](logs/pascal-context/ecfd_tiny_swin_large_480x480_80k_pascal_context/20230813_005419.log.json)      |       |
| Swin-Large | ECFD-small | 480x480   | 80K     | 63.36 | 64.81          | 211M     | 223G  | [config](configs/ecfd_final/pascal_context/ecfd_small_swin_large_480x480_80k_pascal_context.py)     | [Log](logs/pascal-context/ecfd_small_swin_large_480x480_80k_pascal_context/20230813_110427.log.json)     |       |
| ConvNeXt   | ECFD-tiny  | 480x480   | 80K     | 62.65 | 63.93          | 207M     | 199G  | [config](configs/ecfd_final/pascal_context/ecfd_tiny_convnext_large_480x480_80k_pascal_context.py)  | [Log](logs/pascal-context/ecfd_tiny_convnext_large_480x480_80k_pascal_context/20230830_144304.log.json)  |       |
| ConvNeXt   | ECFD-small | 480x480   | 80K     | 63.62 | 64.48          | 212M     | 206G  | [config](configs/ecfd_final/pascal_context/ecfd_small_convnext_large_480x480_80k_pascal_context.py) | [Log](logs/pascal-context/ecfd_small_convnext_large_480x480_80k_pascal_context/20230827_090802.log.json) |       |

### 

### BDD100K

| Backbone   | Method     | Crop Size | Lr Schd | mIoU  | mIoU (ms+flip) | \#params | FLOPs | config                                                                                 | log                                                                                            | model |
|------------|------------|-----------|---------|-------|----------------|----------|-------|----------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------|-------|
| R50        | ECFD-tiny  | 480x480   | 80K     | 64.96 | 66.57          | 41M      | 207G  | [config](configs/ecfd_final/bdd100k/ecfd_tiny_r50_512x1024_80k_bdd100k.py)             | [Log](logs/bdd100k/ecfd_tiny_r50_512x1024_80k_bdd100k/20230815_105623.log.json)                |       |
| R50        | ECFD-small | 480x480   | 80K     | 65.48 | 67.32          | 51M      | 222G  | [config](configs/ecfd_final/bdd100k/ecfd_small_r50_512x1024_80k_bdd100k.py)            | [Log](logs/bdd100k/ecfd_small_r50_512x1024_80k_bdd100k/20230815_105655.log.json)               |       |
| R101       | ECFD-tiny  | 480x480   | 80K     | 64.96 | 66.57          | 60M      | 246G  | [config](configs/ecfd_final/bdd100k/ecfd_tiny_r101_512x1024_80k_bdd100k.py)            | [Log](logs/bdd100k/ecfd_tiny_r101_512x1024_80k_bdd100k/20230813_112607.log.json)               |       |
| R101       | ECFD-small | 480x480   | 80K     | 65.48 | 67.32          | 70M      | 261G  | [config](configs/ecfd_final/bdd100k/ecfd_small_r101_512x1024_80k_bdd100k.py)           | [Log](logs/bdd100k/ecfd_small_r101_512x1024_80k_bdd100k/20230814_234651.log.json)              |       |
| Swin-Large | ECFD-tiny  | 480x480   | 80K     | 67.80 | 68.91          | 209M     | 473G  | [config](configs/ecfd_final/bdd100k/ecfd_tiny_swin_large_512x1024_80k_bdd100k.py)      | [Log](logs/bdd100k/ecfd_tiny_swin_large_512x1024_80k_bdd100k/20230813_102742.log.json)         |       |
| Swin-Large | ECFD-small | 480x480   | 80K     | 67.82 | 69.13          | 218M     | 488G  | [config](configs/ecfd_final/bdd100k/ecfd_small_swin_large_512x1024_80k_bdd100k.py)     | [Log](logs/bdd100k/ecfd_small_swin_large_512x1024_80k_bdd100k/20230813_112340.log.json)        |       |
| ConvNeXt   | ECFD-tiny  | 480x480   | 80K     | 67.23 | 67.77          | 210M     | 453G  | [config](configs/ecfd_final/bdd100k/ecfd_tiny_convnext_large_512x1024_80k_bdd100k.py)  | [Log](logs/bdd100k/ecfd_tiny_convnext_large_512x1024_80k_bdd100k_ss/20230827_012735.log.json)  |       |
| ConvNeXt   | ECFD-small | 480x480   | 80K     | 67.62 | 67.95          | 219M     | 468G  | [config](configs/ecfd_final/bdd100k/ecfd_small_convnext_large_512x1024_80k_bdd100K.py) | [Log](logs/bdd100k/ecfd_small_convnext_large_512x1024_80k_bdd100K_ss/20230827_091141.log.json) |       |

Here is a full script for setting up a conda environment to use ECFD

```
conda create -n ecfd python=3.7
conda activate ecfd
conda install pytorch==1.8.0 torchvision==0.9.0 -c pytorch
ln -s ../detection/ops ./
git clone git@github.com:WalBouss/SenFormer.git && cd ECFD-master
pip install mmcv-full==1.3.0
pip install -e .
pip install timm einops
```

**Notes**:

-   **Pre-trained models can be downloaded from** [Swin Transformer for ImageNet Classification](https://github.com/microsoft/Swin-Transformer).
-   Access code for `baidu` is `swin`.

### Inference

```
# single-gpu testing
python tools/test.py <CONFIG_FILE> <SEG_CHECKPOINT_FILE> --eval mIoU

# multi-gpu testing
sbatch tools/dist_test.sh <CONFIG_FILE> <SEG_CHECKPOINT_FILE> <GPU_NUM> <PORT>
```

**Notes:**

-   Introduction to models：
    -   ECFDFPN，ECFD without RCE and CFB, the experiment involved in the manuscript is the FPN in Table 1
    -   ECFDPatch4，ECFD without RCE, the experiment involved in the manuscript is FPN-CFB (1) in Table 1
    -   ECFDGeneralFPN，Strategy 1 in Table 2, R50 encoder based on ordinary convolution and self attention fusion strategy based on context information;
    -   ECFDGeneral，Strategy 2 in Table 2, R50 encoder based on dilation convolution and self attention fusion strategy based on context information;
    -   ECFDSpatial，Strategy 3 in Table 2, R50 encoder based on ordinary convolution (feature pyramid structure), self attention fusion strategy based on spatial information;
    -   ECFD，Strategy 4 in Table 2, R50 encoder based on ordinary convolution (feature pyramid structure), fusion strategy based on cross information fusion, and final decoder structure proposed by us.
