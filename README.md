# SIRFP: Structural Pruning via Spatial-aware Information Redundancy for Semantic Segmentation

This is a pytorch implementation for paper Structural Pruning via Spatial-aware Information Redundancy for Semantic Segmentation.
This implementation is based on the [DCFP repo](https://github.com/wzx99/DCFP)

## Installation

### 1.Requirements

- Python==3.8.12
- Pytorch==1.10.0
- CUDA==11.3

```bash
conda create -n dcfp python==3.8.12
conda activate dcfp
pip install --upgrade pip
pip install torch==1.10.0+cu113 torchvision==0.11.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html 
pip install tqdm six ordered_set numpy==1.21.2 opencv-python-headless==4.1.2.30 scipy==1.5.4
```

### 2.Datasets
Create a "data" folder. Download datasets(Cityscapes, Pascal context, ADE20k, COCO Stuff). The structure of the data folder is shown below.

  ```bash
  data
  ├── CS
  │   ├── leftImg8bit
  │   │   ├── train
  │   │   ├── test
  │   │   └── val
  │   └── gtFine
  │       ├── train
  │       ├── test
  │       └── val
  ├── ADEChallengeData2016
  │   ├── images
  │   │   ├── training
  │   │   └── validation
  │   └── annotations
  │       ├── training
  │       └── validation
  └── COCO
      ├── images
      └── annotations
  ```

## Training

### 1.Pretraining
 - Create a "pretrained_models" folder. Download pretrained resnet.
```bash
sh scripts/download_pretrianed_models.sh
```
 - Update the path of pretrained models and datasets in "mypath.py"
 - Run the following command to pretrain the model for a few iterations.
```bash
sh scripts/cs/pretrain.sh
```

### 2.Pruning
 - Make sure the pytorch version is 1.10. Other versions may not support our pruning code.
 - Run the following command to get the pruned model.
```bash
sh scripts/cs/prune.sh
```

### 3.Finetuning and evaluation
 - Run the following command to finetune and evaluate the pruned model.
```bash
sh scripts/cs/finetune.sh
```

### 4.TensorRT model (Optional)
 - Install TensorRT.
```bash
pip install pycuda TensorRT==8.5.1.7 packaging
git clone --branch v0.4.0 https://github.com/NVIDIA-AI-IOT/torch2trt 
cd torch2trt
python setup.py install
```
 - Run the following command to get the TensorRT model.
```bash
sh scripts/cs/trt.sh
```
