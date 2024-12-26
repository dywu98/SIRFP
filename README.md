<h2 align="center">SIRFP: Structural Pruning via Spatial-aware Information Redundancy for Semantic Segmentation</h2>
<p align="center"><b>AAAI-2025</b> | <a href="https://arxiv.org/abs/2412.12672">[Paper]</a> | <a href="https://github.com/dywu98/SIRFP">[Code]</a> </p>


SIRFP is a tool for network pruning on semantic segmentation models. It can achieves almost lossless pruning under 60% pruning ratio.

![image](https://github.com/dywu98/SIRFP/blob/main/figs/figure1.png)


## Installation

### 1.Requirements

- Python==3.8.12
- Pytorch==1.10.0
- CUDA==11.3

```bash
conda create -n sirfp python==3.8.12
conda activate sirfp
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

## How to run

### 1.Pretrained model
 - Create a "pretrained_models" folder. Download pretrained resnet.
```bash
sh scripts/download_pretrianed_models.sh
```
 - Update the path of pretrained models and datasets in "mypath.py"

### 2.Training, Pruning, and Finetuning
 - Make sure the pytorch version is 1.10. Other versions may not support the pruning code.
 - Run the following command to prune model using SIRFP.
```bash
sh scripts/cs/prune.sh
```

### 3.TensorRT model (Optional)
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

## Acknowledgement
This implementation is based on the [DCFP repo](https://github.com/wzx99/DCFP)

## Citation
If you find this repository helpful, please consider citing SIRFP:
```Shell
@inproceedings{wu2024structural,
      title={Structural Pruning via Spatial-aware Information Redundancy for Semantic Segmentation}, 
      author={Dongyue Wu and Zilin Guo and Li Yu and Nong Sang and Changxin Gao},
      year={2024},
      booktitle={The 39th Annual AAAI Conference on Artificial Intelligence}
      url={https://arxiv.org/abs/2412.12672}, 
}
```
