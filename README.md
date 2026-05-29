# Structural Feature Modulation for Day-Night Cross-Domain Object Re-Identification
[![GitHub](https://img.shields.io/badge/license-MIT-green)](https://github.com/anosorae/IRRA/blob/main/LICENSE) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cross-modal-implicit-relation-reasoning-and/nlp-based-person-retrival-on-cuhk-pedes)](https://paperswithcode.com/sota/nlp-based-person-retrival-on-cuhk-pedes?p=cross-modal-implicit-relation-reasoning-and)

Official PyTorch implementation of the paper Structural Feature Modulation for Day-Night Cross-Domain Object Re-Identification.

## Updates
- (12/3/2025) Code released!


## Highlights
Day-night cross-domain object re-identification presents significant challenges due to severe illumination-induced domain gaps. Unlike conventional attention mechanisms that suffer from limited dimensional coverage and rely on single-type normalization strategies, we propose a structural feature modulation (SFM) approach that operates from a modulation perspective. Our SFM approach incorporates a gated batch-layer normalization strategy within the modulation architecture, resulting in the construction of a gated normalization-based modulation (GNM) module. This module effectively leverages the complementary advantages of both batch normalization and layer normalization to balance intra-domain discriminability and cross-domain generalization. Furthermore, we develop a multi-granularity modulation (MGM) module that recalibrates features across both edge-granularity and area-granularity pathways, enabling comprehensive structural modulation. Extensive experiments on DN348 and LLCM benchmark datasets demonstrate that our SFM approach consistently outperforms state-of-the-art approaches.

![Overview](images/overview.png)

## Results

### Pretrained Models

**DN348:**

| Mode | mAP(%) | Rank1(%) | Download |
|------|--------|----------|----------|
| Night-to-Day | **50.31** | **83.43** | [百度网盘](https://pan.baidu.com/s/1uKl10HHGyxdtvaWRvJS1wA) (提取码: **y4x4**) · [Google Drive](https://drive.google.com/drive/folders/1_cJJyMv8y85UzMvv118Kc60JiJGPRB6i?usp=drive_link) |
| Day-to-Night | **49.12** | **70.94** | [百度网盘](https://pan.baidu.com/s/1uKl10HHGyxdtvaWRvJS1wA) (提取码: **y4x4**) · [Google Drive](https://drive.google.com/drive/folders/1_cJJyMv8y85UzMvv118Kc60JiJGPRB6i?usp=drive_link) |

**LLCM:**

| Mode | mAP(%) | Rank1(%) | Download |
|------|--------|----------|----------|
| Infrared-to-Visible | **64.78** | **57.91** | [百度网盘](https://pan.baidu.com/s/1fNavwh54QGWc1fZ_vlI4IQ) (提取码: **bn96**) · [Google Drive](https://drive.google.com/drive/folders/1DPIrNwFKBeijB-s_Sa_AD_k5Pn4fVEd5?usp=drive_link) |
| Visible-to-Infrared | **68.55** | **66.01** | [百度网盘](https://pan.baidu.com/s/1fNavwh54QGWc1fZ_vlI4IQ) (提取码: **bn96**) · [Google Drive](https://drive.google.com/drive/folders/1DPIrNwFKBeijB-s_Sa_AD_k5Pn4fVEd5?usp=drive_link) |



## Usage
### Requirements
we use single RTX3090 for training and evaluation. 
```
Python 3.10
PyTorch 2.5.1
Torchvision 0.20.1
CUDA 12.1
prettytable
easydict
```

### Prepare Datasets
Download the datasets and organize them as follows:
```
|-- your dataset root dir/
|   |-- <DN348>/
|       |-- day
|            |-- 00634
|            |-- 00635
|            |-- ...
|       |-- night
|            |-- 00634
|            |-- 00635
|            |-- ...
|       |-- train_test_split
|            |-- test_list_day.txt
|            |-- test_list_night.txt
|            |-- train_list_day.txt
|            |-- train_list_night.txt
|
|   |-- <LLCM>/
|       |-- idx
|            |-- test_id.txt
|            |-- test_nir.txt
|            |-- test_vis.txt
|            |-- train_nir.txt
|            |-- train_vis.txt
|       |-- nir
|            |-- 0000
|            |-- 0001
|            |-- ...
|       |-- vis
|            |-- 0000
|            |-- 0001
|            |-- ...
|       |-- test_nir
|            |-- cam1
|            |-- cam2
|            |-- ... (cam1-cam9)
|       |-- test_vis
|            |-- cam1
|            |-- cam2
|            |-- ... (cam1-cam9)
```
The DN348 dataset can be downloaded from [here](https://github.com/chenjingong/DN-ReID/tree/main/data_path).
The LLCM dataset can be downloaded from [here](https://github.com/ZYK100/LLCM/tree/main/LLCM%20Dataset%20Agreement).


### Training
   Train a model by
  ```bash
# Train on DN348 dataset
python dn348_train_simi.py --dataset dn348 --gpu 0
# Train on LLCM dataset
python llcm_train_simi.py --dataset llcm --gpu 0
```

 ### Test

 Test a model on DN348 or LLCM dataset by 
  ```bash
# Test on DN348 dataset  (Day-to-Night)
python dn348_test_simi.py --mode 'v2t' --resume 'model_path' --gpu 0 --dataset dn348
# Test on DN348 dataset  (Night-to-Day)
python dn348_test_simi.py --mode 't2v' --resume 'model_path' --gpu 0 --dataset dn348

# Test on LLCM dataset (Visible-to-Infrared)
python llcm_test_simi.py --mode 'v2t' --resume 'model_path' --gpu 0 --dataset llcm
# Test on LLCM dataset (Infrared-to-Visible)
python llcm_test_simi.py --mode 't2v' --resume 'model_path' --gpu 0 --dataset llcm
```
## Acknowledgements
This work is built upon several excellent open-source projects. We would like to thank the authors for their contributions.

**DN348:**
```bibtex
@inproceedings{dn348,
  title={Day-Night Cross-domain Vehicle Re-identification},
  author={Li, Hongchao and Chen, Jingong and Zheng, Aihua and Wu, Yong and Luo, Yonglong},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={12626--12635},
  year={2024}
}
```
**DEEN:**
```bibtex
@inproceedings{deen,
  title={Diverse Embedding Expansion Network and Low-Light Cross-Modality Benchmark for Visible-Infrared Person Re-Identification},
  author={Zhang, Yukang and Wang, Hanzi},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  pages={2153--2162},
  year={2023}
}
```

## Contact
If you have any question, please feel free to contact us. E-mail: [SimiTuT@hqu.edu.cn.](mailto:SimiTuT@hqu.edu.cn.), [jqzhu@hqu.edu.cn.](mailto:jqzhu@hqu.edu.cn.)

