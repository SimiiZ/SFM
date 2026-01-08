# Structural Feature Modulation for Day-Night Cross-Domain Object Re-Identification
[![GitHub](https://img.shields.io/badge/license-MIT-green)](https://github.com/anosorae/IRRA/blob/main/LICENSE) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/cross-modal-implicit-relation-reasoning-and/nlp-based-person-retrival-on-cuhk-pedes)](https://paperswithcode.com/sota/nlp-based-person-retrival-on-cuhk-pedes?p=cross-modal-implicit-relation-reasoning-and)

Official PyTorch implementation of the paper Structural Feature Modulation for Day-Night Cross-Domain Object Re-Identification.

## Updates
- (12/3/2025) Code released!


## Highlights
Day-night cross-domain object re-identification presents significant challenges due to severe illumination-induced domain gaps. Unlike conventional attention mechanisms that suffer from limited dimensional coverage and rely on single-type normalization strategies, we propose a structural feature modulation (SFM) approach that operates from a modulation perspective. Our SFM approach incorporates a gated batch-layer normalization strategy within the modulation architecture, resulting in the construction of a gated normalization-based modulation (GNM) module. This module effectively leverages the complementary advantages of both batch normalization and layer normalization to balance intra-domain discriminability and cross-domain generalization. Furthermore, we develop a multi-granularity modulation (MGM) module that recalibrates features across both edge-granularity and area-granularity pathways, enabling comprehensive structural modulation. Extensive experiments on DN-348 and LLCM benchmark datasets demonstrate that our SFM approach consistently outperforms state-of-the-art approaches.![Overview](images/overview.png)


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
The LLCM dataset can be downloaded from [here](https://github.com/ZYK100/LLCM/tree/main/LLCM%20Dataset%20Agreement)

The DN348 dataset can be downloaded from [here](https://github.com/chenjingong/DN-ReID/tree/main/data_path)


### Training
   Train a model by
  ```bash
python llcm_train_simi.py --dataset llcm  --gpu 1
```


 ### Test

 Test a model on dn348 or dnwild dataset by 
  ```bash
python llcm_test_simi.py --mode 'v2t' --resume 'model_path' --gpu 1 --dataset llcm
```


## Contact
If you have any question, please feel free to contact us. E-mail: [SimiTuT@hqu.edu.cn.](mailto:SimiTuT@hqu.edu.cn.), [jqzhu@hqu.edu.cn.](mailto:jqzhu@hqu.edu.cn.)

