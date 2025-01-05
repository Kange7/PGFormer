# PGFormer: A novel Point Cloud Urban Scenes Semantic Segmentation Network Combining Grouped Transformer and KPConv
## Introduction
Official code for the paper PGFormer: A novel Point Cloud Urban Scenes Semantic Segmentation Network Combining Grouped Transformer and KPConv.
As a novel point cloud semantic segmentation method for urban scenes, PGFormer is capable of enhancing local key information without losing global context. 

![Pipeline](https://github.com/Kange7/PGFormer/blob/main/doc/Pipeline.jpg)

![Grouped Transformer](https://github.com/Kange7/PGFormer/blob/main/doc/GroupedTransformer.jpg)

![GRT block](https://github.com/Kange7/PGFormer/blob/main/doc/GRTblock.jpg)

![Pisitional Encoding](https://github.com/Kange7/PGFormer/blob/blob/main/doc/PE.jpg)

## Dependency

1. Create conda environment
```
conda create -n PGFormer python=3.8
conda activate PGFormer
```

2. Install the package using 
```
pip install -r requirement.txt
```

3. Please follow the <a href="https://github.com/HuguesTHOMAS/KPConv-PyTorch">KPConv</a> to compile the C++ extension modules in `cpp_wrappers`

4. Please follow the <a href="https://github.com/Strawberry-Eat-Mango/PCT_Pytorch">PCT</a> to compile the C++ extension modules in `pointnet2_ops_lib`

## Train & test

### Train
```
python train_for_H3D.py

python train_for_Pairs3D.py
```
### Test

```
python test_model -s "result/log_name" -m test
```
### Evaluation
```
python cal_metrics.py
```




## Acknowledgment

Our code uses the <a href="https://github.com/HuguesTHOMAS/KPConv-PyTorch">KPConv</a> as the backbone network.
We would like to thank <a href="https://github.com/PuzoW/One-Class-One-Click">OCOC</a> for their contribution of the related code.

## License
Our code is released under MIT License (see LICENSE file for details).
