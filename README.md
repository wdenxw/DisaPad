# DisPad: Flexible On-Body Displacement of Fabric Sensors for Robust Joint-Motion Tracking

This repository contains code for the IMWUT paper"DisPad: Flexible On-Body Displacement of Fabric Sensors for Robust
Joint-Motion Tracking". It allows you to deal with the sensor displacement and the transferring to new users and new
motions.

## Prerequisites

The code was tested with pytorch 1.11.0, but it should be able to run in newer versions as well. The detail environment
is as follows (assuming that correct CUDA and cuDNN versions are installed):

* pytorch.gpu =1.11.0
* pandas =1.3.5
* numpy =1.21.5
* python=3.7.13
* scikit-learn=1.0.2

Next, make sure to update the path to where you stored the data (*.npy files) under
constants.file_dir in constants.py. 

## Dataset Preparation

It requires that you download the data from the project page (We will update the link later).

## Example Usage

1. Modify the user name and motion name you want to transfer in settings.py
2. launch main.py

```
python main.py
```

## Contact Information

For questions or problems please file an issue or contact 452292660@qq.com or guoshihui@xmu.edu.cn.

##Citation
If you use this code or data for your own work, please use the following citation:

```
@article{10.1145/3580832,
author = {Chen, Xiaowei and Jiang, Xiao and Fang, Jiawei and Guo, Shihui and Lin, Juncong and Liao, Minghong and Luo, Guoliang and Fu, Hongbo},
title = {DisPad: Flexible On-Body Displacement of Fabric Sensors for Robust Joint-Motion Tracking},
year = {2023},
issue_date = {March 2023},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {7},
number = {1},
url = {https://doi.org/10.1145/3580832},
doi = {10.1145/3580832},
abstract = {The last few decades have witnessed an emerging trend of wearable soft sensors; however, there are important signal-processing challenges for soft sensors that still limit their practical deployment. They are error-prone when displaced, resulting in significant deviations from their ideal sensor output. In this work, we propose a novel prototype that integrates an elbow pad with a sparse network of soft sensors. Our prototype is fully bio-compatible, stretchable, and wearable. We develop a learning-based method to predict the elbow orientation angle and achieve an average tracking error of 9.82 degrees for single-user multi-motion experiments. With transfer learning, our method achieves the average tracking errors of 10.98 degrees and 11.81 degrees across different motion types and users, respectively. Our core contributions lie in a solution that realizes robust and stable human joint motion tracking across different device displacements.},
journal = {Proc. ACM Interact. Mob. Wearable Ubiquitous Technol.},
month = {mar},
articleno = {5},
numpages = {27},
keywords = {soft sensors, domain adaption, textile sensors, transfer learning, fuzzy entropy, long short-term memory, motion tracking, robust signal processing}
}
```
