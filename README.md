# MCN-main
The implementation of Interpretable Multiplication Convolutional Network (MCN) in Pytorch.
## [An Interpretable Multiplication-Convolution Network for Equipment Intelligent Edge Diagnosis](https://ieeexplore.ieee.org/document/10443049)

# Implementation of the paper:
Paper:
```
@article{Interpretable MCN,
  title={An Interpretable Multiplication-Convolution Network for Equipment Intelligent Edge Diagnosis},
  author = {Rui Liu and Xiaoxi Ding and Qihang Wu and Qingbo He and Yimin Shao},
  journal={IEEE Transactions on Systems, Man and Cybernetics: Systems},
  volume = {},
  pages = {},
  year = {2024},
  issn = {2168-2216},
  doi = {10.1109/TSMC.2023.3346398},
  url = {},
}
```
# Requirements
* Python 3.8.8 or newer
* torch-geometric 2.3.1
* pytorch  1.11.0
* numpy  1.23.0

# Guide 
We presents a multiplication-convolution network in a novel way of collaborating signal processing and deep learning. Its overall structure is shown in Fig. 2, where this novel network architecture takes the spectrum as input samples and consists of a feature multiplication separator, a feature convolution extractor and a classifier.  It should be noted that different from the conventional neural network with matrix-multiplication feature sensing and deep learning with convolution feature sensing, a series of MFKs are cleverly designed with spectrum samples input in the separator. Each MFK operates as a multiplier to extract the sensitive modes with discriminative fault knowledge learned in a signal-filtering way, such as amplitude, center frequency and bandwidth. The obtained multiplications are stacked into a multichannel mode map, which reserves the fault information and is intuitively interpretable. Then, a convolution layer is used as the feature extractor to abstract the feature maps into high-dimension space. Finally, a dense layer is taken as the classifier for fault identification.
![MCN](https://github.com/CQU-BITS/MCN-main/blob/main/GA.png)

# Pakages
* `datasets` contians the data load methods for different dataset
* `models` contians the implemented models for diagnosis tasks
* `postprocessing` contians the implemented functions for result visualization

# Datasets
Self-collected datasets
* Self-made Gear Dataset (unavailable at present)
### Open source datasets
* [SQ Bearing Dataset](https://github.com/Lvhaixin/SQdataset)

# Acknowledgement
* [WDCNN](https://www.mdpi.com/1424-8220/17/2/425)
* [LaplaceAlexNet](https://github.com/HazeDT/WaveletKernelNet)
* [CNN, LeNet, AlexNet and ResNet18](https://github.com/HazeDT/DL-based-Intelligent-Diagnosis-Benchmark)

# Related works
* [R. Liu, X. Ding, et al., “Sinc-Based Multiplication-Convolution Network for Small-Sample Fault Diagnosis and Edge Application,” IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1-12, 2023.](https://ieeexplore.ieee.org/document/10266990)
* [Q. Wu, X. Ding, et al., "An Interpretable Multiplication-Convolution Sparse Network for Equipment Intelligent Diagnosis in Antialiasing and Regularization Constraint," IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1-12, 2023.](https://ieeexplore.ieee.org/document/10108914)
* [Q. Wu, X. Ding, Q. Zhang, R. Liu, S. Wu, Q. He, An Intelligent Edge Diagnosis System Based on Multiplication-Convolution Sparse Network, IEEE Sens. J., (2023) 1-1.](https://ieeexplore.ieee.org/document/10227888)
* [R. Liu, X. Ding, Y. Shao, W. Huang, An interpretable multiplication-convolution residual network for equipment fault diagnosis via time–frequency filtering, Adv. Eng. Inform., 60 (2024) 102421.](https://www.sciencedirect.com/science/article/pii/S1474034624000697)

