# MCN-main
The implementation of Interpretable Multiplication Convolutional Network (MCN) in Pytorch.
## [An Interpretable Multiplication-Convolution Network for Equipment Intelligent Edge Diagnosis](https://www.sciencedirect.com/science/article)

# Implementation of the paper:
Paper:
```
@article{PHMGNNBenchmark,
  title={An Interpretable Multiplication-Convolution Network for Equipment Intelligent Edge Diagnosis},
  author = {Rui Liu, Xiaoxi Ding, Qihang Wu, Qingbo He and Yimin Shao},
  journal={IEEE Transactions on Systems, Man and Cybernetics: Systems},
  volume = {},
  pages = {},
  year = {},
  issn = {},
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
We presents a multiplication-convolution network in a novel way of collaborating signal processing and deep learning. Its overall structure is shown in Fig. 2, where this novel network architecture takes the spectrum as input samples and consists of a feature multiplication separator, a feature convolution extractor and a classifier.  It should be noted that different from the conventional neural network with matrixmultiplication feature sensing and deep learning with convolution feature sensing, a series of MFKs are cleverly designed with spectrum samples input in the separator. Each MFK operates as a multiplier to extract the sensitive modes with discriminative fault knowledge learned in a signal-filtering way, such as amplitude, center frequency and bandwidth. The obtained multiplications are stacked into a multichannel mode map, which reserves the fault information and is intuitively interpretable. Then, a convolution layer is used as the feature extractor to abstract the feature maps into high-dimension space. Finally, a dense layer is taken as the classifier for fault identification.
![MCN](https://github.com/CQU-BITS/MCN-main/blob/main/GA.png)

# Pakages
* `datasets` contians the data load methods for different dataset
* `models` contians the implemented models for diagnosis tasks
* `postprocessing` contians the implemented functions for result visualization

# Datasets
Self-collected datasets
* Self-made Gear Dataset (unavailable)
### Open source datasets
* [SQ Bearing Dataset](https://github.com/Lvhaixin/SQdataset)

# Related works
* [R. Liu, X. Ding, et al., “Sinc-Based Multiplication-Convolution Network for Small-Sample Fault Diagnosis and Edge Application,” IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1-12, 2023.](https://ieeexplore.ieee.org/document/10266990)
* [Q. Wu, X. Ding, et al., "An Interpretable Multiplication-Convolution Sparse Network for Equipment Intelligent Diagnosis in Antialiasing and Regularization Constraint," IEEE Transactions on Instrumentation and Measurement, vol. 72, pp. 1-12, 2023.](https://ieeexplore.ieee.org/document/10108914)

