#Project Instructions: 

The following Repository Layout contains all files needed to reproduce the work within this project: 

  *Models: All models created from training on CIFAR10 and ImageNet

  *ProjectCode: Any command line files used to train, test, and analyze results within the project

  *StateDirectory: Any state directory path files that can be used to pick off training or reimplement models based on parameters obtained within this project

#GhostNet: 

    *The original publication of GhostNet can be found at: 
    GhostNet: More Features from Cheap Operations. CVPR 2020. [arXiv] [Most Influential CVPR 2020 Papers] 
    By Kai Han, Yunhe Wang, Qi Tian, Jianyuan Guo, Chunjing Xu, Chang Xu.

#Implementation
This folder provides the PyTorch code and pretrained models of the extended version of GhostNet on CIFAR10 & ImageNet.

Requirements:
The code was verified on Python 3.11.12 , Torch Version: 2.6.0+cu124

#Usage
*In the project code folder, rum __ & __ to get the models and accuracy of the original publication model and the extended model with CBAM on ImageNet & CIFAR10.

Data Preparation:
ImageNet data dir should have the following structure, and val and caffe_ilsvrc12 subdirs are essential:

dir/
  train/
    ...
  val/
    n01440764/
      ILSVRC2012_val_00000293.JPEG
      ...
    ...
