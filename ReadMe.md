# Mask-guided attention
## Initialize MMdetection framework
- We are using a detetction toolbox based on PyTorch : [mmdetetcion](https://github.com/open-mmlab/mmdetection). All the information is available on GitHub.
- Follow their [documents](https://mmdetection.readthedocs.io/en/latest/) to install the toolbox completely. 


## Add self-defined detector
After successful installation, we only need to add few codes to set up our self-defined detector.
1. mmdet/model/detector/mgan.py for training and loss
2. mmdet/model/mgan_head/mgan_head.py and __init__.py for mgan branch
3. configs/mgan.py for network structure definition
4. utilis/*
5. backbones/vgg.py 

add class in the init:
1. model/__init__.py: from .mgan_head import *
2. model/detector/__init__.py from .mgan import MGAN, and add 'MGAN' ib the all list
3. backbones/init.py from .vgg import VGG and 'VGG'

python tools/train.py configs/mgan.py 


## Convert your own dataset into COCO style
