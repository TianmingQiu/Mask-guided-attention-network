# Mask-guided attention network
This work reproduces the paper [Mask-Guided Attention Network for Occluded Pedestrian Detection](https://arxiv.org/abs/1910.06160). Original authors have their own [implementations](https://github.com/Leotju/MGAN), but they remove training parts. We use their codes and **rewrite the training parts** on our own. Same as the original implementation, we also work on the open-source toolbox [mmdetetcion](https://github.com/open-mmlab/mmdetection). 

## Initialize MMdetection framework
- We are using a detetction toolbox based on PyTorch: [mmdetetcion](https://github.com/open-mmlab/mmdetection). All the information is available on GitHub.
- Follow their [documents](https://mmdetection.readthedocs.io/en/latest/) to install the toolbox completely. 


## Add self-defined detector
After successful installation, we only need to add few codes to set up our self-defined detector. This repository only include added codes. You can just add them into mmdetection framework.
- `mmdet/model/detector/mgan.py`: for MGAN forward training and loss
- `model/detector/__init__.py`: add `from .mgan import MGAN`, and add `'MGAN'` in `__all__` list
- `mmdet/model/mgan_head/`: MGAN brach
- `mmdet/utilis/`: tools for network calculations
- `mmdet/models/backbones/vgg.py `: VGG backbone
- `configs/mgan.py`: network structure claim, hyperparamter, and directories

Then you can train MGAN network by:
```sh
python tools/train.py configs/mgan.py
```
You can test your codes by:
```sh
python tools/test_mgan.py configs/mgan.py models/50_65.pth --eval bbox --out result/50_65_bdd.pkl
```
where the checkpoint can be downloaded from original authors' [GitHub](https://github.com/Leotju/MGAN).

You can also use our [docker](https://hub.docker.com/repository/docker/justinchiu1024/mmdetect) to reproduce it.


## Convert your own dataset into COCO style
The dataset should have the same annotation convention as COCO dataset. For example, for BDD dataset can be converted in this [way](https://github.com/ucbdrive/bdd100k/blob/master/bdd100k/bdd2coco.py).
## Visualize attention as heatmap
ToDo
