# Medical Image Segmentation with Guided Attention
This repository contains the code of our paper: 'Multi-scale guided attention for medical image segmentation'

## Requirements

- The code has been written in Python (3.6) and requires [pyTorch](https://pytorch.org) (version 1.1.0)
- You should also have installed:
-- [Pillow](https://pillow.readthedocs.io/en/stable/index.html)
-- [nibabel](https://nipy.org/nibabel/)
-- [medpy](https://pypi.org/project/MedPy/)
-- [skimage](https://scikit-image.org)
-- [dill](https://pypi.org/project/dill/)

## Preparing your data
You have to split your data into three folders: train/val/test. Each folder will contain two sub-folders: Img and GT, which contain the png files for the images and their corresponding ground truths. The naming of these images is important, as the code to save the results temporarily to compute the 3D DSC, for example, is sensitive to their names.

Specifically, the convention we follow for the names is as follows:
- Subj_Xslice_Y.png where X indicates the subject number (or ID) and Y is the slice number within the whole volume. (Do not use 0 padding for numbers, i.e., the first slice should be 1 and not 01)
- The corresponding mask must be named in the same way as the image.

## Running the code
To run the code you simply need to use the following script:

```
CUDA_VISIBLE_DEVICES=0 python main.py
```

