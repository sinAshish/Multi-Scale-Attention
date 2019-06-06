# Medical Image Segmentation with Guided Attention
This repository contains the main framework to perform the segmentation.

## Important

- The code computes the DSC over training and validation on a 2D-basis. The final reported values should be in 3D
- The loader needs the data to be split in 'train' and 'val' folders. Insided each of these folders, another two sub-folders, i.e., 'Img' and 'GT' will contain ALL the images for training/validation. In addition, the naming of images and GT should be defined to ensure that the names in an ordered list correspond to the same image and GT pair. To do so, I have created the function 'splitDataset', which should be run as follows:

```
python splitDataset.py ./CT 1,2,3,4,5,6,7,8  
```

where the first argument is the main folder containing all the images, and the second is a list of integers, indicating which subjects go to training/validation. Furthermore, imgName and maskName in lines 39 and 40 of the splitDataset function should be change to 'val' or 'train' accordingly.
- If the CT images are employed, all of them have a resolution of 512x512 pixels. Nevertheless, in the case of MRI the slice sizes change. I find the best option is to modify (either resize or crop) the images offline, so that when the medicalDataLoader loads the images they are already with the same size.
