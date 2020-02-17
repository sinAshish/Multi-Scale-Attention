import sys
import pdb
from os.path import isfile, join
import os
import numpy as np
import nibabel as nib
import scipy.io as sio

from scipy import misc
import re

from scipy import ndimage
from scipy import misc

def getImageImageList(imagesFolder):
    if os.path.exists(imagesFolder):
       imageNames = [f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))]

    imageNames.sort()

    return imageNames
    
def normalizeImages(argv):
    
    imagesFolder = argv[0]

    imageNames = getImageImageList(imagesFolder)
    numImages = len(imageNames)
   
    for s_i in range(numImages):
        
        image = ndimage.imread(imagesFolder+imageNames[s_i])
        
        if image.shape[0] > 256:
            centerX = int(image.shape[0]/2)
            centerY = int(image.shape[1]/2)
            newImage = image[centerX-128:centerX+128,centerY-128:centerY+128]
            image = newImage
            
        normalized = (image-np.min(image))/(np.max(image)-np.min(image))
        pdb.set_trace()
        normalized = normalized*255
        
        image = normalized.astype('uint8')
        
        print('Min value: {}  and Max value: {}....Total values: {}'.format(np.min(image),np.max(image),len(np.unique(image))))
        
        gt = np.zeros((image.shape))
        
        if s_i < 10:
            misc.imsave('./Demo_Corstem/val/Img/Img_0' + str(s_i) + '.png',image)
            misc.imsave('./Demo_Corstem/val/GT/Img_0' + str(s_i) + '.png',gt)
        else:
            misc.imsave('./Demo_Corstem/val/Img/Img_' + str(s_i) + '.png',image)
            misc.imsave('./Demo_Corstem/val/GT/Img_0' + str(s_i) + '.png',gt)
        
if __name__ == '__main__':
    normalizeImages(sys.argv[1:])
