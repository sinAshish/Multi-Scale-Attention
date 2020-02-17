import os
import numpy as np
import scipy.io as sio
import pdb
import time
from os.path import isfile, join

import nibabel as nib
from PIL import Image
from medpy.metric.binary import dc,hd
import skimage.transform as skiTransf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision

from .progressBar import printProgressBar

def load_nii(imageFileName, printFileNames):
    if printFileNames == True:
        print (" ... Loading file: {}".format(imageFileName))

    img_proxy = nib.load(imageFileName)
    imageData = img_proxy.get_data()

    return (imageData, img_proxy)

def evaluate3D(modelName):

    # Path where the ground truth (as nifti file) should be for comparisons
    path_GT = './DataSet_Challenge/GT_Nifti/Val_1'

    # Path where the predictions are saved
    path_Pred = 'Results/Images/' + modelName + '/Nifti'

    if not os.path.exists('Results/Images/' + modelName + '/Nifti/'):
        os.makedirs('Results/Images/' + modelName + '/Nifti/',exist_ok=True)
    GT_names = getImageImageList(path_GT)
    Pred_names = getImageImageList(path_Pred)

    GT_names.sort()
    Pred_names.sort()

    numClasses = 4
    DSC = np.zeros((len(Pred_names), numClasses))

    for s_i in range(len(Pred_names)):
        path_Subj_GT = path_GT +'/'+GT_names[s_i]
        path_Subj_pred = path_Pred +'/'+Pred_names[s_i]

        [imageDataGT, img_proxy] = load_nii(path_Subj_GT, printFileNames=False)
        [imageDataCNN, img_proxy] = load_nii(path_Subj_pred, printFileNames=False)

        for c_i in range(numClasses):
            label_GT = np.zeros(imageDataGT.shape, dtype=np.int8)
            label_CNN = np.zeros(imageDataCNN.shape, dtype=np.int8)
            idx_GT = np.where(imageDataGT == c_i+1)
            label_GT[idx_GT] = 1
            idx_CNN = np.where(imageDataCNN == c_i+1)
            label_CNN[idx_CNN] = 1

            DSC[s_i,c_i] = dc(label_GT,label_CNN)

    return DSC


def getImageImageList(imagesFolder):
    if os.path.exists(imagesFolder):
       imageNames = [f for f in os.listdir(imagesFolder) if isfile(join(imagesFolder, f))]

    imageNames.sort()

    return imageNames

def to_var(x):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x)


def reconstruct3D(modelName,epoch, isBest=False):

    path = 'Results/Images/' + modelName + '/' + str(epoch)
    subjNames = os.listdir(path)


    for s_i in range(len(subjNames)):
         path_Subj = path +'/'+subjNames[s_i]
         imgNames = getImageImageList(path_Subj)

         numImages = len(imgNames)

         xSize = 256
         ySize = 256
         vol_numpy = np.zeros((xSize, ySize, numImages))

         for t_i in range(numImages-1):

             imagePIL = Image.open(path_Subj + '/'+str(t_i+1)+'.png').convert('LA')
             imageNP = np.array(imagePIL)
             vol_numpy[:, :, t_i] = imageNP[:,:,0]/63 # To have labels in the range [0,1,2]

         xform = np.eye(4) * 2
         imgNifti = nib.nifti1.Nifti1Image(vol_numpy, xform)
         if not os.path.exists('Results/Images/' + modelName + '/Nifti/'):
             os.makedirs('Results/Images/' + modelName + '/Nifti/',exist_ok=True)
         niftiName = 'Results/Images/' + modelName + '/Nifti/' + subjNames[s_i]
         nib.save(imgNifti, niftiName)

         if (isBest):
             if not os.path.exists('Results/Images/' + modelName + '/Nifti_Best/'):
                 os.makedirs('Results/Images/' + modelName + '/Nifti_Best/', exist_ok=True)
             niftiName = 'Results/Images/' + modelName + '/Nifti_Best/' + subjNames[s_i]
             nib.save(imgNifti, niftiName)


        
class computeDiceOneHot(nn.Module):
    def __init__(self):
        super(computeDiceOneHot, self).__init__()

    def dice(self, input, target):
        inter = (input * target).float().sum()
        sum = input.sum() + target.sum()
        if (sum == 0).all():
            return (2 * inter + 1e-8) / (sum + 1e-8)

        return 2 * (input * target).float().sum() / (input.sum() + target.sum())

    def inter(self, input, target):
        return (input * target).float().sum()

    def sum(self, input, target):
        return input.sum() + target.sum()

    def forward(self, pred, GT):
        # GT is 4x320x320 of 0 and 1
        # pred is converted to 0 and 1
        batchsize = GT.size(0)
        DiceN = to_var(torch.zeros(batchsize, 2))
        DiceB = to_var(torch.zeros(batchsize, 2))
        DiceW = to_var(torch.zeros(batchsize, 2))
        DiceT = to_var(torch.zeros(batchsize, 2))
        DiceZ = to_var(torch.zeros(batchsize, 2))

        for i in range(batchsize):
            DiceN[i, 0] = self.inter(pred[i, 0], GT[i, 0])
            DiceB[i, 0] = self.inter(pred[i, 1], GT[i, 1])
            DiceW[i, 0] = self.inter(pred[i, 2], GT[i, 2])
            DiceT[i, 0] = self.inter(pred[i, 3], GT[i, 3])
            DiceZ[i, 0] = self.inter(pred[i, 4], GT[i, 4])

            DiceN[i, 1] = self.sum(pred[i, 0], GT[i, 0])
            DiceB[i, 1] = self.sum(pred[i, 1], GT[i, 1])
            DiceW[i, 1] = self.sum(pred[i, 2], GT[i, 2])
            DiceT[i, 1] = self.sum(pred[i, 3], GT[i, 3])
            DiceZ[i, 1] = self.sum(pred[i, 4], GT[i, 4])

        return DiceN, DiceB , DiceW, DiceT, DiceZ


def DicesToDice(Dices):
    sums = Dices.sum(dim=0)
    return (2 * sums[0] + 1e-8) / (sums[1] + 1e-8)

    
def getSingleImage(pred):
    # input is a 4-channels image corresponding to the predictions of the net
    # output is a gray level image (1 channel) of the segmentation with "discrete" values
    num_classes = 5
    Val = to_var(torch.zeros(num_classes))

    # Chaos MRI
    Val[1] = 0.24705882
    Val[2] = 0.49411765
    Val[3] = 0.7411765
    Val[4] = 0.9882353
    
    x = predToSegmentation(pred)
   
    out = x * Val.view(1, 5, 1, 1)

    return out.sum(dim=1, keepdim=True)


def predToSegmentation(pred):
    Max = pred.max(dim=1, keepdim=True)[0]
    x = pred / Max
    return (x == 1).float()


def getOneHotSegmentation(batch):
    backgroundVal = 0

    # Chaos MRI (These values are to set label values as 0,1,2,3 and 4)
    label1 = 0.24705882
    label2 = 0.49411765
    label3 = 0.7411765
    label4 = 0.9882353
    
    oneHotLabels = torch.cat((batch == backgroundVal, batch == label1, batch == label2, batch == label3, batch == label4),
                             dim=1)
    
    return oneHotLabels.float()


def getTargetSegmentation(batch):
    # input is 1-channel of values between 0 and 1
    # values are as follows : 0, 0.3137255, 0.627451 and 0.94117647
    # output is 1 channel of discrete values : 0, 1, 2 and 3
    
    denom = 0.24705882 # for Chaos MRI  Dataset this value

    return (batch / denom).round().long().squeeze()


def saveImages_for3D(net, img_batch, batch_size, epoch, modelName, deepSupervision=False, isBest= False):
    # print(" Saving images.....")
    path = 'Results/Images/' + modelName + '/' + str(epoch)
    if not os.path.exists(path):
        os.makedirs(path)


    total = len(img_batch)
    net.eval()
    softMax = nn.Softmax()
    for i, data in enumerate(img_batch):
        image, labels, img_names = data

        MRI = to_var(image)
        Segmentation = to_var(labels)

        segmentation_prediction = net(MRI)

        pred_y = softMax(segmentation_prediction)
        
        segmentation = getSingleImage(pred_y)

        out = torch.cat((MRI, segmentation, Segmentation))

        str_1 = img_names[0].split('/Img/')
        str_subj = str_1[1].split('slice')

        path_Subj = path + '/' + str_subj[0]
        if not os.path.exists(path_Subj):
            os.makedirs(path_Subj)

        str_subj = str_subj[1].split('_')
        torchvision.utils.save_image(segmentation.data, os.path.join(path_Subj, str_subj[1]))
    printProgressBar(total, total, done="Images saved !")


def inference(net, img_batch):
    total = len(img_batch)

    Dice1 = torch.zeros(total, 2)
    Dice2 = torch.zeros(total, 2)
    Dice3 = torch.zeros(total, 2)
    Dice4 = torch.zeros(total, 2)
    
    net.eval()
    img_names_ALL = []

    dice = computeDiceOneHot().cuda()
    softMax = nn.Softmax().cuda()
    for i, data in enumerate(img_batch):
        
        printProgressBar(i, total, prefix="[Inference] Getting segmentations...", length=30)
        image, labels, img_names = data
        img_names_ALL.append(img_names[0].split('/')[-1].split('.')[0])

        MRI = to_var(image)
        Segmentation = to_var(labels)

        segmentation_prediction = net(MRI)

        pred_y = softMax(segmentation_prediction)
        Segmentation_planes = getOneHotSegmentation(Segmentation)

        segmentation_prediction_ones = predToSegmentation(pred_y)
        DicesN, Dices1, Dices2, Dices3, Dices4 = dice(segmentation_prediction_ones, Segmentation_planes)

        Dice1[i] = Dices1.data
        Dice2[i] = Dices2.data
        Dice3[i] = Dices3.data
        Dice4[i] = Dices4.data

    printProgressBar(total, total, done="[Inference] Segmentation Done !")

    ValDice1 = DicesToDice(Dice1)
    ValDice2 = DicesToDice(Dice2)
    ValDice3 = DicesToDice(Dice3)
    ValDice4 = DicesToDice(Dice4)
   
    return [ValDice1,ValDice2,ValDice3,ValDice4]



class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).float()

