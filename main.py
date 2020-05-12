# Atrial Multi-label Segmentation Network
#
# Copyright (c) Cardiac Electromechanics Research Group.
# All rights reserved. CemrgNet is available as free open-source software under a 3-clause BSD license.
# This software is distributed WITHOUT ANY WARRANTY or SUPPORT.
# This software SHOULD NOT be used for diagnosis or treatment of patients.
#
# The generic UNet utilised here is inspired by the work of:
# J. Akeret, et al. Astronomy and Computing, vol. 18, pp. 35-39, 2017.
#
# Author: 
# Orod Razeghi
# King's College London


import os
import csv
import random
import numpy as np
import SimpleITK as sitk
from skimage.transform import resize


#setup
NOCLASS = 4
SAMPLES = 265
RESIZES = (320,320)
NIIPATH = "Data/CMR/CEMRA3D"
PRCPATH = "Data/Input/CEMRA2D"

#preparing data loading
for imgIDX in range(SAMPLES):
	imgPTH = NIIPATH + "/lge_" + str(imgIDX) + ".nii"
	segPTH = NIIPATH + "/lge_" + str(imgIDX) + "_label.nii"
	imgSTK = sitk.GetArrayFromImage(sitk.ReadImage(imgPTH))
	segSTK = sitk.GetArrayFromImage(sitk.ReadImage(segPTH))
	imgSCL = np.zeros(tuple([imgSTK.shape[0]]) + RESIZES)
	segSCL = np.zeros(tuple([segSTK.shape[0]]) + RESIZES)
	for idx in range(imgSTK.shape[0]):
		imgSCL[idx] = resize(imgSTK[idx], RESIZES, mode='constant', anti_aliasing=True)
		segSCL[idx] = resize(segSTK[idx], RESIZES, mode='constant', anti_aliasing=True, order=0)
	imgSTK = imgSCL
	segSTK = segSCL
	for z in range(imgSTK.shape[0]):
		if not os.path.exists(PRCPATH): os.makedirs(PRCPATH)
		X_path = '{}/img_{}_{}.nii.gz'.format(PRCPATH, imgIDX, z)
		Y_path = '{}/img_{}_{}_label.nii.gz'.format(PRCPATH, imgIDX, z)
		slicedImg = np.reshape(imgSTK[z], (1,) + imgSTK[z].shape)
		hotLabels = np.zeros((NOCLASS, segSTK[z].shape[0], segSTK[z].shape[1]), dtype=np.uint8)
		hotLabels[3,...] = ((segSTK[z]*255).astype(np.uint8)==3)
		hotLabels[2,...] = ((segSTK[z]*255).astype(np.uint8)==2)
		hotLabels[1,...] = ((segSTK[z]*255).astype(np.uint8)==1)
		hotLabels[0,...] = ((segSTK[z]*255).astype(np.uint8)==0)
		sitk.WriteImage(sitk.GetImageFromArray(slicedImg), X_path)
		sitk.WriteImage(sitk.GetImageFromArray(hotLabels), Y_path)
		print ('Slicing Image {} out of {} images'.format(imgIDX, SAMPLES))

#prepare image sets
np.random.seed(0)
set_idx = range(110, SAMPLES)
r_p = np.random.permutation(set_idx)
train_idx = r_p[range(0, int((SAMPLES-110)*.8))]
tests_idx = r_p[range(int((SAMPLES-110)*.8), SAMPLES-110)]

with open('./Data/trainIDX.csv', 'w') as f: csv.writer(f).writerow(train_idx)
with open('./Data/testsIDX.csv', 'w') as f: csv.writer(f).writerow(tests_idx)
