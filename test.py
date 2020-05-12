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
import glob
import shutil
import numpy as np
import SimpleITK as sitk
import tensorflow as tf
from tf_unet import unet, image_util


#setup
PRCPATH = "Data/Input/CEMRA2D"
TSTPATH = "Data/Input/TestsSet"
RESPATH = "Data/Predictions"
MDLPATH = "Data/Model5Observer/model.ckpt"
CHANNEL = 1
NOCLASS = 4
UNDEPTH = 5
NOFEATS = 32
THRSHLD = 0.5

def calcMetrics(A, B):
	metrics = {}
	A = A.astype(bool)
	B = B.astype(bool)
	TP = np.sum(np.logical_and(A, B))
	FP = np.sum(np.logical_and(A, np.logical_not(B)))
	TN = np.sum(np.logical_and(np.logical_not(A), np.logical_not(B)))
	FN = np.sum(np.logical_and(np.logical_not(A), B))
	if  2 * TP + FP + FN != 0: metrics["Dice"] = (2 * TP) / float(2 * TP + FP + FN)
	else: metrics["Dice"] = 1
	if TP + TN + FP + FN != 0: metrics["Accuracy"] = (TP + TN) / float(TP + TN + FP + FN)
	else: metrics["Accuracy"] = 1
	if TP + FN != 0: metrics["Sensitivity"] = TP / float(TP + FN)
	else: metrics["Sensitivity"] = 1
	if TN + FP != 0: metrics["Specificity"] = TN / float(TN + FP)
	else: metrics["Specificity"] = 1
	if TP + FP != 0: metrics["Precision"] = TP / float(TP + FP)
	else: metrics["Precision"] = 1
	return metrics

#prepare test images
with open("./Data/trainIDX5observer.csv") as f: tests_idx = [r for r in csv.reader(f)]; tests_idx = [int(i) for i in tests_idx[0]]
for imgIDX in tests_idx:
	for f in glob.glob(os.path.join(PRCPATH, "img_" + str(imgIDX) + "*.nii.gz")):
		shutil.copy(f, TSTPATH)

#network
data_provider = image_util.ImageDataProvider(TSTPATH + "/*", data_suffix=".nii.gz", mask_suffix="_label.nii.gz", n_class=NOCLASS)
net = unet.Unet(layers=UNDEPTH, features_root=NOFEATS, channels=CHANNEL, n_class=NOCLASS, cost="dice_coefficient")

#prediction
results = []
noFiles = len(data_provider.data_files)
init = tf.global_variables_initializer()
with tf.Session() as sess:	
	sess.run(init)
	net.restore(sess, MDLPATH)
	for img in range(noFiles):
		fileName = data_provider.data_files[img]
		fileName = fileName[fileName.rfind('/')+1:]
		fileName = fileName[0:fileName.find('.')]
		x_test, y_test = data_provider(1)
		prdcts = net.fastPredict(sess, x_test)
		x = np.transpose(x_test[0,...], (2,0,1))
		y = np.transpose(y_test[0,...], (2,0,1))
		p = np.transpose(prdcts[0,...], (2,0,1))
		l = np.zeros((1,x.shape[1],x.shape[2]))
		l[0, y[1,...]>THRSHLD] = 1
		l[0, y[2,...]>THRSHLD] = 2
		l[0, y[3,...]>THRSHLD] = 3
		o = np.zeros((1,x.shape[1],x.shape[2]))
		o[0, p[1,...]>THRSHLD] = 1
		o[0, p[2,...]>.1] = 2
		o[0, p[3,...]>THRSHLD] = 3
		sitk.WriteImage(sitk.GetImageFromArray(x), RESPATH + "/" + fileName + ".nii.gz")
		sitk.WriteImage(sitk.GetImageFromArray(l), RESPATH + "/" + fileName + "_labels.nii.gz")
		sitk.WriteImage(sitk.GetImageFromArray(p), RESPATH + "/" + fileName + "_probls.nii.gz")
		sitk.WriteImage(sitk.GetImageFromArray(o), RESPATH + "/" + fileName + "_output.nii.gz")
		metrics = calcMetrics(l,o)
		results.append(metrics)
		print ("Testing image {}".format(img))

#3D stack
for imgIDX in tests_idx:
	lbl3D = []
	out3D = []
	stack = len(glob.glob(os.path.join(RESPATH, "img_" + str(imgIDX) + "_*_labels.nii.gz")))
	for z in range(0,stack):
		lbl2D = sitk.GetArrayFromImage(sitk.ReadImage(RESPATH + "/img_" + str(imgIDX) + "_" + str(z) + "_labels.nii.gz"))
		lbl3D.append(lbl2D[0,...])
		out2D = sitk.GetArrayFromImage(sitk.ReadImage(RESPATH + "/img_" + str(imgIDX) + "_" + str(z) + "_output.nii.gz"))
		out3D.append(out2D[0,...])
	sitk.WriteImage(sitk.GetImageFromArray(np.array(lbl3D)), RESPATH + "/lbl_" + str(imgIDX) + "_3D.nii.gz")
	sitk.WriteImage(sitk.GetImageFromArray(np.array(out3D)), RESPATH + "/out_" + str(imgIDX) + "_3D.nii.gz")

#All results
metric_means = {}
metric_stdvs = {}
for metric in results[0].items():
	all = [results[i][metric[0]] for i in range(len(results))]
	metric_means[metric[0]] = np.mean(all)
	metric_stdvs[metric[0]] = np.std(all)
print ("=========== Results ===========")
print ("Means of metrics: \n{}\n".format(metric_means))
print ("Standard deviations of metrics: \n{}\n".format(metric_stdvs))
