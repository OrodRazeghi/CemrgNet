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
import tensorflow as tf
from tf_unet import unet, util, image_util


#setup
PRCPATH = "Data/Input/CEMRA2D"
TRNPATH = "Data/Input/TrainSet"
OUTPATH = "Data/Model5Observer"
CHANNEL = 1
NOCLASS = 4
UNDEPTH = 5
NOFEATS = 32
DROPOUT = 0.5
TBATCHS = 16
VBATCHS = 32
ITERATS = 500
NEPOCHS = 100


#prepare augmented training images
with open("./Data/trainIDX5observer.csv") as f: train_idx = [r for r in csv.reader(f)]; train_idx = [int(i) for i in train_idx[0]]
for imgIDX in train_idx:
	for f in glob.glob(os.path.join(PRCPATH, "img_" + str(imgIDX) + "*.nii.gz")):
		shutil.copy(f, TRNPATH)
data_provider = image_util.ImageDataProvider(TRNPATH + "/*", data_suffix=".nii.gz", mask_suffix='_label.nii.gz', n_class=NOCLASS)

#network
net = unet.Unet(layers=UNDEPTH, features_root=NOFEATS, channels=CHANNEL, n_class=NOCLASS, cost='dice_coefficient')
print (np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables()]))

#training
trainer = unet.Trainer(net, batch_size=TBATCHS, verification_batch_size=VBATCHS, optimizer='adam')
path = trainer.train(data_provider, OUTPATH, training_iters=ITERATS, epochs=NEPOCHS, dropout=DROPOUT, restore=True)
