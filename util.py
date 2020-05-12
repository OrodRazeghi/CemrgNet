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


from __future__ import print_function, division, absolute_import, unicode_literals
import numpy as np
from PIL import Image


def to_rgb(img):
    img = np.atleast_3d(img)
    channels = img.shape[2]
    if channels < 3:
        img = np.tile(img, 3)    
    img[np.isnan(img)] = 0
    img -= np.amin(img)
    if np.amax(img) != 0: img /= np.amax(img)
    img *= 255
    return img

def crop_to_shape(data, shape):
    offset0 = (data.shape[1] - shape[1])//2
    offset1 = (data.shape[2] - shape[2])//2
    return data[:, offset0:(-offset0), offset1:(-offset1)]

def combine_img_prediction(data, gt, pred):
    ny = pred.shape[2]
    ch = data.shape[3]
    #img = np.concatenate((to_rgb(crop_to_shape(data, pred.shape).reshape(-1, ny, ch)), 
    #                      to_rgb(crop_to_shape(gt[..., 1], pred.shape).reshape(-1, ny, 1)), 
    #                      to_rgb(pred[..., 1].reshape(-1, ny, 1))), axis=1)
    img = np.concatenate((to_rgb(data.reshape(-1, ny, ch)), 
                          to_rgb(gt[..., 1].reshape(-1, ny, 1)), 
                          to_rgb(pred[..., 1].reshape(-1, ny, 1))), axis=1)
    return img

def save_image(img, path):
    Image.fromarray(img.round().astype(np.uint8)).save(path, 'JPEG', dpi=[300,300], quality=90)
