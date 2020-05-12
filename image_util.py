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
import glob
import random
import numpy as np
from PIL import Image
import SimpleITK as sitk


class BaseDataProvider(object):
    
    n_class = 2
    channels = 1

    def __init__(self, a_min=None, a_max=None):
        self.a_min = a_min if a_min is not None else -np.inf
        self.a_max = a_max if a_min is not None else np.inf

    def _load_data_and_label(self):
        data, label = self._next_data()            
        train_data = self._process_data(data)
        labels = self._process_labels(label)        
        train_data, labels = self._post_process(train_data, labels)        
        nx = train_data.shape[1]
        ny = train_data.shape[0]
        return train_data.reshape(1, ny, nx, self.channels), labels.reshape(1, ny, nx, self.n_class),
    
    def _process_labels(self, label):
        if self.n_class == 2:
            nx = label.shape[1]
            ny = label.shape[0]
            labels = np.zeros((ny, nx, self.n_class), dtype=np.float32)
            labels[..., 1] = label[...,1]
            labels[..., 0] = label[...,0]
            return labels        
        return label
    
    def _process_data(self, data):
        data = np.clip(np.fabs(data), self.a_min, self.a_max)
        data -= np.amin(data)
        #In case input image has only background
        if np.amax(data) != 0: data /= np.amax(data)
        return data
    
    def __call__(self, n):
        train_data, labels = self._load_data_and_label()
        nx = train_data.shape[1]
        ny = train_data.shape[2]    
        X = np.zeros((n, nx, ny, self.channels))
        Y = np.zeros((n, nx, ny, self.n_class))    
        X[0] = train_data
        Y[0] = labels
        for i in range(1, n):
            train_data, labels = self._load_data_and_label()
            X[i] = train_data
            Y[i] = labels    
        return X, Y

class ImageDataProvider(BaseDataProvider):
    
    def __init__(self, search_path, a_min=None, a_max=None, data_suffix=".tif", mask_suffix='_mask.tif', shuffle_data=True, n_class = 2):
        super(ImageDataProvider, self).__init__(a_min, a_max)
        self.data_suffix = data_suffix
        self.mask_suffix = mask_suffix
        self.file_idx = -1
        self.shuffle_data = shuffle_data
        self.n_class = n_class        
        self.data_files = self._find_data_files(search_path)        
        if self.shuffle_data:
            np.random.shuffle(self.data_files)        
        assert len(self.data_files) > 0, "No training files"
        print("Number of files used: %s" % len(self.data_files))        
        img = self._load_file(self.data_files[0])
        self.channels = 1 if len(img.shape) == 2 else img.shape[-1]
        
    def _find_data_files(self, search_path):
        all_files = glob.glob(search_path)
        return [name for name in all_files if self.data_suffix in name and not self.mask_suffix in name]
        
    def _load_file(self, path, dtype=np.float32):
        return np.transpose(sitk.GetArrayFromImage(sitk.ReadImage(path)).astype(dtype), (1,2,0))

    def _cylce_file(self):
        self.file_idx += 1
        if self.file_idx >= len(self.data_files):
            self.file_idx = 0 
            if self.shuffle_data:
                #print ('SHUFFLE')
                #return
                np.random.shuffle(self.data_files)
        
    def _next_data(self):
        #choice = random.choice(['BP','MV'])
        #while True:
           self._cylce_file()
           image_name = self.data_files[self.file_idx]
           label_name = image_name.replace(self.data_suffix, self.mask_suffix)
           img = self._load_file(image_name, np.float32)
           label = self._load_file(label_name, np.uint8)
           #if choice == 'BP':
           #   cBP = np.sum(label[:,:,1]==1)
           #   if cBP > 1: break
           #   else: continue 
           #elif choice == 'VN':
           #   cVN = np.sum(label[:,:,2]==1)
           #   if cVN > 25: break
           #   else: continue
           #elif choice == 'MV':
           #   cMV = np.sum(label[:,:,3]==1)
           #   if cMV > 10: break
           #   else: continue
           return img,label
