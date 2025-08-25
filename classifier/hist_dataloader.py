#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import print_function, division
import os
import sys
import torch
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.models as models
import torch.distributions as distributions
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pandas as pd
from skimage import io, transform, util
from skimage.color import rgb2gray
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time

import cv2

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


# In[2]:


class tiny_Dataset(Dataset):
    """Aida-17k dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with labels, comma .
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.csv = pd.read_csv(csv_file, header=None, dtype=str)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # ori
        _ipath = self.csv.iloc[idx, 0]
        img_name = os.path.join(self.root_dir,
                                _ipath)
#         print('idx', idx)
#         print(_ipath)
        image = io.imread(img_name)
        
        _lb = self.csv.iloc[idx, 1]
        label = np.zeros(6)
        label[int(_lb)] = 1.
    
        cls_label = np.array(label)
        sample = image, cls_label
        if self.transform:
            sample = self.transform(sample)

        return sample
    
class ProtoDataset(Dataset):
    """Read the dataset for prototypical network.
    Args:
        csv_file (string): Path to the csv file with labels, comma .
        root_dir (string): Directory with all the images.
        anchor_csv_file (string): Path to the csv file with anchor images.
        anchor_dir (string): Directory with all anchor images.
        transform (callable, optional): Optional transform to be applied on a sample.
        nb_cls (int): Number of classes in the dataset.
    """
    def __init__(self, csv_file, root_dir, anchor_csv_file=None, anchor_dir=None, transform=None, nb_cls=6):
        self.csv = pd.read_csv(csv_file, header=None, dtype=str)
        self.root_dir = root_dir
        self.transform = transform
        self.nb_cls = nb_cls

        self.support = {}  # class_id -> support image tensor
        self.anchor_dir = anchor_dir
        self.anchor_csv = pd.read_csv(anchor_csv_file, header=None, dtype=str) if anchor_csv_file else None
        for i in range(nb_cls):
            _ipath = self.anchor_csv.iloc[i, 0]
            img_name = os.path.join(self.anchor_dir, _ipath)
            image = io.imread(img_name)
            self.support[i] = transform((image,np.array([-1])))[0] if transform else image

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and label
        _ipath = self.csv.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, _ipath)
        image = io.imread(img_name)
        
        _lb = np.array([int(self.csv.iloc[idx, 1])])

        sample = image, _lb
        
        if self.transform:
            sample = self.transform(sample)

        return sample
    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
#         image, label, diqa = sample
        image, label = sample
        h, w = image.shape[:-1]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        image = transform.resize(image, (new_h, new_w))
        return image, label


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
#         image, label, diqa = sample
        image, label = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
#         image = np.expand_dims(image, axis=2)
        image = image.transpose((2, 0, 1))
        
        return torch.from_numpy(image).type(torch.FloatTensor), torch.from_numpy(label).type(torch.FloatTensor)

class Normalize(transforms.Normalize):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        super(Normalize, self).__init__(mean, std)

    def __call__(self, sample):
#         image, label, diqa = sample
        image, label = sample
        image = super(Normalize, self).__call__(image)
        return image, label
# In[ ]:

class RandomCrop(object):
    """Crop randomly the ROI of a image in a sample.
    Args:
        nb_crop (int): Number of crops to be made.
    """

    def __init__(self, nb_crop):
        assert isinstance(nb_crop, int) and nb_crop > 0
        self.nb_crop = nb_crop

    def __call__(self, sample):
        image, label = sample

        # copy labels to match the number of crops
        label = [label] * self.nb_crop

        crops = []
        try:
            # find the ROI
            gray = rgb2gray(image)
            # binary image
            th, im_th = cv2.threshold((gray*255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            im_th = util.invert(im_th)
            # find contours
            contours, hierarchy = cv2.findContours(im_th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            # find the biggest contour
            max_area = 0
            ci = -1
            img_cnt = image.copy()
            for i in range(len(contours)):
                cnt = contours[i]
                area = cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    ci = i
            cnt = contours[ci]
            # find countours with similar size (>80% of the biggest one)
            similar_cnt = [cnt]
            for i in range(len(contours)):
                if i == ci:
                    continue
                cnt = contours[i]
                area = cv2.contourArea(cnt)
                if area > 0.8 * max_area:
                    similar_cnt.append(cnt)

            # merge all similar contours
            all_cnt = np.vstack(similar_cnt)
            x, y, w, h = cv2.boundingRect(all_cnt)

            roi = image[y:y+h, x:x+w, :]
            img_h, img_w = image.shape[:-1]
            roi_h, roi_w = roi.shape[:-1]
            
            for _ in range(self.nb_crop):
                if roi_h > roi_w:
                    new_h = np.random.randint(0, roi_h/2-10)
                    new_w = new_h
                else:
                    new_w = np.random.randint(0, roi_w/2-10)
                    new_h = new_w
                top = max(int(roi_h/4), np.random.randint(1, roi_h - new_h))
                left = max(int(roi_h/4), np.random.randint(1, roi_w - new_w))

                crop = roi[new_h: min(top + new_h, roi_h), new_w: min(left + new_w, roi_w), :]
                crops.append(crop)
        except:
            print("Random Crop Failed, Use Original Image")
            crops.append(image)
        
        return crops, label
    
class MultiCropToTensor(object):
    """Convert list of ndarrays in sample to Tensors."""

    def __call__(self, sample):
        images, label = sample

        crops = []
        for image in images:
            # swap color axis because
            # numpy image: H x W x C
            # torch image: C x H x W
            image = image.transpose((2, 0, 1))
            crops.append(torch.from_numpy(image).type(torch.FloatTensor))
        
        return torch.stack(crops), torch.from_numpy(label).type(torch.FloatTensor)
    
class MultiCropRescale(object):
    """Rescale the list of crops in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        images, label = sample
        resized_crops = []
        for image in images:
            h, w = image.shape[:-1]
            if isinstance(self.output_size, int):
                if h > w:
                    new_h, new_w = self.output_size * h / w, self.output_size
                else:
                    new_h, new_w = self.output_size, self.output_size * w / h
            else:
                new_h, new_w = self.output_size
            new_h, new_w = int(new_h), int(new_w)
            image = transform.resize(image, (new_h, new_w))
            resized_crops.append(image)
        
        return resized_crops, label

class MultiCropNormalize(transforms.Normalize):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        super(MultiCropNormalize, self).__init__(mean, std)

    def __call__(self, sample):
        images, label = sample
        # images is a N * C * H * W tensor
        norm_crops = []
        for image in images:
            image = super(MultiCropNormalize, self).__call__(image)
            norm_crops.append(image)
        norm_crops = torch.stack(norm_crops)
        
        return norm_crops, label


