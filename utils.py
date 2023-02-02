import os
import glob
import numpy as np
from skimage import color, io, util, morphology

import constants as const

import pdb


def overlap_predictions(img, pred):
    """Overlaps the prediction on top of the input image. """

    red, green, blue = np.zeros(pred.shape), np.zeros(pred.shape), np.zeros(pred.shape)
    for i, color in enumerate(const.LABEL_COLORS):
        red[pred == i] = color[0]
        green[pred == i] = color[1]
        blue[pred == i] = color[2]

    # ensure image is 0-to-1 float
    if img.dtype == np.ubyte or img.max() > 1: img = (img/255).astype(float)
    overlap = overlap_rgb_labels(img,red=red,green=green,blue=blue)
    return overlap


def overlap_rgb_labels(img,red=None,green=None,blue=None,alpha=0.5):
    red = np.zeros_like(img) if red is None  else red
    green = np.zeros_like(img) if green is None  else green
    blue = np.zeros_like(img) if blue is None  else blue

    labels = np.stack((red,green,blue),axis=-1)
    label_ind = np.repeat(np.logical_or.reduce((red,green,blue))[:,:,np.newaxis],3,axis=-1)
    overlap = np.repeat(img[:,:,np.newaxis],3,axis=2)
    overlap[label_ind] = alpha*labels[label_ind] + (1-alpha)*overlap[label_ind]
    overlap = (overlap*255.0).astype(np.ubyte)

    return overlap

def postprocess_slice(pred, cube):
    """ postprocess single cross-sectional slice prediction """
    out = np.zeros_like(pred)
    for i in range(len(const.LABEL_COLORS)):
        if i == 0: continue
        # get image with single class
        tmp = np.zeros_like(pred)
        tmp[pred == i] = 1
        # remove unallowed slices
        allowed = np.zeros_like(pred)
        if const.ALLOWED_SLICES[cube][i]:
            allowed[const.ALLOWED_SLICES[cube][i],:] = 1
        tmp[np.logical_not(allowed)] = 0
        # remove small objects and holes
        # tmp = morphology.remove_small_objects(tmp,
        #                              min_size=const.MIN_INST_SIZE,
        #                              connectivity=2)
        # tmp = morphology.binary_closing(tmp,selem=morphology.disk(5)).astype(np.uint8)
        out[tmp == 1] = i

    return out
