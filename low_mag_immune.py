#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 16:38:09 2023

@author: brandon
"""


import numpy as np
import matplotlib.pyplot as plt
from skimage.filters import gaussian, threshold_multiotsu
from skimage.measure import regionprops, label
from skimage.morphology import binary_opening, disk
from skimage.io import imread
from glob import glob
from scipy.interpolate import interp1d
from matplotlib import rc


def mask_larvae(im, thresh=150):
    im = gaussian(im, sigma=3, preserve_range=True)
    mask = im > thresh
    mask = binary_opening(mask, disk(5))
    labels = label(mask)
    regions = regionprops(labels)
    areas = np.array([region.area for region in regions])
    big_id = np.where(areas == np.max(areas))[0]
    mask = labels == big_id + 1
    
    return mask.astype('float32')


def compute_line_dist(im, larvae_thresh=150, signal_thresh=None, correction=None, bins=None, short_axis=0):
    "short_axis=0 means larva is horizontal"
    if bins is None:
        bins = np.linspace(0, 1, 100)

    # segment out just the larvae
    larvae_mask = mask_larvae(im, larvae_thresh)
    im = (im * larvae_mask).astype('float32')
    
    # auto thresh: multiotsu on log-transformed image
    if signal_thresh is None:
        signal_thresh = 10 ** threshold_multiotsu(np.log10(im[larvae_mask > 0]))[-1]
    
    # background subtract
    im = im - signal_thresh
    im[im < 0] = 0
    
    # compute line dist
    line_dist = np.sum(im, axis=short_axis) #/ np.clip(np.sum(im > 0, axis=short_axis), a_min=1, a_max=np.inf)
    
    # normalize line dist length from the beginning to end of the larvae mask
    mask_1d = np.sum(larvae_mask, axis=short_axis) > 0
    mask_1d_ids = np.where(mask_1d)[0]
    start = mask_1d_ids[0]
    stop = mask_1d_ids[-1]
    line_dist = line_dist[start:stop]
    
    # use interpolation to ensure the line dist has a precise number of x bins
    x = np.linspace(0, 1, len(line_dist))
    line_dist = interp1d(x, line_dist)(bins)
    if correction is not None:
        scale_loc = int(0.8 * len(line_dist))
        correction = correction * line_dist[scale_loc] / correction[scale_loc]
        line_dist -= correction
    
    line_dist[line_dist < 0] = 0
        
    return line_dist
    

def compute_all_line_dists(path_list, larvae_thresh=150, signal_thresh=200, correction=None, bins=None, short_axis=0):
    all_line_dists = []
    for path in path_list:
        files = sorted(glob(path + '/*.tif'))
        for file in files:
            im = imread(file)
            line_dist = compute_line_dist(im, larvae_thresh, signal_thresh, correction, bins, short_axis=short_axis)
            all_line_dists.append(line_dist)
    
    return np.array(all_line_dists)
            
        
def compute_median_inten(im, larvae_thresh, signal_thresh=None):
    larvae_mask = mask_larvae(im, larvae_thresh)
    im = (im * larvae_mask).astype('float32')
    
    # auto thresh: multiotsu on log-transformed image
    if signal_thresh is None:
        signal_thresh = 10 ** threshold_multiotsu(np.log10(im[larvae_mask > 0]))[-1]
    
    # background subtract
    im = im - signal_thresh
    im[im < 0] = 0
    
    return np.median(im[im > 0])
        

def compute_all_median_intens(path_list, larvae_thresh, signal_thresh):
    all_medians = []
    for path in path_list:
        files = sorted(glob(path + '/*.tif'))
        for file in files:
            im = imread(file)
            line_dist = compute_median_inten(im, larvae_thresh, signal_thresh)
            all_medians.append(line_dist)
    
    return np.array(all_medians)
    

def compute_intens_stat(im, larvae_thresh, signal_thresh, func):
    """compute an arbitrary statistic (specified by func) of all pixel intensities within the signal mask"""
    larvae_mask = mask_larvae(im, larvae_thresh)
    
    # throw out last 10% of larvae to remove food/reflection in posterior.
    mask_1d = np.sum(larvae_mask, axis=0) > 0
    mask_1d_ids = np.where(mask_1d)[0]
    start = mask_1d_ids[0]
    stop = mask_1d_ids[-1]
    length = stop - start
    length_90 = int(start + 0.9 * length)
    larvae_mask[:, length_90:] = 0
    
    im = (im * larvae_mask).astype('float32')
    
    
    # auto thresh: multiotsu on log-transformed image
    if signal_thresh is None:
        signal_thresh = 10 ** threshold_multiotsu(np.log10(im[larvae_mask > 0]))[-1]
    
    # background subtract
    im = im - signal_thresh
    im[im < 0] = 0
    
    return func(im[im > 0])
        

def compute_all_intens_stat(path_list, larvae_thresh, signal_thresh, func):
    all_stats = []
    for path in path_list:
        files = sorted(glob(path + '/*.tif'))
        for file in files:
            im = imread(file)
            stats = compute_intens_stat(im, larvae_thresh, signal_thresh, func)
            all_stats.append(stats)
    
    return np.array(all_stats)


def compute_left_right_dist(im, ap_start=0.3, ap_stop=0.7, larvae_thresh=150, signal_thresh=None, bins=None, short_axis=0):
    "short_axis=0 means larva is horizontal. compute left-right distribution of fluorescence intensity, within the ap bounds of ap_start and ap_stop"
    if bins is None:
        bins = np.linspace(0, 1, 100)
    if short_axis == 0:
        long_axis = 1
    elif short_axis == 1:
        long_axis = 0
    
    # segment out just the larvae
    larvae_mask = mask_larvae(im, larvae_thresh)
    im = (im * larvae_mask).astype('float32')
    
    # auto thresh: multiotsu on log-transformed image
    if signal_thresh is None:
        signal_thresh = 10 ** threshold_multiotsu(np.log10(im[larvae_mask > 0]))[-1]
    
    # background subtract
    im = im - signal_thresh
    im[im < 0] = 0
    
    # get ap axis
    mask_1d = np.sum(larvae_mask, axis=short_axis) > 0
    mask_1d_ids = np.where(mask_1d)[0]
    start = mask_1d_ids[0]
    stop = mask_1d_ids[-1]
    pixel_ids = np.arange(start, stop)
    pixel_start = int(np.round(ap_start * len(pixel_ids))) + start
    pixel_stop = int(np.round(ap_stop * len(pixel_ids))) + start

    if long_axis == 1:
        im = im[:, pixel_start:pixel_stop]
    else:
        im = im[pixel_start:pixel_stop]
        
    # get start and stop of left-right axis
    left_right_mask = np.sum(im, axis=long_axis) > 0
    left_right_mask_ids = np.where(left_right_mask)[0]
    if len(left_right_mask_ids) == 0:
        return np.zeros(len(bins))
    
    left_start = left_right_mask_ids[0]
    right_end = left_right_mask_ids[-1]
    if long_axis == 1:
        """TODO: need to flip dist in this case"""
        im = im[left_start:right_end]
    else:
        im = im[:, left_start:right_end]
    
    left_right_dist = np.sum(im, axis=long_axis) / np.clip(np.sum(im > 0, axis=long_axis), a_min=1, a_max=np.inf)
    
    # use interpolation to ensure the line dist has a precise number of x bins
    x = np.linspace(0, 1, len(left_right_dist))
    left_right_dist = interp1d(x, left_right_dist)(bins)
    
    
    left_right_dist[left_right_dist < 0] = 0
    
    return left_right_dist


def compute_all_left_right_dists(path_list, ap_start=0.3, ap_stop=0.7, larvae_thresh=150, signal_thresh=None, bins=None, short_axis=0):
    all_left_right_dists = []
    for path in path_list:
        files = sorted(glob(path + '/*.tif'))
        for file in files:
            im = imread(file)
            left_right_dist = compute_left_right_dist(im, ap_start, ap_stop, larvae_thresh, signal_thresh, bins, short_axis=short_axis)
            all_left_right_dists.append(left_right_dist)
    
    return np.array(all_left_right_dists)
    