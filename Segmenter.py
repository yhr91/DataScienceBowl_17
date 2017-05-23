# Functional that performs segmentation
# Inputs: Image, threshold
# BMI 260 Assignment 1

import dicom
import numpy as np
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.color import label2rgb

from os import listdir
from os.path import join

from mpl_toolkits.mplot3d import Axes3D

volume_thresh = 1e6

def vol_segment(vol, thresh):
    # This is a function that accepts a lung CT image and segments out the lungs from the background
    # It requires a binary treshold value

    # Let's create a volume of the segmented slices
    segmented_slices = np.zeros(vol.shape)
    for d in range(vol.shape[2]):
            segmented_slices[:,:,d] = area_segment(vol[:,:,d], thresh)/50

    # It looks like there's still some noise to clean up. Using the volume instead of the areas may be a good approach
    labelled_vols, no_regions = label(segmented_slices, return_num = 'TRUE')

    regions = {}
    region_vols = np.zeros(no_regions+1)
    for r in range(no_regions+1):
        regions[r] = np.where(labelled_vols == r) # Track the pixels themselves
        region_vols[r] = len(regions[r][0]) # As well as the volume of the region

    select_vols = np.where(region_vols > volume_thresh)[0][1:] # Remove the first element becuase it's background
    
    top_vols = region_vols[np.argsort(region_vols)[::-1][:3]]
    
    for r in select_vols:
        labelled_vols[labelled_vols == r] = -1
    labelled_vols[labelled_vols != -1] = 0
    labelled_vols[labelled_vols == -1] = 1
    
    return labelled_vols, top_vols

def area_segment(image, thresh):
    
    # Use the previous calculated global Otsu threshold to create a binary image
    threshed_image = np.zeros(image.shape)
    threshed_image[image >= thresh]=1
    
    # Let's close up the contours
    kernel = np.ones((4,4))
    closed = cv2.morphologyEx(threshed_image, cv2.MORPH_OPEN, kernel)
    
    # Label the different regions using simple connectivity maps
    labelled, no_regions = label(closed, background = 1, return_num='TRUE')

    # Keep track of the pixels assigned to each region
    regions = {}
    for r in range(no_regions+1):
        regions[r] = np.where(labelled == r) # Track the pixels themselves
        
    # Fine region labels that have a presence along the edges
    edges = np.concatenate([labelled[1,:], labelled[-1,:], labelled[:,1], labelled[:,-1]])
    edge_regions = np.unique(edges)

    # Add the background region '0' here too
    edge_regions = np.append(edge_regions,0)
    
    # Remove the edge regions and background
    select_regions = [i for i in range(no_regions+1) if i not in edge_regions]
    
    # Now create an image with only the selected regions
    for r in select_regions:
        labelled[labelled == r] = 50
    labelled[labelled != 50] = 0
    
    return labelled