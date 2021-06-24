
# Histogram Normalization 
import os
import numpy as np
import nibabel as nib
from MRI_intensity_normalization.nyul import nyul_train_standard_scale, nyul_apply_standard_scale
import matplotlib.pyplot as plt
import pandas as pd
import os.path
from os import path

# Mariana's code histogram normalization - https://github.com/sergivalverde/MRI_intensity_normalization 

# Input all images healthy (T1+PET) or normalise just T1 and then just PET ?

img_dir = '/media/mds19/Storage/LongT1/ADNI/hist_norm/'
#mask_dir = '/media/mds19/Storage/LongT1/ADNI/Tissue_seg_reg/'

# def mask_bluring(mask):
#     blurred_mask = gaussian_filter(input=mask, sigma=0.5)
#     #blurred_mask[mask == 1] = 1
#     return blurred_mask

# get standard scale

#standard_scale, perc = nyul_train_standard_scale(img_dir,mask_dir)
standard_scale, perc = nyul_train_standard_scale(img_dir)


for filename in sorted(os.listdir('/media/mds19/Storage/OASIS/T1_brain_reg')):
    img = 'D:/dataset PET-MR/ForHelena_20210602/controls/t1_biascorr' + filename #this has skull 
    img = nib.load(img)
    affine = img.affine
    img = np.asanyarray(img.dataobj)
    masked = '/media/mds19/Storage/OASIS/Tissue_seg_reg/' + filename
    masked = nib.load(masked)
    masked = np.asanyarray(masked.dataobj)
    img_norm = do_hist_normalization(img, perc, standard_scale) # shouldn't it be nyul_apply_standard_scale ?
    
    # use mask if image is not brain-extracted: 
    #mask = np.zeros_like(masked)
    #mask[masked > 0] = 1
    #mask=mask_bluring(mask)
    #img_norm = np.multiply(img_norm, mask)
    
    nib.save(nib.Nifti1Image(img_norm, affine), os.path.join('D:/dataset PET-MR/ForHelena_20210602/histogram_norm/t1', filename))



