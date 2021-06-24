
# Histogram Normalization 
import os
import numpy as np
import nibabel as nib
from nyul import nyul_train_standard_scale, nyul_apply_standard_scale, do_hist_normalization
import matplotlib.pyplot as plt
import pandas as pd
import os.path
from os import path

# # Mariana's code histogram normalization - https://github.com/sergivalverde/MRI_intensity_normalization 

# os.chdir("C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/controls/t1_biascorr/")

# # Input all images healthy (T1+PET) or normalise just T1 and then just PET ?
# dir_images="C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/controls/t1_biascorr/"
# img_dir = os.listdir("C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/controls/t1_biascorr")

# dir_list=[]

# for i in range(len(img_dir)):
#     im = "C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/controls/t1_biascorr/" + str(img_dir[i])
#     dir_list.append(im)

# standard_scale, perc = nyul_train_standard_scale(dir_list)


# for filename in sorted(os.listdir(dir_images)):
#     img = 'C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/controls/t1_biascorr/' + filename #this has skull 
#     img = nib.load(img)
#     affine = img.affine
#     img = np.asanyarray(img.dataobj)
    
#     img_norm = do_hist_normalization(img, perc, standard_scale) # shouldn't it be nyul_apply_standard_scale ?
    
#     nib.save(nib.Nifti1Image(img_norm, affine), os.path.join('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/histogram_norm/t1', filename))

# dir_images="C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/controls/petsuv/"
# img_dir = os.listdir("C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/controls/petsuv")

# dir_list=[]

# for i in range(len(img_dir)):
#     im = "C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/controls/petsuv/" + str(img_dir[i])
#     dir_list.append(im)


# standard_scale, perc = nyul_train_standard_scale(dir_list)


# for filename in sorted(os.listdir(dir_images)):
#     img = 'C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/controls/petsuv/' + filename #this has skull 
#     img = nib.load(img)
#     affine = img.affine
#     img = np.asanyarray(img.dataobj)
    
#     img_norm = do_hist_normalization(img, perc, standard_scale) # shouldn't it be nyul_apply_standard_scale ?
    
#     nib.save(nib.Nifti1Image(img_norm, affine), os.path.join('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/histogram_norm/pet', filename))



# # # what is the affine in the image?? why is it needed? It is saved as nib image affine ?
# # # Dividir imagens por 100 para ter entre 0-1 intensity values ?

# # Verify normalised images:
# # load both images:
# im_not_norm = nib.load('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/controls/t1_biascorr/ASSMI.nii.gz')
# im_norm = nib.load('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/histogram_norm/t1/ASSMI.nii.gz')

# im_not_norm = im_not_norm.get_fdata()
# im_norm = im_norm.get_fdata()

# #plot images:
# fig= plt.figure()
# ax = fig.add_subplot(1, 2, 1)
# plt.imshow(im_not_norm[:,:,90])
# ax.set_title('T1 original')
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# ax = fig.add_subplot(1, 2, 2)
# plt.imshow(im_norm[:,:,90])
# ax.set_title('T1 normalized')
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)


# # plot histogram before and after: 
# fig, axs = plt.subplots(2, 1, constrained_layout=True)
# f1 = axs[0].hist(im_not_norm.flatten(), bins=64, range=(-10,300))
# f2 = axs[1].hist(im_norm.flatten(), bins=64, range=(-10,300))
# axs[0].set_title('Image Original')
# axs[1].set_title('Image Normalized')


# # load both images:
# im_not_norm = nib.load('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/controls/petsuv/ASSMI.nii.gz')
# im_norm = nib.load('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/histogram_norm/pet/ASSMI.nii.gz')

# im_not_norm = im_not_norm.get_fdata()
# im_norm = im_norm.get_fdata()

# #plot images:
# fig= plt.figure()
# ax = fig.add_subplot(1, 2, 1)
# plt.imshow(im_not_norm[:,:,90])
# ax.set_title('PET original')
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# ax = fig.add_subplot(1, 2, 2)
# plt.imshow(im_norm[:,:,90])
# ax.set_title('PET normalized')
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)


# # plot histogram before and after: 
# fig, axs = plt.subplots(2, 1, constrained_layout=True)
# f1 = axs[0].hist(im_not_norm.flatten(), bins=64, range=(-10,300))
# f2 = axs[1].hist(im_norm.flatten(), bins=64, range=(-10,300))
# axs[0].set_title('Image Original')
# axs[1].set_title('Image Normalized')

# =============================================================================
# Using only one image as reference
# =============================================================================
os.chdir("C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/controls/t1_biascorr")

dir_images="C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/controls/t1_biascorr/"

img_dir = os.listdir("C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/controls/t1_biascorr")

#im_ref="C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/controls/t1_biascorr/ASSMI.nii.gz"

dir_list=[]

im = "C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/controls/petsuv/" + str(img_dir[1])
dir_list.append(im)


standard_scale, perc = nyul_train_standard_scale(dir_list)


for filename in sorted(os.listdir(dir_images)):
    img = 'C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/controls/t1_biascorr/' + filename #this has skull 
    img = nib.load(img)
    affine = img.affine
    img = np.asanyarray(img.dataobj)
    
    img_norm = do_hist_normalization(img, perc, standard_scale) # shouldn't it be nyul_apply_standard_scale ?
    
    nib.save(nib.Nifti1Image(img_norm, affine), os.path.join('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/histogram_norm/with_one_ref/t1', filename))

# load both images:
im_not_norm = nib.load('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/controls/t1_biascorr/ASSMI.nii.gz')
im_norm = nib.load('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/histogram_norm/with_one_ref/t1/ASSMI.nii.gz')

im_not_norm = im_not_norm.get_fdata()
im_norm = im_norm.get_fdata()

#plot images:
fig= plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(im_not_norm[:,:,90])
ax.set_title('T1 original')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = fig.add_subplot(1, 2, 2)
plt.imshow(im_norm[:,:,90])
ax.set_title('T1 normalized')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


# plot histogram before and after: 
fig, axs = plt.subplots(2, 1, constrained_layout=True)
f1 = axs[0].hist(im_not_norm.flatten(), bins=64, range=(-10,300))
f2 = axs[1].hist(im_norm.flatten(), bins=64, range=(-10,300))
axs[0].set_title('Image Original')
axs[1].set_title('Image Normalized')


