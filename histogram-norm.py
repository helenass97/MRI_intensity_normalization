
# Histogram Normalization 
import os
import numpy as np
import nibabel as nib
from nyul import nyul_train_standard_scale, nyul_apply_standard_scale, do_hist_normalization
import matplotlib.pyplot as plt
import pandas as pd
import os.path
from os import path

# # =============================================================================
# # USING ALL DATA TO TRAIN HIST NORM - uncomment so save the images in folder
# # =============================================================================

# # #FOR T1
# os.chdir("C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/t1/")

# dir_images="C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/t1/"
# img_dir = os.listdir("C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/t1")
# dir_list=[]

# for i in range(len(img_dir)):
#     im = "C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/t1/" + str(img_dir[i])
#     dir_list.append(im)

# standard_scale, perc = nyul_train_standard_scale(dir_list)


# for filename in sorted(os.listdir(dir_images)):
#     img = 'C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/t1/' + filename 
#     img = nib.load(img)
#     affine = img.affine # what is the affine in the image?? why is it needed? It is saved as nib image affine ?
#     img = np.asanyarray(img.dataobj)
#     img_norm = do_hist_normalization(img, perc, standard_scale) 
#     img_norm = img_norm/100
#     nib.save(nib.Nifti1Image(img_norm, affine), os.path.join('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/histogram_norm_brain/to_all_images/t1', filename))

# # FOR PET 
# os.chdir("C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/pet/")
# dir_images="C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/pet/"
# img_dir = os.listdir("C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/pet")
# dir_list=[]

# for i in range(len(img_dir)):
#     im = "C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/pet/" + str(img_dir[i])
#     dir_list.append(im)

# standard_scale, perc = nyul_train_standard_scale(dir_list)

# for filename in sorted(os.listdir(dir_images)):
#     img = 'C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/pet/' + filename 
#     img = nib.load(img)
#     affine = img.affine
#     img = np.asanyarray(img.dataobj)
#     img_norm = do_hist_normalization(img, perc, standard_scale) 
#     img_norm = img_norm/100
#     nib.save(nib.Nifti1Image(img_norm, affine), os.path.join('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/histogram_norm_brain/to_all_images/pet', filename))


# # Save average reference images T1 and PET 
# dir_images_pet = "C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/pet/"
# dir_images_t1 = "C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/t1/"

# # PET average image
# to_average_pet=[]
# for filename in sorted(os.listdir(dir_images_pet)):
#     img = 'C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/pet/' + filename 
#     img = nib.load(img)
#     affine = img.affine
#     img = np.asanyarray(img.dataobj)
#     to_average_pet.append(img)
    
# sum=0
# for i in range(len(to_average_pet)):
#     sum += to_average_pet[i]

# average_im_pet= sum/ len(to_average_pet)
    

# # T1 average image
# to_average_t1=[]
# for filename in sorted(os.listdir(dir_images_t1)):
#     img = 'C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/t1/' + filename 
#     img = nib.load(img)
#     affine = img.affine
#     img = np.asanyarray(img.dataobj)
#     to_average_t1.append(img)
    
# sum=0
# for i in range(len(to_average_t1)):
#    sum += to_average_t1[i]

# average_im_t1= sum/ len(to_average_t1)

# #save average images PET and T1 
# nib.save(nib.Nifti1Image(average_im_pet, affine), os.path.join('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/histogram_norm_brain/to_all_images/pet_ref_average', 'average_pet_ref'))
# nib.save(nib.Nifti1Image(average_im_t1, affine), os.path.join('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/histogram_norm_brain/to_all_images/t1_ref_average', 'average_t1_ref'))


# plot figures and histograms 

# load both images:
im_not_norm_pet = nib.load('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/histogram_norm_brain/to_all_images/pet_ref_average/average_pet_ref.nii')
im_not_norm_t1 = nib.load('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/histogram_norm_brain/to_all_images/t1_ref_average/average_t1_ref.nii')

im_norm_pet = nib.load('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/histogram_norm_brain/to_all_images/pet/ASSMI_brain.nii.gz')
im_norm_t1 = nib.load('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/histogram_norm_brain/to_all_images/t1/ASSMI_brain.nii.gz')

im_not_norm_pet = im_not_norm_pet.get_fdata()
im_not_norm_t1 = im_not_norm_t1.get_fdata()

im_norm_pet = im_norm_pet.get_fdata()
im_norm_t1 = im_norm_t1.get_fdata()

#plot images:
fig= plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(im_not_norm_pet[:,:,90])
ax.set_title('PET average image not norm')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = fig.add_subplot(1, 2, 2)
plt.imshow(im_norm_pet[:,:,90])
ax.set_title('PET normalized')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


# plot histogram before and after: 
fig, axs = plt.subplots(2, 1, constrained_layout=True)
f1 = axs[0].hist(im_not_norm_pet.flatten(), bins=64, range=(-10,50))
f2 = axs[1].hist(im_norm_pet.flatten(), bins=64, range=(-1,2))
axs[0].set_title('Image PET Original')
axs[1].set_title('Image PET Normalized')


#plot images:
fig= plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(im_not_norm_t1[:,:,90])
ax.set_title('T1 average image not norm')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = fig.add_subplot(1, 2, 2)
plt.imshow(im_norm_t1[:,:,90])
ax.set_title('T1 normalized')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


# plot histogram before and after: 
fig, axs = plt.subplots(2, 1, constrained_layout=True)
f1 = axs[0].hist(im_not_norm_t1.flatten(), bins=64, range=(-10,25))
f2 = axs[1].hist(im_norm_t1.flatten(), bins=64, range=(-1,2))
axs[0].set_title('Image T1 Original')
axs[1].set_title('Image T1 Normalized')


# #=============================================================================
# # Using only one image as reference - uncomment so save the images in folder
# #=============================================================================
# # For PET 
# os.chdir("C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/pet")
# dir_images="C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/pet/"
# img_dir = os.listdir("C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/pet")

# dir_list=[]
# im = "C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/pet/ASSMI_brain.nii.gz"
# dir_list.append(im)

# standard_scale, perc = nyul_train_standard_scale(dir_list)

# for filename in sorted(os.listdir(dir_images)):
#     img = 'C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/pet/' + filename 
#     img = nib.load(img)
#     affine = img.affine
#     img = np.asanyarray(img.dataobj)
#     img_norm = do_hist_normalization(img, perc, standard_scale) 
#     img_norm = img_norm/100
#     nib.save(nib.Nifti1Image(img_norm, affine), os.path.join('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/histogram_norm_brain/with_one_ref/pet', filename))


# # For T1 
# os.chdir("C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/t1")
# dir_images="C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/t1/"
# img_dir = os.listdir("C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/t1")

# dir_list=[]
# im = "C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/t1/ASSMI_brain.nii.gz"
# dir_list.append(im)

# standard_scale, perc = nyul_train_standard_scale(dir_list)

# for filename in sorted(os.listdir(dir_images)):
#     img = 'C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/t1/' + filename 
#     img = nib.load(img)
#     affine = img.affine
#     img = np.asanyarray(img.dataobj)
#     img_norm = do_hist_normalization(img, perc, standard_scale) 
#     img_norm = img_norm/100
#     nib.save(nib.Nifti1Image(img_norm, affine), os.path.join('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/histogram_norm_brain/with_one_ref/t1', filename))



# load both images:
im_not_norm_pet = nib.load('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/pet/ASSMI_brain.nii.gz')
im_norm_pet = nib.load('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/histogram_norm_brain/with_one_ref/pet/ASSMI_brain.nii.gz')

im_not_norm_t1 = nib.load('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/t1/ASSMI_brain.nii.gz')
im_norm_t1 = nib.load('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/histogram_norm_brain/with_one_ref/t1/ASSMI_brain.nii.gz')

im_not_norm_pet = im_not_norm_pet.get_fdata()
im_not_norm_t1 = im_not_norm_t1.get_fdata()

im_norm_pet = im_norm_pet.get_fdata()
im_norm_t1 = im_norm_t1.get_fdata()

#plot images:
fig= plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(im_not_norm_pet[:,:,90])
ax.set_title('PET original')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = fig.add_subplot(1, 2, 2)
plt.imshow(im_norm_pet[:,:,90])
ax.set_title('PET normalized')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


# plot histogram before and after: 
fig, axs = plt.subplots(2, 1, constrained_layout=True)
f1 = axs[0].hist(im_not_norm_pet.flatten(), bins=64, range=(-5,30))
f2 = axs[1].hist(im_norm_pet.flatten(), bins=64, range=(-1,2))
axs[0].set_title('Image PET Original')
axs[1].set_title('Image PET Normalized')


#plot images:
fig= plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(im_not_norm_t1[:,:,90])
ax.set_title('T1 original')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = fig.add_subplot(1, 2, 2)
plt.imshow(im_norm_t1[:,:,90])
ax.set_title('T1 normalized')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


# plot histogram before and after: 
fig, axs = plt.subplots(2, 1, constrained_layout=True)
f1 = axs[0].hist(im_not_norm_t1.flatten(), bins=64, range=(-5,10))
f2 = axs[1].hist(im_norm_t1.flatten(), bins=64, range=(-1,2))
axs[0].set_title('Image T1 Original')
axs[1].set_title('Image T1 Normalized')
