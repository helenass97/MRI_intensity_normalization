
# Histogram Normalization 
import os
import numpy as np
import nibabel as nib
from nyul import nyul_train_standard_scale, nyul_apply_standard_scale, do_hist_normalization
import matplotlib.pyplot as plt
import pandas as pd
import os.path
from os import path
import torch 
import torchvision 

# =============================================================================
# USING ALL DATA TO TRAIN HIST NORM - uncomment so save the images in folder
# =============================================================================

# #FOR CONTROLS

# #FOR T1 
# dir_images="E:/6.Helena/Affine_reg_MNI_1mm/controls/t1/"
# img_dir = os.listdir(dir_images)
# dir_list=[]

# for i in range(len(img_dir)):
#     im = dir_images + str(img_dir[i])
#     dir_list.append(im)

# standard_scale, perc = nyul_train_standard_scale(dir_list)


# for filename in sorted(os.listdir(dir_images)):
#     img = dir_images + filename 
#     img = nib.load(img)
#     affine = img.affine # what is the affine in the image?? why is it needed? It is saved as nib image affine ?
#     img = np.asanyarray(img.dataobj)
#     img_norm = do_hist_normalization(img, perc, standard_scale) 
#     img_norm = img_norm / 100
#     nib.save(nib.Nifti1Image(img_norm, affine), os.path.join('E:/6.Helena/Norm_data_aff_1mm/controls/t1/', filename)) 

# #FOR PET 
# dir_images="E:/6.Helena/Affine_reg_MNI_1mm/controls/pet/"
# img_dir = os.listdir(dir_images)
# dir_list=[]

# for i in range(len(img_dir)):
#     im = dir_images + str(img_dir[i])
#     dir_list.append(im)

# standard_scale, perc = nyul_train_standard_scale(dir_list)

# for filename in sorted(os.listdir(dir_images)):
#     img = dir_images + filename 
#     img = nib.load(img)
#     affine = img.affine
#     img = np.asanyarray(img.dataobj)
#     img_norm = do_hist_normalization(img, perc, standard_scale) 
#     img_norm = img_norm / 100
#     nib.save(nib.Nifti1Image(img_norm, affine), os.path.join('E:/6.Helena/Norm_data_aff_1mm/controls/pet/', filename))



# #FOR PATIENTS - need to normalise it in respect to the controls 

# dir_images="E:/6.Helena/Affine_reg_MNI_1mm/controls/t1/"
# img_dir = os.listdir(dir_images)
# dir_list=[]

# #patients data to normalize
# dir_images_save="E:/6.Helena/Affine_reg_MNI_1mm/patients/t1/"

# for i in range(len(img_dir)):
#     im = dir_images + str(img_dir[i])
#     dir_list.append(im)

# standard_scale, perc = nyul_train_standard_scale(dir_list)


# for filename in sorted(os.listdir(dir_images_save)):
#     img = dir_images_save + filename 
#     img = nib.load(img)
#     affine = img.affine # what is the affine in the image?? why is it needed? It is saved as nib image affine ?
#     img = np.asanyarray(img.dataobj)
#     img_norm = do_hist_normalization(img, perc, standard_scale) 
#     img_norm = img_norm / 100
#     nib.save(nib.Nifti1Image(img_norm, affine), os.path.join('E:/6.Helena/Norm_data_aff_1mm/patients/t1/', filename)) 

# # FOR PET 
# dir_images="E:/6.Helena/Affine_reg_MNI_1mm/controls/pet/"
# img_dir = os.listdir(dir_images)
# dir_list=[]

# #patients data to normalize
# dir_images_save="E:/6.Helena/Affine_reg_MNI_1mm/patients/pet/"

# for i in range(len(img_dir)):
#     im = dir_images + str(img_dir[i])
#     dir_list.append(im)

# standard_scale, perc = nyul_train_standard_scale(dir_list)

# for filename in sorted(os.listdir(dir_images_save)):
#     img = dir_images_save + filename 
#     img = nib.load(img)
#     affine = img.affine
#     img = np.asanyarray(img.dataobj)
#     img_norm = do_hist_normalization(img, perc, standard_scale) 
#     img_norm = img_norm / 100
#     nib.save(nib.Nifti1Image(img_norm, affine), os.path.join('E:/6.Helena/Norm_data_aff_1mm/patients/pet/', filename))


#### Dont need:

# # Save average reference images T1 and PET 
# dir_images_pet = "C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/patients/pet/"
# dir_images_t1 = "C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/patients/t1/"

# # PET average image
# to_average_pet=[]
# for filename in sorted(os.listdir(dir_images_pet)):
#     img = dir_images_pet + filename 
#     img = nib.load(img)
#     affine_pet = img.affine
#     img = np.asanyarray(img.dataobj)
#     to_average_pet.append(img)
    
# sum=0
# for i in range(len(to_average_pet)):
#     sum += to_average_pet[i]

# average_im_pet= sum/ len(to_average_pet)
    

# # T1 average image
# to_average_t1=[]
# for filename in sorted(os.listdir(dir_images_t1)):
#     img = dir_images_t1 + filename 
#     img = nib.load(img)
#     affine_t1 = img.affine
#     img = np.asanyarray(img.dataobj)
#     to_average_t1.append(img)
    
# sum=0
# for i in range(len(to_average_t1)):
#     sum += to_average_t1[i]

# average_im_t1= sum/ len(to_average_t1)

# #save average images PET and T1 
# nib.save(nib.Nifti1Image(average_im_pet, affine_pet), os.path.join('C:/Users/helen/Desktop/dataset_norm/patients/avg_pet/', 'average_pet_ref'))
# nib.save(nib.Nifti1Image(average_im_t1, affine_t1), os.path.join('C:/Users/helen/Desktop/dataset_norm/patients/avg_t1/', 'average_t1_ref'))


## plot figures and histograms 

#load images:
im_norm_pet_p = nib.load('E:/6.Helena/Norm_data_aff_1mm/patients/pet/mMR_BR1_067_brain_aff_1mm.nii.gz')
im_norm_t1_p = nib.load('E:/6.Helena/Norm_data_aff_1mm/patients/t1/mMR_BR1_067_brain_aff_1mm.nii.gz')

im_norm_t1_c = nib.load('E:/6.Helena/Norm_data_aff_1mm/controls/t1/AZAIS_brain_aff_1mm.nii.gz')
im_norm_pet_c = nib.load('E:/6.Helena/Norm_data_aff_1mm/controls/pet/AZAIS_brain_aff_1mm.nii.gz')

#controls
im_norm_pet_c = im_norm_pet_c.get_fdata()
im_norm_t1_c = im_norm_t1_c.get_fdata()

#patients
im_norm_pet_p = im_norm_pet_p.get_fdata()
im_norm_t1_p = im_norm_t1_p.get_fdata()


#plot images:
fig= plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(im_norm_pet_c[:,:,90], cmap="gray")
ax.set_title('PET norm controls')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = fig.add_subplot(1, 2, 2)
plt.imshow(im_norm_t1_c[:,:,90], cmap="gray")
ax.set_title('T1 norm controls')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


# plot histogram before and after: 
fig, axs = plt.subplots(2, 1, constrained_layout=True)
f1 = axs[0].hist(im_norm_pet_c.flatten(), bins=64, range=(0,1))
f2 = axs[1].hist(im_norm_t1_c.flatten(), bins=64, range=(0,1))
axs[0].set_title('PET controls')
axs[1].set_title('T1 controls')


#plot images:
fig= plt.figure()
ax = fig.add_subplot(1, 2, 1)
plt.imshow(im_norm_pet_p[:,:,90], cmap="gray")
ax.set_title('PET norm patients')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)
ax = fig.add_subplot(1, 2, 2)
plt.imshow(im_norm_t1_p[:,:,90], cmap="gray")
ax.set_title('T1 norm patients')
ax.get_xaxis().set_visible(False)
ax.get_yaxis().set_visible(False)


# plot histogram before and after: 
fig, axs = plt.subplots(2, 1, constrained_layout=True)
f1 = axs[0].hist(im_norm_pet_p.flatten(), bins=64, range=(0,1))
f2 = axs[1].hist(im_norm_t1_p.flatten(), bins=64, range=(0,1))
axs[0].set_title('PET patients')
axs[1].set_title('T1 patients')



# # Try get range 0-1 - modify nyul to 0.05-0.95

# #FOR T1

# #normalized before
# im_norm_t1 = nib.load('C:/Users/helen/Desktop/norm_data_1mm/patients/t1/mMR_BR1_002_MNI_nonlin_1mm.nii.gz')
# im_norm_t1 = im_norm_t1.get_fdata()

# dir_images="C:/Users/helen/Desktop/dataset_PET-MR/Nonlin-affine-MNI_1mm/controls/t1/"
# img_dir = os.listdir(dir_images)
# dir_list=[]


# for i in range(len(img_dir)):
#     im = dir_images + str(img_dir[i])
#     dir_list.append(im)

# standard_scale, perc = nyul_train_standard_scale(dir_list)


# img = "C:/Users/helen/Desktop/dataset_PET-MR/Nonlin-affine-MNI_1mm/patients/t1/" + 'mMR_BR1_002_MNI_nonlin_1mm.nii.gz' 
# img = nib.load(img)
# affine = img.affine
# img = np.asanyarray(img.dataobj)
# img_norm = do_hist_normalization(img, perc, standard_scale) 
# img_norm = img_norm/100
# nib.save(nib.Nifti1Image(img_norm, affine), os.path.join('C:/Users/helen/Desktop/hist_norm_new_range/', 'mMR_BR1_002_t1_new_range.nii.gz'))


# #plot images:
# fig= plt.figure()
# ax = fig.add_subplot(1, 2, 1)
# plt.imshow(img_norm[:,:,90])
# ax.set_title('T1 new norm')
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# ax = fig.add_subplot(1, 2, 2)
# plt.imshow(im_norm_t1[:,:,90])
# ax.set_title('T1 old norm')
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)


# # plot histogram before and after: 
# fig, axs = plt.subplots(2, 1, constrained_layout=True)
# f1 = axs[0].hist(img_norm.flatten(), bins=64, range=(np.min(img_norm),np.max(img_norm)))
# f2 = axs[1].hist(im_norm_t1.flatten(), bins=64, range=(np.min(im_norm_t1),np.max(im_norm_t1)))
# axs[0].set_title('Image T1 new norm')
# axs[1].set_title('Image T1 old Norm')

# #####

# fig, axs = plt.subplots(2, 1, constrained_layout=True)
# f1 = axs[0].hist(img_norm.flatten(), bins=64, range=(0,1))
# f2 = axs[1].hist(im_norm_t1.flatten(), bins=64, range=(0,1))
# axs[0].set_title('Hist T1 : percent range 5-95')
# axs[1].set_title('Hist T1 : percent range 0-100')

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



# # load both images:
# im_not_norm_pet = nib.load('C:/Users/helen/Desktop/dataset_PET-MR/Nonlin-affine-MNI_1mm/controls/t1/FERRA_MNI_nonlin_1mm.nii.gz')
# im_norm_pet = nib.load('C:/Users/helen/Desktop/dataset_PET-MR/Nonlin-affine-MNI_1mm/controls/pet/FERRA_MNI_nonlin.nii.gz')

# # im_not_norm_t1 = nib.load('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/original_data_brain/controls/t1/ASSMI_brain.nii.gz')
# # im_norm_t1 = nib.load('C:/Users/helen/Desktop/dataset_PET-MR/ForHelena_20210602/histogram_norm_brain/with_one_ref/t1/ASSMI_brain.nii.gz')

# im_not_norm_pet = im_not_norm_pet.get_fdata()
# #im_not_norm_t1 = im_not_norm_t1.get_fdata()

# im_norm_pet = im_norm_pet.get_fdata()
# #im_norm_t1 = im_norm_t1.get_fdata()

# #plot images:
# fig= plt.figure()
# ax = fig.add_subplot(1, 2, 1)
# plt.imshow(im_not_norm_pet[:,:,90])
# ax.set_title('PET original')
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# ax = fig.add_subplot(1, 2, 2)
# plt.imshow(im_norm_pet[:,:,90])
# ax.set_title('PET normalized')
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)


# plot histogram before and after: 
# fig, axs = plt.subplots(2, 1, constrained_layout=True)
# f1 = axs[0].hist(im_not_norm_pet.flatten(), bins=64, range=(-5,30))
# f2 = axs[1].hist(im_norm_pet.flatten(), bins=64, range=(-1,2))
# axs[0].set_title('Image PET Original')
# axs[1].set_title('Image PET Normalized')


# #plot images:
# fig= plt.figure()
# ax = fig.add_subplot(1, 2, 1)
# plt.imshow(im_not_norm_t1[:,:,90])
# ax.set_title('T1 original')
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)
# ax = fig.add_subplot(1, 2, 2)
# plt.imshow(im_norm_t1[:,:,90])
# ax.set_title('T1 normalized')
# ax.get_xaxis().set_visible(False)
# ax.get_yaxis().set_visible(False)


# # plot histogram before and after: 
# fig, axs = plt.subplots(2, 1, constrained_layout=True)
# f1 = axs[0].hist(im_not_norm_t1.flatten(), bins=64, range=(-5,10))
# f2 = axs[1].hist(im_norm_t1.flatten(), bins=64, range=(-1,2))
# axs[0].set_title('Image T1 Original')
# axs[1].set_title('Image T1 Normalized')

