""" Histogram normalization: AML course code """
import nibabel as nib
import os 
import numpy as np
from scipy.interpolate import interp1d
import numpy as np


def calc_landmarks_from_percentiles(image, percentile_list):
    landmarks = np.percentile(image, percentile_list)
    return landmarks

def create_trained_mapping(images, percentile_list, scale_min, scale_max, p_min, p_max):
    average_mapped_landmarks = np.zeros(len(percentile_list),)
    for image in images:
        # 2. Calculate the landmarks for each image
        landmarks = calc_landmarks_from_percentiles(image, percentile_list)
        # 3.  Calculate the image intensities corresponding to your chosen minimum and maximum percentiles
        # These will anchor the mapping
        intensity_min = np.percentile(image, p_min)
        intensity_max = np.percentile(image, p_max)
        # 4. Create mapping by interpolating between the image's minimum + max values and those of the standard scale
        mapping = interp1d([intensity_min, intensity_max], [scale_min, scale_max], fill_value='extrapolate')
        # 5. Map the image landmarks to these values
        mapped_landmarks = np.array(mapping(landmarks))
        # Sum the mapped landmarks iteratively
        average_mapped_landmarks += mapped_landmarks
    # Average the summed landmarks
    average_mapped_landmarks = average_mapped_landmarks / len(images)
    return average_mapped_landmarks

def standardise_image(image, percentile_list, average_mapped_landmarks):
    landmarks = calc_landmarks_from_percentiles(image, percentile_list)
    mapping = interp1d(landmarks, average_mapped_landmarks, fill_value='extrapolate')
    new_image = mapping(image)
    return new_image


# Apply histogram normalization method 

# =============================================================================
# T1 patients
# =============================================================================

dir_t1 = 'E:/Master/PET-MR/PET-MR-Controls/Affine_remasked/controls/t1/'
names_t1 = os.listdir(dir_t1)

dataset_t1 = []
affine_t1 =[]

for i in range(len(names_t1)):
    name = names_t1[i]
    image = nib.load(dir_t1 + name)
    affine = image.affine
    image = image.get_fdata()+(1e-05)
        
    dataset_t1.append(image)
    affine_t1.append(affine)
    
    
       
percentile_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
trained_mapping = create_trained_mapping(images=dataset_t1,
                                          percentile_list = percentile_list,
                                          scale_min=0, scale_max=1,
                                          p_min=5, p_max=95)


for i, image in enumerate(dataset_t1):
    name = names_t1[i]
    norm_image = standardise_image(image, percentile_list, trained_mapping)
    
    nib.save(nib.Nifti1Image(norm_image, affine_t1[i]), os.path.join('E:/Master/PET-MR/PET-MR-Controls/Norm_hist_aml_data/controls/t1/', name))
       
    
# # =============================================================================
# # # PET patients 
# # =============================================================================

dir_pet = 'E:/Master/PET-MR/PET-MR-Controls/Affine_remasked/controls/pet/'
names_pet = os.listdir(dir_pet)

dataset_pet = []
affine_pet =[]

for i in range(len(names_pet)):
    name = names_pet[i]
    image = nib.load(dir_pet + name)
    affine = image.affine
    image = image.get_fdata()+(1e-05)
    
    dataset_pet.append(image)
    affine_pet.append(affine)


    
percentile_list = [10, 20, 30, 40, 50, 60, 70, 80, 90]
trained_mapping = create_trained_mapping(images=dataset_pet,
                                          percentile_list = percentile_list,
                                          scale_min=0, scale_max=1,
                                          p_min=5, p_max=95)


for i, image in enumerate(dataset_pet):
    name = names_pet[i]
    norm_image = standardise_image(image, percentile_list, trained_mapping)
    
    nib.save(nib.Nifti1Image(norm_image, affine_pet[i]), os.path.join('E:/Master/PET-MR/PET-MR-Controls/Norm_hist_aml_data/controls/pet/', name))
    