from monai.utils import first
from monai.transforms import (
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandSpatialCropd,
    ScaleIntensityRanged,
    Spacingd,
    Resized,
    ToTensord,
    ScaleIntensityd,
    RandRotated,
    RandZoomd,
    RandGaussianNoised,
    Flipd,
    RandAffined,
)
from monai.data import Dataset, DataLoader

import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import torch
import nibabel as nib
from tqdm import tqdm

def create_transforms(preprocess=True):
    keys = ['image', 'label']
    if preprocess:
        return Compose(
            [
                LoadImaged(keys=keys),
                AddChanneld(keys=keys),
                Spacingd(keys=keys, pixdim=(1.5, 1.5, 2.0), mode=('bilinear', 'nearest')),
                Orientationd(keys=keys, axcodes="RAS"),
                ScaleIntensityRanged(keys=["image"], a_min=-200, a_max=200,b_min=0.0, b_max=1.0, clip=True,),
                RandAffined(keys=keys, prob=0.5, translate_range=10),
                RandRotated(keys=keys, prob=0.5, range_x=10.0),
                RandGaussianNoised(keys='image', prob=0.5),
                ToTensord(keys=keys),
            ]
        )
    else:
        return Compose(
            [
                LoadImaged(keys=keys),
                AddChanneld(keys=keys),
                RandAffined(keys=keys, prob=0.5, translate_range=10),
                RandRotated(keys=keys, prob=0.5, range_x=10.0),
                RandGaussianNoised(keys='image', prob=0.5),
                ToTensord(keys=keys),
            ]
        )

def save_nifti(in_image, in_label, out, index=0):
    vol = np.array(in_image.detach().cpu()[0, :, :, :], dtype=np.float32)
    label = np.array(in_label.detach().cpu()[0, :, :, :], dtype=np.float32)

    vol = nib.Nifti1Image(vol, np.eye(4))
    label = nib.Nifti1Image(label, np.eye(4))

    img_out_path = os.path.join(out, 'Images')
    label_out_path = os.path.join(out, 'Labels')

    if not os.path.exists(img_out_path):
        os.mkdir(img_out_path)
    if not os.path.exists(label_out_path):
        os.mkdir(label_out_path)

    nib.save(vol, os.path.join(img_out_path, f'patient_generated_{index}.nii.gz'))
    nib.save(label, os.path.join(label_out_path, f'patient_generated_{index}.nii.gz'))
    print(f'patient_generated_{index} is saved', end='\r')

def generate_synthetic_data(data_files, output_path, number_runs, preprocess=True):
    transforms = create_transforms(preprocess)
    
    for i in range(number_runs):
        name_folder = 'generated_data_' + str(i)
        os.mkdir(os.path.join(output_path, name_folder))
        output = os.path.join(output_path, name_folder)
        check_ds = Dataset(data=data_files, transform=transforms)
        check_loader = DataLoader(check_ds, batch_size=1)
        
        for index, patient in enumerate(check_loader):
            save_nifti(patient['image'], patient['label'], output, index)
        print(f'step {i} done')
        
def display_patient_slice(data_files, preprocess=True):
    transforms = create_transforms(preprocess)

    keys = ['image', 'label']
    orig_transforms = Compose(
        [
            LoadImaged(keys=keys),
            AddChanneld(keys=keys),
            Spacingd(keys=keys, pixdim=(1.5, 1.5, 2.0), mode=('bilinear', 'nearest')),
            Orientationd(keys=keys, axcodes='RAS'),
            ScaleIntensityRanged(keys='image', a_min=-200, a_max=200, b_min=0.0, b_max=1.0, clip=True),  
            ToTensord(keys=keys),
        ]
    )   
    
    original_ds = Dataset(data=data_files, transform=orig_transforms)
    original_loader = DataLoader(original_ds, batch_size=1)
    original_patient = first(original_loader)
    
    generat_ds = Dataset(data=data_files, transform=transforms)
    generat_loader = DataLoader(generat_ds, batch_size=1)
    generat_patient = first(generat_loader)
    
    slice_idx = 30
    plt.figure("display", (12, 6))
    plt.subplot(1, 2, 1)
    plt.title(f"Original patient slice {slice_idx}")
    plt.imshow(original_patient["image"][0, 0, :, :, slice_idx], cmap="gray")
    plt.subplot(1, 2, 2)
    plt.title(f"Generated patient slice {slice_idx}")
    plt.imshow(generat_patient["image"][0, 0, :, :, slice_idx], cmap="gray")
    plt.show()

data_dir = 'datasets'

file_extension = '*.nii.gz'
train_imgs = sorted(glob(os.path.join(data_dir, 'TrainData', file_extension)))
train_labels = sorted(glob(os.path.join(data_dir, 'TrainLabels', file_extension)))
val_imgs = sorted(glob(os.path.join(data_dir, 'ValData', file_extension)))
val_labels = sorted(glob(os.path.join(data_dir, 'ValLabels', file_extension)))

train_files = [{'image': image_name, 'label': label_name} for image_name, label_name in zip(train_imgs, train_labels)]
val_files = [{'image': image_name, 'label': label_name} for image_name, label_name in zip(val_imgs, val_labels)]

output_path = 'generated_data'
number_runs = 10
use_preprocess = True

generate_synthetic_data(train_files, output_path, number_runs, use_preprocess)
display_patient_slice(train_files, use_preprocess)




