# 3D Volumes Augmentation for Segmentation using MONAI

This repository provides implementation for augmenting 3D medical image datasets using MONAI, an open-source framework built on PyTorch. The focus is on generating synthetic data to improve the performance and robustness of segmentation models.

## Features

- Apply various affine transformations such as flipping, rotation, translation, and Gaussian noise.
- Supports both online (during training) and offline (before training) data augmentation.
- Save augmented data in NIfTI format for further use.
- Visualize original and augmented data slices.


