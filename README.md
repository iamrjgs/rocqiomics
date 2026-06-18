rocqiomics
A modern radiomics framework bridging the gap between classical radiomics and deep learning pipelines.
rocqiomics wraps validated radiomics engines in a flexible, MONAI-inspired interface, with tooling designed around augmentation, perturbation analysis, and voxel-wise modeling workflows.

🚀 Overview

Wrapping established radiomics engines

Supports PyRadiomics and fastrad
Maintains validated, IBSI-compliant feature extraction
Adds a modern, Pythonic API

Integrating natively with MONAI transforms

Direct support for monai.transforms
Enables complex preprocessing and augmentation pipelines
Simplifies perturbation-based radiomics workflows

Supporting Habitat Radiomics

Designed for voxel-wise feature extraction and clustering
Enables generation of spatially distinct “habitats” within images


🏗️ Core Components
The package is built around three main classes:

🔹 Rocqiomics
The primary feature extraction interface.
A flexible wrapper around radiomics engines that:

Handles image loading, preprocessing, and augmentation
Supports both tabular features and voxel-wise feature maps
Accepts MONAI transforms directly
Performs input validation and robust extraction

Key capabilities:

Engine abstraction (pyradiomics, fastrad)
MONAI transform integration
Automated dataset handling
Feature map extraction (voxel_based=True)
Flexible saving and metadata handling

🔹 HabitatGenerator
Voxel clustering engine for multi-channel imaging data.
Clusters voxels based on feature vectors defined across multiple channels:

Radiomics feature maps
Multiparametric MRI sequences
Any multi-channel volumetric data

Supported algorithms:

Mini-batch KMeans
Gaussian Mixture Models
Birch clustering

Key features:

Channel-aware clustering
Optional normalization across features
Batch-wise processing for large 3D datasets

🔹 RadiomicsHabitatGenerator
End-to-end pipeline for habitat radiomics.
Combines:

Rocqiomics → generates voxel-wise feature maps
HabitatGenerator → clusters voxels into habitats

This enables segmentation of images into biologically meaningful regions based on radiomic patterns.
Key features:

Automated feature map extraction + clustering
Augmentation-aware pipelines