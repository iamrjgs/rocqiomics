# rocqiomics

**A Monai-inspired radiomics framework tailored to habitat imaging.**

**`rocqiomics`** wraps validated radiomics engines in a flexible, MONAI-inspired interface and features tooling designed with augmentation, perturbation analysis, and voxel-wise modeling workflows in mind.

---

## ⚡ Key Features

- ✅ Radiomics pipeline with deep-learning–inspired **data dictionary API**
- ✅ Native **MONAI transform** [1] support
- ✅ **Voxel-wise** feature extraction
- ✅ Built for **Habitat Radiomics**
- ✅ Multi-engine backend (**PyRadiomics** [2], **fastrad** [3])

---

## Overview

**`rocqiomics`** provides:

### 🔹 Radiomics from validated engines
- Supports **PyRadiomics** and **fastrad**
- Maintains IBSI-compliant feature extraction
- Adds a clean, modern Python interface

### 🔹 MONAI-native pipelines
- Direct support for Monai dictionary transforms (e.g. ScaleIntensityd, Spacingd)
- Enables complex preprocessing and augmentation workflows
- Simplifies perturbation-based radiomics [4]

### 🔹 Habitat Radiomics support
- Designed for voxel-wise feature extraction
- Enables clustering into spatially distinct **image habitats** [5]
---

## Core Components

The library is built around three main classes:

---

### 🔹 `Rocqiomics`

**Primary feature extraction interface**

A flexible wrapper around radiomics engines that:

- Handles image loading, preprocessing, and augmentation
- Supports both **tabular features** and **voxel-wise feature maps**
- Accepts MONAI transforms directly
- Performs input validation and robust extraction

#### Key capabilities
- Engine abstraction (`pyradiomics`, `fastrad`)
- MONAI transform integration
- Automated dataset handling
- Feature map extraction (`voxel_based=True`)
- Flexible saving and metadata handling

---

### 🔹 `HabitatGenerator`

**Voxel clustering engine for multi-channel imaging**

Clusters voxels based on feature vectors across channels:

- Radiomics feature maps  
- Multiparametric MRI sequences  
- Any multi-channel volumetric data  

#### Supported algorithms
- MiniBatch KMeans
- Gaussian Mixture Models (GMM)
- Birch clustering

#### Key features
- Channel-aware clustering
- Optional feature normalization
- Batch-wise processing for large 3D volumes


### 🔹 `RadiomicsHabitatGenerator`

**End-to-end habitat radiomics pipeline**

Combines:

- `Rocqiomics` → voxel-wise feature extraction  
- `HabitatGenerator` → voxel clustering  

Enables segmentation of images into biologically meaningful **radiomic habitats**.

#### Key features
- Automated feature map extraction + clustering
- Augmentation-aware pipelines
- Designed for large-scale voxel-based workflows

## Installation

```bash
pip install rocqiomics

git clone https://github.com/iamrjgs/rocqiomics.git
cd rocqiomics
pip install -e .
```

## Dependencies
# Core

monai
torch
SimpleITK
numpy
pydicom
itk

# Optional engines

pyradiomics
fastrad

## References

[1] The MONAI Consortium. (2020). Project MONAI. Zenodo. https://doi.org/10.5281/zenodo.4323059

[2] van Griethuysen, J. J. M., Fedorov, A., Parmar, C., Hosny, A., Aucoin, N., Narayan, V., Beets-Tan, R. G. H., Fillion-Robin, J. C., Pieper, S., Aerts, H. J. W. L. (2017). Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Research, 77(21), e104–e107. https://doi.org/10.1158/0008-5472.CAN-17-0339 | https://github.com/AIM-Harvard/pyradiomics/tree/master

[3] Sánchez-Femat, Erika and Celaya-Padilla, José-María and Galvan-Tejada, Carlos Eric, fastrad: Complete, IBSI-Validated GPU Acceleration of the Full PyRadiomics Feature Set. Available at SSRN: https://ssrn.com/abstract=6436486 or http://dx.doi.org/10.2139/ssrn.6436486 | https://github.com/helloerikaaa/fastrad

[4] Zwanenburg, A., Leger, S., Agolli, L. et al. Assessing robustness of radiomic features by image perturbation. Sci Rep 9, 614 (2019). https://doi.org/10.1038/s41598-018-36938-4

[5] Prior O, Macarro C, Navarro V, Monreal C, Ligero M, Garcia-Ruiz A, Serna G, Simonetti S, Braña I, Vieito M, Escobar M, Capdevila J, Byrne AT, Dienstmann R, Toledo R, Nuciforo P, Garralda E, Grussu F, Bernatowicz K, Perez-Lopez R. Identification of precise 3D CT radiomics for habitat computation by machine learning in cancer. Radiology: Artificial Intelligence. 2024;6(2):e230118. https://doi.org/10.1148/ryai.230118