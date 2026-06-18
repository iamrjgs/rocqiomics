# 🧠 rocqiomics

**A modern radiomics framework bridging classical radiomics and deep learning pipelines.**

`rocqiomics` wraps validated radiomics engines in a flexible, MONAI-inspired interface, with tooling designed for augmentation, perturbation analysis, and voxel-wise modeling workflows.

---

## ⚡ Key Features

- ✅ Radiomics pipeline with deep-learning–inspired **data dictionary API**
- ✅ Native **MONAI transform** support
- ✅ **Voxel-wise feature extraction**
- ✅ Built for **Habitat Radiomics**
- ✅ Multi-engine backend (**PyRadiomics**, **fastrad**)

---

## 🚀 Overview

`rocqiomics` provides:

### 🔬 Radiomics from validated engines
- Supports **PyRadiomics** and **fastrad**
- Maintains IBSI-compliant feature extraction
- Adds a clean, modern Python interface

### 🔄 MONAI-native pipelines
- Direct support for `monai.transforms`
- Enables complex preprocessing and augmentation workflows
- Simplifies perturbation-based radiomics

### 🧩 Habitat Radiomics support
- Designed for voxel-wise feature extraction
- Enables clustering into spatially distinct **image habitats**

---

## 🏗️ Core Components

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

---

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

---

## 🔄 Workflow

Typical pipeline:

Image + Mask
↓
MONAI preprocessing / augmentations

↓

Rocqiomics (feature extraction)

↓

Voxel-wise feature maps

↓

HabitatGenerator (clustering)

↓

Habitat maps


## 🧪 Design Philosophy

- **Composable** → Built around MONAI-style transforms  
- **Engine-agnostic** → Wraps validated radiomics engines  
- **Voxel-first** → Designed for spatial feature modeling  
- **Scalable** → Supports batch and disk-based workflows  

---

## 📦 Installation

```bash
pip install rocqiomics

git clone https://github.com/yourusername/rocqiomics.git
cd rocqiomics
pip install -e .

## 📚 Dependencies
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