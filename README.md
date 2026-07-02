# rocqiomics

**A Deep Learning-inspired Radiomics Framework Tailored to Voxel-wise Habitat Imaging.**

**`rocqiomics`** provides a MONAI-inspired interface to IBSI-compliant radiomics engines, and features tooling designed with voxel-wise modeling workflows in mind.

---

## Key Features

- Radiomics pipeline with modern, deep-learning–inspired **data dictionary API**
- Classes for **Habitat Radiomics** - voxel-wise clustering based on radiomics feature maps [2]
- Native support for **MONAI dictionary transforms** for preprocessing and augmentation [1]
- Validated radiomics engine **PyRadiomics** [3] or its GPU-based alternative **fastrad** [4]
---

## Core Components

The library has three main classes:

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

#### Usage

```
### Imports ###
import rocqiomics as rq
from rocqiomics.transforms import N4ITKBiasFieldCorrection

from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    Spacingd,
    Rotated
)

### Define data dicts with your case data (metadata optional) ###
data_dicts = [
    {
        'case_id' : 'id_1',
        'image' : # path/to/image_1,
        'mask' : # path/to/mask_1,
        'metadata' : {
            'modality' : 'CT'
        }
    },
    {
        'case_id' : 'id_2',
        'image' : # path/to/image_2,
        'mask' : # path/to/mask_2,
        'metadata' : {
            'modality' : 'CT'
        }
    },
]

### Define extractor object with desired engine and desired preprocessing and augmentation steps ###
extractor = rq.Rocqiomics(
    preprocessing=Compose([
        N4ITKBiasFieldCorrection(image_key='image', mask_key='mask', max_iterations=20),
        NormalizeIntensityd(keys=['image']),
        ScaleIntensityd(keys=['image'], factor=99.0, minv=None, maxv=None),
        Spacingd(keys=['image', 'mask'], pixdim=(1.0, 1.0, 1.0), mode=[3, 'nearest']),
    ]),
    augmentations=[
        Rotated(keys=['image', 'mask'], angle=0.1, mode=['bilinear', 'nearest]),
    ],
    bin_width=10.0,
    voxel_based=False,
    engine='pyradiomics'
)

"""
Simply run on your data dicts.
Output: 
Pandas DataFrame of features (voxel_based=False)
Dictionary of feature map SimpleITK images (voxel_based=True)
"""
results = extractor.run_pipeline(data_dicts)
```

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

## References

[1] The MONAI Consortium. (2020). Project MONAI. Zenodo. https://doi.org/10.5281/zenodo.4323059

[2] Prior O, Macarro C, Navarro V, Monreal C, Ligero M, Garcia-Ruiz A, Serna G, Simonetti S, Braña I, Vieito M, Escobar M, Capdevila J, Byrne AT, Dienstmann R, Toledo R, Nuciforo P, Garralda E, Grussu F, Bernatowicz K, Perez-Lopez R. Identification of precise 3D CT radiomics for habitat computation by machine learning in cancer. Radiology: Artificial Intelligence. 2024;6(2):e230118. https://doi.org/10.1148/ryai.230118

[3] van Griethuysen, J. J. M., Fedorov, A., Parmar, C., Hosny, A., Aucoin, N., Narayan, V., Beets-Tan, R. G. H., Fillion-Robin, J. C., Pieper, S., Aerts, H. J. W. L. (2017). Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Research, 77(21), e104–e107. https://doi.org/10.1158/0008-5472.CAN-17-0339 | https://github.com/AIM-Harvard/pyradiomics/tree/master

[4] Sánchez-Femat, Erika and Celaya-Padilla, José-María and Galvan-Tejada, Carlos Eric, fastrad: Complete, IBSI-Validated GPU Acceleration of the Full PyRadiomics Feature Set. Available at SSRN: https://ssrn.com/abstract=6436486 or http://dx.doi.org/10.2139/ssrn.6436486 | https://github.com/helloerikaaa/fastrad

[5] Zwanenburg, A., Leger, S., Agolli, L. et al. Assessing robustness of radiomic features by image perturbation. Sci Rep 9, 614 (2019). https://doi.org/10.1038/s41598-018-36938-4

