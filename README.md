# rocqiomics

![Habitat Map](assets/habitat_example.png)

**A Deep Learning-inspired Radiomics Framework Tailored For Habitat Imaging.**

**`rocqiomics`** provides a MONAI-inspired interface to IBSI-compliant radiomics engines, and features tooling designed with voxel-wise "habitat radiomics" workflows in mind.

Habitat Radiomics extracts voxel-wise feature maps to cluster voxels into several 'habitats' that contain distinct textures. [1]

## Key Features

- Radiomics pipeline with modern, deep-learning–inspired **data dictionary API**
- Classes for **Habitat Radiomics** - voxel-wise clustering based on multi-channel feature maps 
- Native support for **MONAI dictionary transforms** for preprocessing and augmentation [2]
- Uses validated radiomics engine **PyRadiomics** [3] or its GPU-based alternative **fastrad** [4]
---

## Core Components

The library has three main classes.

### 🔹 `Rocqiomics` - Primary feature extraction interface

- Handles image loading, preprocessing, and augmentation with Monai Transforms
- Supports both **standard numerical features** and **voxel-wise feature maps**
- Performs input validation and robust extraction
- Optionally handles flexible results saving and metadata handling
- Allows seamless portability of existing Pyradiomics workflows with a user-friendly interface


#### Usage

```
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
        'image' : # ../path/to/image_1,
        'mask' : # ../path/to/mask_1,
        'metadata' : {
            'modality' : 'CT'
        }
    },
    {
        'case_id' : 'id_2',
        'image' : # ../path/to/image_2,
        'mask' : # ../path/to/mask_2,
        'metadata' : {
            'modality' : 'CT'
        }
    },
]

### Define extractor object with engine and `Monai.transform` preprocessing and augmentation steps ###
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
    filter_types=['Original', 'Wavelet'],
    bin_width=10.0,
    voxel_based=False,
    engine='pyradiomics'
)

"""
Simply run on your data dicts. Output is:
 - Pandas DataFrame of features (voxel_based=False)
 - Dictionary of feature map SimpleITK images (voxel_based=True)
"""
results = extractor.run_pipeline(data_dicts)
```

### 🔹 `RadiomicsHabitatGenerator` - **End-to-end habitat radiomics pipeline**

Combines:

- `Rocqiomics` → voxel-wise feature extraction  
- `HabitatGenerator` → voxel clustering  

#### Key features
- Automated feature map extraction + voxel-wise clustering
- Generates maps dynamically for memory efficiency
- Augmentation-aware pipeline
- Can use clustering algorithm implemented for HabitatGenerator

#### Usage

```
import rocqiomics as rq
from rocqiomics.transforms import N4ITKBiasFieldCorrection

from monai.transforms import (
    Compose,
    NormalizeIntensityd,
    Spacingd,
    Rotated
)

### Define data dicts same as for rq.Rocqiomics feature extraction ###
data_dicts = [
    {
        'case_id' : 'id_1',
        'image' : # ../path/to/image_1,
        'mask' : # ../path/to/mask_1,
        'metadata' : {
            'modality' : 'CT'
        }
    },
    {
        'case_id' : 'id_2',
        'image' : # ../path/to/image_2,
        'mask' : # ../path/to/mask_2,
        'metadata' : {
            'modality' : 'CT'
        }
    },
]

### Define generator which automatically extracts maps and fits or predicts habitat clusters ####
radhab = rq.RadiomicsHabitatGenerator(
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
    features=['Mean', 'Autocorrelation', 'Entropy],
    filter_types=['Original', 'Square'],
    algorithm='kmeans',
    n_clusters=4,
    batch_size=25,
    save_vector_dirpath=../path/for/temporary/4d_vectors_storage/before/clustering, # required
)

predictions, result_ddicts = radhab.fit_predict(data_dicts)
```


---

### 🔹 `HabitatGenerator` - **Voxel clustering engine for multi-channel imaging**

Clusters voxels based on 4D feature vectors across channels:

- Radiomics feature maps  
- Multiparametric MRI images 
- Any multi-channel volumetric data  

#### Supported algorithms
- MiniBatch KMeans
- Gaussian Mixture Models (GMM)
- Birch clustering

#### Key features
- Channel-aware clustering
- Optional feature normalization
- Batch-wise processing for large 3D volumes

#### Usage

```

adc_img = sitk.ReadImage(../path/to/ADCmap.nrrd)
t1map_img = sitk.ReadImage(../path/to/T1map.nrrd)
mask_img = sitk.ReadImage(../path/to/mask.seg.nrrd) # Optional

vector_img = sitk.Compose([adc_img, timap_img])

data_dicts = [
    {
        'image' : vector_img_1,
        'mask : mask_img_1 # optional
    }
]

### Define habitat generator with channel names and number of clusters to fit
habitat_generator = HabitatGenerator(
    channels=["ADC", "T1map"],
    n_clusters=4,
    algorithm='gmm'
)

prediction = gmm.fit_predict(data=data_dicts)

for r in res:
    fig, ax = plt.subplots(1,, figsize=(5,5))
    ax.imshow(r[2,:,:], cmap='turbo')
```


## Installation

```bash
git clone https://github.com/iamrjgs/rocqiomics.git
cd rocqiomics
pip install -e .
```

## References

[1] Prior O, Macarro C, Navarro V, Monreal C, Ligero M, Garcia-Ruiz A, Serna G, Simonetti S, Braña I, Vieito M, Escobar M, Capdevila J, Byrne AT, Dienstmann R, Toledo R, Nuciforo P, Garralda E, Grussu F, Bernatowicz K, Perez-Lopez R. Identification of precise 3D CT radiomics for habitat computation by machine learning in cancer. Radiology: Artificial Intelligence. 2024;6(2):e230118. https://doi.org/10.1148/ryai.230118

[2] The MONAI Consortium. (2020). Project MONAI. Zenodo. https://doi.org/10.5281/zenodo.4323059

[3] van Griethuysen, J. J. M., Fedorov, A., Parmar, C., Hosny, A., Aucoin, N., Narayan, V., Beets-Tan, R. G. H., Fillion-Robin, J. C., Pieper, S., Aerts, H. J. W. L. (2017). Computational Radiomics System to Decode the Radiographic Phenotype. Cancer Research, 77(21), e104–e107. https://doi.org/10.1158/0008-5472.CAN-17-0339 | https://github.com/AIM-Harvard/pyradiomics/tree/master

[4] Sánchez-Femat, Erika and Celaya-Padilla, José-María and Galvan-Tejada, Carlos Eric, fastrad: Complete, IBSI-Validated GPU Acceleration of the Full PyRadiomics Feature Set. Available at SSRN: https://ssrn.com/abstract=6436486 or http://dx.doi.org/10.2139/ssrn.6436486 | https://github.com/helloerikaaa/fastrad

[5] Zwanenburg, A., Leger, S., Agolli, L. et al. Assessing robustness of radiomic features by image perturbation. Sci Rep 9, 614 (2019). https://doi.org/10.1038/s41598-018-36938-4

