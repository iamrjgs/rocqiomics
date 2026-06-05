import copy
import monai

class AugmentedDataset(monai.data.Dataset):
    """
    Dataset that decouples loading and augmentations from preprocessing transforms.
    
    In the standard monai.data.Dataset, preprocessing and augmentation are typically
    achieved by combining preprocessing and augmentation transforms into a single Compose
    object fed to the `transforms` keyword. While this increases the diversity of the dataset,
    it doesn't increase its size; original images are replaced by the augmented copy.

    This class preserves the original images while also generating augmented copies, thereby
    increasing both the diversity and size of the dataset.

    In this class:
    
    1) Augmentation transforms increase the size of the dataset.
       For each augmentation transform, a new, transformed copy of each image is added 
       to the dataset. Augmentation transforms are applied prior to preprocessing transforms.
    
    2) Preprocessing transforms are applied to all images, both the originals and augmented copies.
       They do not generate any additional images.

    To recover standard monai.data.Dataset behavior, simply leave augmentations list empty,
    set load_transform as the loading transform and put all other transforms in preprocessing.
    """

    def __init__(self, data, load_transform, preprocessing=None, augmentations=None):
        self.data = data
        self.load_transform = load_transform
        self.preprocessing = preprocessing
        self.augmentations = augmentations or []
        self.num_augmentations = len(augmentations)
    
    def __len__(self):
        return len(self.data) * (self.num_augmentations + 1)

    def __getitem__(self, idx):
        image_index = idx // (self.num_augmentations + 1)
        aug_index = idx % (self.num_augmentations + 1)

        base_data = self.load_transform(self.data[image_index])

        # Apply augmentation if needed
        if aug_index > 0:
            loaded_data = copy.deepcopy(base_data)
            loaded_data = self.augmentations[aug_index - 1](loaded_data)
        else:
            loaded_data = base_data    
    
        # Add augmentation index to metadata
        if isinstance(loaded_data, dict) and 'metadata' in loaded_data:
            loaded_data = dict(loaded_data)
            loaded_data['metadata'] = dict(loaded_data['metadata'])
            loaded_data['metadata']['augmentation'] = aug_index

        # Preprocess image/mask data
        if self.preprocessing is not None:
            loaded_data = self.preprocessing(loaded_data)
        
        return loaded_data