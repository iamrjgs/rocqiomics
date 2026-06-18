from abc import ABC, abstractmethod

class FeatureExtractionEngine(ABC):
    def __init__(self,
                name=None,
                voxel_based=False,
                **kwargs
                ):
        self.name = name
        self.voxel_based = voxel_based
        self.extractor = self.prepare_extractor(**kwargs)

    @abstractmethod
    def preprocess(self, **kwargs):
        pass

    @abstractmethod
    def extract(self, **kwargs):
        pass

    @abstractmethod
    def postprocess(self, **kwargs):
        pass

    @abstractmethod
    def prepare_extractor(self, **kwargs):
        pass