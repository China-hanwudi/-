"""CMU-MOSEI packed adapter for scalar sentiment regression."""
from .packed_dataset import PackedDataset

class MOSEIPackedDataset(PackedDataset):
    def __init__(self, path):
        super().__init__(path, "regression")
