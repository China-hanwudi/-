"""M3ED packed adapter (official train/valid only; test is sealed)."""
from .packed_dataset import PackedDataset

M3ED_LABELS = ("anger", "joy", "sadness", "neutral", "fear", "surprise", "disgust")

class M3EDPackedDataset(PackedDataset):
    def __init__(self, path):
        super().__init__(path, "classification")
