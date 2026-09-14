"""M3ED packed adapter (official train/valid only; test is sealed).

Class order is taken verbatim from the packed audit artifact
``M3ED/packed/data_audit.json`` -> ``label_field = EmoAnnotation.final_main_emo``
with ``labels`` listed in packed-id order:

    0 Happy, 1 Neutral, 2 Sad, 3 Disgust, 4 Anger, 5 Fear, 6 Surprise

The previously hard-coded tuple (anger, joy, sadness, neutral, fear, surprise,
disgust) matched neither this order nor the official M3ED taxonomy, so every
per-class F1 row would have carried the wrong class name.
"""
from .packed_dataset import PackedDataset

M3ED_LABELS = ("Happy", "Neutral", "Sad", "Disgust", "Anger", "Fear", "Surprise")


class M3EDPackedDataset(PackedDataset):
    def __init__(self, path):
        super().__init__(path, "classification")
