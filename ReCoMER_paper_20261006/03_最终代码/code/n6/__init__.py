"""model6 (UGF: Utility-Gated Fusion) package.

All modalities are utterance-level pooled vectors; no temporal sequences.
Only packed train.pt / valid.pt are ever opened (see n6.data.open_split);
the sealed test split is never referenced anywhere in this package.
"""

__version__ = "6.1.0"
