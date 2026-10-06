"""DatasetSpec — single source of truth for per-dataset task geometry.

A01 / P0-02 (framework 2026-09-23): previously ``num_classes`` and class
names were hardcoded to the M3ED 7-class geometry in several places
(``train.py`` cfg construction, ``metrics.M3ED_CLASS_NAMES`` fallback),
so IEMOCAP 4-class runs loaded with wrong per-class names and, before
2026-09-23, refused to load at all.  All dataset facts now live here:
``dataset_id, task, num_classes, class_names, label_map, feature_dims,
split_protocol`` (framework 01 P0-02).  Trainers write the resolved spec
into the checkpoint cfg; evaluators re-resolve from the checkpoint's
saved values, falling back to the pack path when a legacy checkpoint
carries no dataset_id (explicit legacy adaptation — never a silent
label guess).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple


@dataclass(frozen=True)
class DatasetSpec:
    dataset_id: str
    task: str                       # "cls" | "reg"
    num_classes: int
    class_names: Tuple[str, ...]    # label order, empty for reg
    label_map: Dict[str, int]       # name -> index, empty for reg
    feature_dims: Optional[Dict[str, int]]  # pinned dims, None = recorded not enforced
    split_protocol: str


def _cls(dataset_id: str, names, split_protocol: str) -> DatasetSpec:
    return DatasetSpec(
        dataset_id=dataset_id, task="cls", num_classes=len(names),
        class_names=tuple(names),
        label_map={n: i for i, n in enumerate(names)},
        feature_dims=None, split_protocol=split_protocol)


_SPECS = (
    # Label orders are from the pack build manifests, not from convention:
    #   M3ED:    data/M3ED/packed/data_audit.json ("labels" list, official
    #            EmoAnnotation.final_main_emo)
    #   MELD:    train.json manifest label->emotion cross-tab (verified
    #            2026-09-23: 0=anger 1=disgust 2=fear 3=joy 4=neutral
    #            5=sadness 6=surprise)
    #   IEMOCAP: data/IEMOCAP/build_iemocap_skeleton.py LABEL_ORDER
    _cls("m3ed",
         ["Happy", "Neutral", "Sad", "Disgust", "Anger", "Fear", "Surprise"],
         "leave-series-out; valid = 7 held-out series (disclosed: small "
         "series count)"),
    _cls("meld",
         ["anger", "disgust", "fear", "joy", "neutral", "sadness",
          "surprise"],
         "official utterance split; pack dialogue ids are per-split local "
         "(dialogue/episode cross-split disjointness not verifiable from "
         "the pack alone)"),
    _cls("meld_frames",
         ["anger", "disgust", "fear", "joy", "neutral", "sadness",
          "surprise"],
         "MELD + frame-level CLIP Vf [N,8,512]; same split protocol as "
         "meld"),
    _cls("iemocap",
         ["angry", "happy", "sad", "neutral"],
         "session-grouped; valid = Ses03 (F/M are subgroups of one "
         "session, not independent sessions); Session2 permanently "
         "sealed after two one-shot ceremonies"),
    DatasetSpec("cmu_mosi", "reg", 1, (), {}, None,
                "official split; confirm ceremony one-shot"),
    DatasetSpec("mosei_full", "reg", 1, (), {}, None,
                "full-protocol regression; single round-2 evaluation"),
    DatasetSpec("chsims_full", "reg", 1, (), {}, None,
                "continuous-regression protocol; 577/1034 rows missing "
                "audio+video disclosed"),
)

_SPECS_BY_ID = {s.dataset_id: s for s in _SPECS}

# Pack-path directory tokens -> dataset_id.  Matches on any path component,
# most specific first (a MELD_frames path also contains "meld" text).
_PATH_TOKENS = (
    ("meld_frames", "meld_frames"),
    ("meld", "meld"),
    ("m3ed", "m3ed"),
    ("iemocap", "iemocap"),
    ("cmu-mosi", "cmu_mosi"),
    ("cmu_mosi", "cmu_mosi"),
    ("mosei", "mosei_full"),
    ("ch-sims", "chsims_full"),
    ("chsims", "chsims_full"),
)


def get_spec(dataset_id: str) -> DatasetSpec:
    key = dataset_id.strip().lower()
    if key in _SPECS_BY_ID:
        return _SPECS_BY_ID[key]
    raise ValueError(
        "unknown dataset_id %r (known: %s).  Add the dataset here with its "
        "pack-manifest label order — never guess labels."
        % (dataset_id, sorted(_SPECS_BY_ID)))


def resolve_dataset_id(pack_path) -> str:
    """Resolve the dataset from a pack path's directory names."""
    parts = [p.lower() for p in Path(str(pack_path)).parts]
    for token, ds_id in _PATH_TOKENS:
        if any(token in p for p in parts):
            return ds_id
    raise ValueError(
        "cannot resolve dataset from pack path %r; pass the dataset "
        "explicitly or extend _PATH_TOKENS" % (pack_path,))


def spec_for_pack(pack_path) -> DatasetSpec:
    return get_spec(resolve_dataset_id(pack_path))


def class_names_for(cfg, pack_path=None) -> Optional[list]:
    """Class names to use when reporting a checkpoint's metrics.

    Order of authority (most specific first):
    1. ``cfg.class_names`` saved in the checkpoint (post-A01 runs);
    2. the pack path's DatasetSpec (evaluation target always wins over
       legacy checkpoint metadata);
    3. None -> callers keep their legacy behaviour (documented fallback;
       pre-A01 reports stay byte-identical and are covered by errata).
    """
    saved = getattr(cfg, "class_names", None)
    if saved:
        return [str(x) for x in saved]
    if pack_path is not None:
        try:
            spec = spec_for_pack(pack_path)
            if spec.class_names:
                return list(spec.class_names)
        except ValueError:
            return None
    return None
