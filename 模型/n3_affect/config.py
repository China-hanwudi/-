"""Trainable N3 configuration aligned with the repo freeze protocol."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


DEFAULT_LABELS = (
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise",
)

ALLOWED_TEXT_TOWERS = frozenset(
    {"composer_n3", "qwen3_omni_30b_a3b", "emoberta_base", "xlm_roberta_large"}
)
# Main-line omni LLM (Apache-2.0). Do NOT vendor weights into git.
DEFAULT_HF_TEXT_MODEL = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
_PKG_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LOCAL_EMOBERTA = _PKG_ROOT / "artifacts" / "pretrained" / "emoberta-base"
DEFAULT_LOCAL_QWEN = _PKG_ROOT / "artifacts" / "pretrained" / "qwen3-omni-30b-a3b-instruct"
LOCAL_OMNI_PATH_FILE = _PKG_ROOT / "local_omni_path.txt"


def _read_local_omni_override() -> Path | None:
    """Optional one-line absolute path in 模型/local_omni_path.txt."""
    if not LOCAL_OMNI_PATH_FILE.is_file():
        return None
    for line in LOCAL_OMNI_PATH_FILE.read_text(encoding="utf-8").splitlines():
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        path = Path(text)
        if path.is_dir():
            return path
    return None



@dataclass
class N3TrainConfig:
    """Hyperparameters for :class:`~n3_affect.model.N3EmotionModel`."""

    text_dim: int = 2048
    audio_dim: int = 1536
    video_dim: int = 768
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 384
    num_classes: int = 7
    dropout: float = 0.1
    parameter_budget: int = 2_000_000
    relation_rank: int = 32
    gate_hidden: int = 64
    # Safe local default for unit tests; main JSON pins qwen3_omni_30b_a3b.
    text_tower: str = "composer_n3"
    hf_text_model_id: str = DEFAULT_HF_TEXT_MODEL
    hf_text_local_path: str = str(DEFAULT_LOCAL_QWEN)
    emotion_label_order: tuple[str, ...] = field(default_factory=lambda: DEFAULT_LABELS)
    lr: float = 3e-4
    weight_decay: float = 1e-2
    batch_size: int = 8
    max_epochs: int = 5
    grad_clip: float = 1.0
    emotion_loss_weight: float = 1.0
    utility_loss_weight: float = 0.2
    vad_loss_weight: float = 0.1
    mix_tau: float = 1.0
    mix_kl_weight: float = 0.0
    mix_peak_weight: float = 0.05
    mix_peak_cap: float = 0.40
    seed: int = 17

    def validate(self) -> None:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.num_classes != 7:
            raise ValueError("N3 train config currently requires 7 emotion classes")
        if self.text_tower not in ALLOWED_TEXT_TOWERS:
            raise ValueError(f"unknown text_tower: {self.text_tower}")
        if len(self.emotion_label_order) != self.num_classes:
            raise ValueError("emotion_label_order length must equal num_classes")

    def resolved_text_model_source(self) -> str:
        """Prefer local snapshot when present, else Hugging Face id."""
        local = Path(self.hf_text_local_path)
        if self.text_tower == "emoberta_base":
            candidates = [local, DEFAULT_LOCAL_EMOBERTA]
        elif self.text_tower == "qwen3_omni_30b_a3b":
            override = _read_local_omni_override()
            candidates = [c for c in [override, local, DEFAULT_LOCAL_QWEN] if c is not None]
        else:
            candidates = [local]
        for path in candidates:
            if not path.is_dir():
                continue
            has_weights = (
                any(path.glob("*.bin"))
                or any(path.glob("*.safetensors"))
                or any(path.glob("model*.safetensors"))
                or (path / "model.safetensors.index.json").exists()
            )
            if has_weights:
                return str(path)
        return self.hf_text_model_id

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["emotion_label_order"] = list(self.emotion_label_order)
        return data

    @classmethod
    def from_json(cls, path: str | Path) -> "N3TrainConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        arch = raw.get("architecture", {})
        dims = raw.get("input_dims", {})
        train = raw.get("training", {})
        weights = train.get("loss_weights", {})
        llm = raw.get("builtin_llm", {})
        main = llm.get("main_text_tower") or llm.get("recommended_text_tower") or {}
        branch = llm.get("branch_text_towers") or {}
        optional = llm.get("optional_text_tower") or main
        default_mode = str(llm.get("default_mode", "qwen3_omni_30b_a3b"))
        if default_mode in ALLOWED_TEXT_TOWERS:
            text_tower = default_mode
        elif main.get("tower_key") in ALLOWED_TEXT_TOWERS:
            text_tower = str(main["tower_key"])
        elif optional.get("tower_key") in ALLOWED_TEXT_TOWERS:
            text_tower = str(optional["tower_key"])
        else:
            text_tower = "qwen3_omni_30b_a3b"
        source_meta = main if text_tower == main.get("tower_key") else optional
        if text_tower == "emoberta_base" and "emoberta_base" in branch:
            source_meta = branch["emoberta_base"]
        local = source_meta.get("local_path")
        if local and not Path(str(local)).is_absolute():
            local_path = str((_PKG_ROOT / local).resolve())
        elif local:
            local_path = str(local)
        elif text_tower == "emoberta_base":
            local_path = str(DEFAULT_LOCAL_EMOBERTA)
        else:
            local_path = str(DEFAULT_LOCAL_QWEN)
        return cls(
            text_dim=int(dims.get("text_dim", 2048)),
            audio_dim=int(dims.get("audio_dim", 1536)),
            video_dim=int(dims.get("video_dim", 768)),
            d_model=int(arch.get("d_model", 128)),
            num_heads=int(arch.get("num_heads", 4)),
            num_layers=int(arch.get("num_layers", 4)),
            ffn_dim=int(arch.get("ffn_dim", 384)),
            num_classes=int(arch.get("num_classes", 7)),
            parameter_budget=int(arch.get("parameter_budget", 2_000_000)),
            text_tower=text_tower,
            hf_text_model_id=str(source_meta.get("model_id", DEFAULT_HF_TEXT_MODEL)),
            hf_text_local_path=local_path,
            emotion_label_order=tuple(raw.get("emotion_label_order", DEFAULT_LABELS)),
            lr=float(train.get("lr", 3e-4)),
            weight_decay=float(train.get("weight_decay", 1e-2)),
            batch_size=int(train.get("batch_size", 8)),
            max_epochs=int(train.get("max_epochs", 5)),
            grad_clip=float(train.get("grad_clip", 1.0)),
            emotion_loss_weight=float(weights.get("emotion", 1.0)),
            utility_loss_weight=float(weights.get("utility", 0.2)),
            vad_loss_weight=float(weights.get("vad", 0.1)),
            mix_tau=float(train.get("mix_tau", 1.0)),
            mix_kl_weight=float(weights.get("mix_kl", 0.0)),
            mix_peak_weight=float(weights.get("mix_peak", 0.05)),
            mix_peak_cap=float(train.get("mix_peak_cap", 0.40)),
            seed=int((train.get("seeds") or [17])[0]),
        )
