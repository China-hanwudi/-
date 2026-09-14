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
    label_smoothing: float = 0.05
    current_aux_loss_weight: float = 0.25
    history_aux_loss_weight: float = 0.50
    risk_loss_weight: float = 0.50
    coverage_loss_weight: float = 0.10
    risk_margin: float = 0.0
    risk_threshold: float = 0.50
    risk_feature_version: str = "relation_v1"
    # Training-only stochastic corruption of historical evidence.  Evaluation
    # is deterministic and uses the full observed history/mask.
    history_dropout: float = 0.15
    modality_dropout: float = 0.10
    # ---- v5 innovation weights -------------------------------------------
    # Weight of the *measured* leave-one-modality-out utility alignment.  When
    # the caller supplies ``cf_measured_targets`` the router head is trained
    # against a genuine ablation instead of a self-referential rank target.
    counterfactual_loss_weight: float = 0.30
    cross_modal_consistency_weight: float = 0.03
    # Redundancy-aware terms: sign agreement between modalities and
    # decorrelation of private channels.  Both default to zero so legacy
    # checkpoints and legacy objectives are bit-exact unless explicitly enabled.
    sign_consistency_weight: float = 0.0
    private_orthogonality_weight: float = 0.0
    # v6 "solve point": penalise the all-reject routing collapse observed on
    # CMU-MOSEI (history use rate 0.0064) and the modality shortcut.
    risk_collapse_weight: float = 0.0
    text_shortcut_weight: float = 0.0
    # v6 core: supervise each modality against its own label when the dataset
    # provides them (CH-SIMS v2).  Zero disables the term for other datasets.
    unimodal_loss_weight: float = 0.0
    # Adaptive accept budget for the hard fallback (v6 repair).
    use_adaptive_budget: bool = True
    accept_budget_init: float = 0.35
    accept_warmup_epochs: int = 3
    # Keep the empty-history contract strict by default.  Isolated-clip
    # experiments may opt into a current-as-candidate ablation explicitly.
    allow_current_candidate_without_history: bool = False

    def validate(self) -> None:
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.num_classes < 2:
            raise ValueError("N3 train config requires at least two emotion classes")
        if self.text_tower not in ALLOWED_TEXT_TOWERS:
            raise ValueError(f"unknown text_tower: {self.text_tower}")
        if len(self.emotion_label_order) != self.num_classes:
            raise ValueError("emotion_label_order length must equal num_classes")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in [0, 1)")
        if not 0.0 < self.risk_threshold < 1.0:
            raise ValueError("risk_threshold must be in (0, 1)")
        if self.risk_feature_version not in {"relation_v1", "conflict_v2"}:
            raise ValueError("risk_feature_version must be relation_v1 or conflict_v2")
        if not 0.0 < self.accept_budget_init < 1.0 or self.accept_warmup_epochs < 0:
            raise ValueError("accept_budget_init must be in (0,1) and accept_warmup_epochs non-negative")
        if any(weight < 0.0 for weight in (
            self.counterfactual_loss_weight, self.cross_modal_consistency_weight,
            self.sign_consistency_weight, self.private_orthogonality_weight,
            self.risk_collapse_weight, self.text_shortcut_weight, self.unimodal_loss_weight,
        )):
            raise ValueError("innovation loss weights must be non-negative")
        if any(weight < 0.0 for weight in (self.current_aux_loss_weight, self.history_aux_loss_weight, self.risk_loss_weight, self.coverage_loss_weight)):
            raise ValueError("auxiliary loss weights must be non-negative")
        if not 0.0 <= self.history_dropout < 1.0 or not 0.0 <= self.modality_dropout < 1.0:
            raise ValueError("history/modality dropout must be in [0, 1)")

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
            label_smoothing=float(train.get("label_smoothing", 0.05)),
            current_aux_loss_weight=float(weights.get("current_aux", 0.25)),
            history_aux_loss_weight=float(weights.get("history_aux", 0.50)),
            risk_loss_weight=float(weights.get("risk", 0.50)),
            coverage_loss_weight=float(weights.get("coverage", 0.10)),
            risk_margin=float(train.get("risk_margin", 0.0)),
            risk_threshold=float(train.get("risk_threshold", 0.50)),
            risk_feature_version=str(arch.get("risk_feature_version", "conflict_v2")),
            history_dropout=float(train.get("history_dropout", 0.15)),
            modality_dropout=float(train.get("modality_dropout", 0.10)),
            seed=int((train.get("seeds") or [17])[0]),
        )
