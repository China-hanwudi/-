"""Configuration for the explicit continuous MOSEI sentiment adaptation."""
from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass
class N3RegressionConfig:
    text_dim: int
    audio_dim: int
    video_dim: int
    d_model: int = 128
    num_heads: int = 4
    num_layers: int = 4
    ffn_dim: int = 384
    dropout: float = 0.1
    relation_rank: int = 32
    gate_hidden: int = 64
    parameter_budget: int = 2_000_000
    text_tower: str = "composer_n3"
    task: str = "mosei_sentiment_regression"
    output_dim: int = 1
    target_scale: float = 3.0
    risk_feature_version: str = "regression_v1"
    risk_threshold: float = 0.50
    # Regression history is an additive correction to the current anchor.  A
    # bounded residual prevents an occasional noisy past clip from dominating
    # the scalar prediction (the old implementation allowed unbounded jumps).
    history_delta_cap: float = 0.35
    # Fixed, pre-registered uncertainty inflation used by the hard fallback.
    risk_upper_coef: float = 0.15
    risk_margin: float = 0.0
    current_aux_loss_weight: float = 0.25
    history_aux_loss_weight: float = 0.50
    risk_loss_weight: float = 0.50
    coverage_loss_weight: float = 0.10
    history_dropout: float = 0.15
    modality_dropout: float = 0.10
    mix_tau: float = 1.0
    mix_kl_weight: float = 0.0
    mix_peak_weight: float = 0.05
    mix_peak_cap: float = 0.40
    lr: float = 3e-4
    weight_decay: float = 0.01
    batch_size: int = 64
    max_epochs: int = 20
    grad_clip: float = 1.0
    seed: int = 17

    def validate(self) -> None:
        if min(self.text_dim, self.audio_dim, self.video_dim, self.d_model,
               self.num_heads, self.num_layers, self.ffn_dim, self.relation_rank, self.gate_hidden) < 1:
            raise ValueError("Dimensions and layer counts must be positive")
        if self.d_model % self.num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        if self.task != "mosei_sentiment_regression" or self.output_dim != 1:
            raise ValueError("This model implements scalar sentiment regression only")
        if self.text_tower != "composer_n3":
            raise ValueError("Regression adaptation accepts precomputed features only")
        if self.target_scale != 3.0 or self.risk_feature_version != "regression_v1":
            raise ValueError("Expected y/3 normalization and regression_v1 risk features")
        if not 0 < self.risk_threshold < 1:
            raise ValueError("risk_threshold must be in (0,1)")
        if self.history_delta_cap <= 0:
            raise ValueError("history_delta_cap must be positive")
        if self.risk_upper_coef < 0:
            raise ValueError("risk_upper_coef must be non-negative")
        if any(not 0 <= p < 1 for p in (self.dropout, self.history_dropout, self.modality_dropout)):
            raise ValueError("Dropout probabilities must be in [0,1)")
        if any(w < 0 for w in (self.current_aux_loss_weight, self.history_aux_loss_weight,
                              self.risk_loss_weight, self.coverage_loss_weight,
                              self.mix_kl_weight, self.mix_peak_weight, self.risk_margin)):
            raise ValueError("Loss weights and the relative-harm margin must be nonnegative")

    def to_dict(self) -> dict:
        return asdict(self)
