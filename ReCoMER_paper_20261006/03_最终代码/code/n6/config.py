"""Configuration for model6 (UGF).

Everything the trainer / evaluator needs to rebuild a run is stored in this
plain dataclass so that ``best.pt`` alone is sufficient to reconstruct the
model and the resolved hyperparameters.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class M6Config:
    # input dims (per modality, pooled vectors)
    text_dim: int = 768
    audio_dim: int = 2048
    video_dim: int = 342
    # task: "cls" (num_classes from DatasetSpec) or "reg" (num_classes = 1)
    task: str = "cls"
    num_classes: int = 7
    # A01/P0-02: dataset provenance written at training time.  Legacy
    # checkpoints carry "" / None and evaluators re-resolve from the pack
    # path (datasetspec.class_names_for); labels are never guessed silently.
    dataset_id: str = ""
    class_names: Optional[List[str]] = None
    # architecture
    d_model: int = 192
    dropout: float = 0.15
    num_heads: int = 6
    ff_dim: int = 576
    use_history: bool = True
    use_abstain: bool = False
    history_k: int = 3
    # Innovation-2 router.  ``legacy`` preserves old utility-head checkpoints;
    # ``evidence`` is the revised per-modality EvidenceRouter and ``constant``
    # is a fixed-preference control.
    use_bounded_w: bool = False
    bounded_lambda: float = 0.3
    contrib_correct: bool = False
    detach_utility_path: bool = False
    gate_architecture: str = "legacy"  # legacy | evidence | constant
    gate_detach_inputs: bool = False
    # Optional history-state features exposed to the EvidenceRouter.  Kept at
    # zero for legacy checkpoints so their router tensor shapes remain stable.
    history_feature_dim: int = 0
    # Old checkpoints may contain a null parameter even when it was disabled.
    legacy_history_null: bool = False
    legacy_history_null_active: bool = False
    # Innovation 3 v2: history admission.  ``none`` is the original attention
    # pool, ``legacy_null`` is the previous empty-candidate implementation and
    # ``sentence`` is the new one gate per utterance.  New checkpoints carry
    # these fields explicitly; absent fields in old checkpoints remain none.
    history_gate_mode: str = "none"       # none | legacy_null | sentence
    history_gate_execution: str = "hard"  # soft | hard (sentence only)
    history_gate_target: str = "label_match"  # label_match | utility
    history_gate_features: str = "base"   # base | semantic
    history_gate_lambda: float = 0.3
    history_gate_threshold: float = 0.5
    history_gate_hidden: int = 32
    history_gate_warmup_epochs: int = 3
    history_gate_soft_epochs: int = 2
    history_gate_temperature: float = 1.0
    # Innovation 3 core empty-candidate variants.  These keep the original
    # history candidates + one empty candidate competition; utility variants
    # only recalibrate the empty-candidate logit before normalisation.
    history_abstain_variant: str = "none"  # none|legacy|utility_softmax|utility_semantic|utility_sparsemax
    history_null_beta: float = 1.0
    history_utility_lambda: float = 0.3
    history_utility_hidden: int = 32
    history_utility_temperature: float = 1.0
    history_utility_cache: str = ""
    # utility gating
    utility: str = "shapley"      # "uniform" | "gate" | "shapley"
    deploy: str = "solo_weighted"  # + "closed_loop" for the integrated framework
    utility_head_version: int = 2  # 2 = (mu, sigma) double head; 1 = legacy single head
    lambda_u: float = 0.3
    lambda_c: float = 0.05
    eps_floor: float = 0.05
    tau: float = 1.0
    # iteration-3 switchable modules (all default OFF -> round-1/2 arms reproduce)
    use_m1: bool = False          # counterfactual-view training of the joint head
    use_m2: bool = False          # regret gating: w = softmax((mu - kappa*sigma)/tau)
    use_m3: bool = False          # utility-consistent manifold mixup
    lambda_j: float = 0.5         # M1: full-joint supervision weight
    lambda_v: float = 0.3         # M1: counterfactual-view supervision weight
    mix_alpha: float = 0.4        # M3: Beta(alpha, alpha) mix coefficient
    kappa: float = 1.0            # M2: risk-aversion coefficient
    # 2026-09-24 performance round-1 (plan 04): additive residual on the
    # deployed logits.  "none" (default) = existing checkpoints reproduce.
    residual: str = "none"        # "none" | "concat" | "pair"
    resid_rank: int = 48          # pair residual low-rank
    resid_hidden: int = 96        # concat residual hidden width
    # W6 gap-fix switches (default OFF -> all earlier arms reproduce)
    grl_lambda: float = 0.0       # series-adversarial head strength (0 = off)
    selection_series: bool = False  # select on per-series WF1 mean (valid only)
    n_series: int = 0             # series classes (train ids), set by the trainer
    # W9 StreamFusion switches (default OFF -> all earlier arms reproduce)
    use_vpath: bool = False       # frame-level video path (needs pack key Vf)
    use_mpath: bool = False       # SAM2-style memory attention over history
    use_ugate: bool = False       # micro-gates on the V/M corrections
    c1_alpha: float = 0.0         # prior-calibrated evidence blend (cls only)
    class_prior: Optional[List[float]] = None  # train class prior for C1
    # audit bugfix lineage: 1 = pre-fix code (rounds 1-W9 first pass);
    # 2 = 2026-09-21 audit fixes (V-path single application, speaker-mask
    # reorder, KL batchmean).  New training writes 2; old checkpoints load
    # with the default 1.
    bugfix_version: int = 1
    # mask-aware output fusion (framework candidate 2, stage 2): absent
    # modalities contribute nothing to the deployed sum and weights
    # renormalize over present modalities.  DEFAULT False so existing
    # checkpoints evaluate with their saved behavior; new training runs
    # enable it explicitly.
    mask_fusion_fix: bool = False
    # objective details
    label_smoothing: float = 0.05
    class_weight_cap: float = 2.0
    # optimisation (recorded for the manifest; the loop owns the schedule)
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip: float = 1.0

    def validate(self) -> None:
        if self.gate_architecture not in ("legacy", "evidence", "constant"):
            raise ValueError("unknown gate_architecture")
        if self.gate_architecture != "legacy":
            if self.utility != "shapley" or not self.use_bounded_w:
                raise ValueError("new router requires Shapley and bounded weights")
            if self.contrib_correct or self.use_m2 or self.use_m3:
                raise ValueError("new router requires no contribution residual, M2, or M3")
            if not self.gate_detach_inputs or not self.detach_utility_path:
                raise ValueError("new router requires detached inputs and deployment weights")
        if self.history_feature_dim not in (0, 4):
            raise ValueError("history_feature_dim must be 0 or 4")
        if self.task not in ("cls", "reg"):
            raise ValueError("task must be 'cls' or 'reg', got %r" % (self.task,))
        if self.utility not in ("uniform", "gate", "shapley"):
            raise ValueError(
                "utility must be one of uniform|gate|shapley, got %r" % (self.utility,))
        if self.deploy not in ("solo_weighted", "joint_uniform", "joint_softgate", "closed_loop"):
            raise ValueError(
                "deploy must be one of solo_weighted|joint_uniform|joint_softgate|closed_loop, "
                "got %r" % (self.deploy,))
        if self.use_m1 and self.deploy in ("joint_softgate", "closed_loop"):
            raise ValueError(
                "M1 currently uses an unweighted counterfactual token view; "
                "it cannot be combined with weighted joint deployment. "
                "Disable --m1 or use a solo/joint-uniform deployment.")
        if self.history_gate_mode not in ("none", "legacy_null", "sentence"):
            raise ValueError("history_gate_mode must be none|legacy_null|sentence")
        if self.history_gate_execution not in ("soft", "hard"):
            raise ValueError("history_gate_execution must be soft|hard")
        if self.history_gate_target not in ("label_match", "utility"):
            raise ValueError("history_gate_target must be label_match|utility")
        if self.history_gate_features not in ("base", "semantic"):
            raise ValueError("history_gate_features must be base|semantic")
        if self.history_abstain_variant not in ("none", "legacy", "utility_softmax", "utility_semantic", "utility_sparsemax"):
            raise ValueError("invalid history_abstain_variant")
        if self.history_null_beta < 0.0 or self.history_utility_lambda < 0.0 or self.history_utility_hidden <= 0 or self.history_utility_temperature <= 0.0:
            raise ValueError("invalid history utility parameters")
        if not 0.0 <= self.history_gate_threshold <= 1.0:
            raise ValueError("history_gate_threshold must be in [0,1]")
        if (self.history_gate_lambda < 0.0 or self.history_gate_hidden <= 0
                or self.history_gate_warmup_epochs < 0
                or self.history_gate_soft_epochs < 0
                or self.history_gate_temperature <= 0.0):
            raise ValueError("invalid history-gate hyperparameters")
        if self.mix_alpha <= 0.0:
            raise ValueError("mix_alpha must be positive")
        # 2026-09-23 fix (third-party blocker): num_classes was hardcoded to 7 for cls,
        # rejecting IEMOCAP 4-class checkpoints on reload. Relaxed to >=2 / ==1.
        if self.task == "cls":
            if self.num_classes < 2:
                raise ValueError("num_classes must be >= 2 for cls, got %r" % (self.num_classes,))
        elif self.num_classes != 1:
            raise ValueError("num_classes must be 1 for reg, got %r" % (self.num_classes,))
        if self.residual not in ("none", "concat", "pair"):
            raise ValueError(
                "residual must be one of none|concat|pair, got %r"
                % (self.residual,))
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.class_names is not None and len(self.class_names) != self.num_classes:
            raise ValueError(
                "class_names must have num_classes=%d entries, got %d"
                % (self.num_classes, len(self.class_names)))
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        if self.lambda_u < 0.0 or self.lambda_c < 0.0:
            raise ValueError("loss weights must be non-negative")
        if self.eps_floor < 0.0 or self.tau <= 0.0:
            raise ValueError("need eps_floor >= 0 and tau > 0")
        if self.bounded_lambda < 0.0:
            raise ValueError("bounded_lambda must be non-negative")

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, object]) -> "M6Config":
        known = {f for f in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        filtered = {k: v for k, v in d.items() if k in known}
        cfg = cls(**filtered)  # type: ignore[arg-type]
        cfg.validate()
        return cfg
