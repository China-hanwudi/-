"""Complete feature-input ReCoMER: latest MHnoU + frozen cRBEF + local_current."""
import torch
from torch import nn

from .crbef_expert import CRBEFExpert
from .crbef_fusion import LocalCurrentConfig, LocalCurrentFusion
from .evaluate import load_checkpoint
from .config import M6Config
from .model import UGFModel


@torch.no_grad()
def mhnou_reference_outputs(model, batch):
    """Use the SAME MHnoU checkpoint for historical and no-history references."""
    # Targets are unnecessary for inference and must not enter the wrapper.
    b = {k: v for k, v in batch.items() if k != "history_label"}
    out = model(b)
    nohist = dict(b, history_mask=torch.zeros_like(b["history_mask"]),
                  history_modality_mask=torch.zeros_like(b["history_modality_mask"]))
    out0 = model(nohist)
    L0 = out0["solo_stack"]
    M = b["modality_mask"]
    w = M / M.sum(1, keepdim=True).clamp_min(1)
    centered = L0 - L0.mean(2, keepdim=True)
    C = (centered - (centered * w[:, :, None]).sum(1, keepdim=True)) * M[:, :, None]
    return dict(MHnoU=out["deployed"].softmax(1), no_history=out0["deployed"].softmax(1),
                C=C, L0=L0, Lh=out["solo_stack"], M=M)


def combine_inputs(mh, cr):
    if mh["MHnoU"].shape != cr["CRBEF"].shape or cr["TAV"].shape != cr["CRBEF"].shape:
        raise ValueError("MHnoU and cRBEF predictions must have equal [N,C] shapes")
    P = torch.stack([cr["CRBEF"], mh["MHnoU"], cr["TAV"], mh["no_history"]], dim=1)
    return P, mh["C"]


class ReCoMER(nn.Module):
    """The three trained modules form one inference object; MHnoU is preserved."""
    def __init__(self, mhnou, expert, fusion=None):
        super().__init__()
        self.mhnou, self.expert, self.fusion = mhnou, expert, fusion
        cfg = mhnou.cfg
        if cfg.task != "cls" or tuple(cfg.class_names or ()) != expert.class_names:
            raise ValueError("this expert requires M3ED classification and matching class order")
        if cfg.deploy != "closed_loop" or cfg.utility != "shapley" or not cfg.use_bounded_w or cfg.gate_architecture != "evidence" or cfg.history_abstain_variant != "utility_softmax":
            raise ValueError("ReCoMER requires the final MHnoU with all three components enabled")
        self.eval()

    @classmethod
    def from_bundle(cls, path, device="cpu"):
        ck = torch.load(path, map_location="cpu", weights_only=True)
        if ck.get("format") != "ReCoMER_feature_bundle_v1":
            raise ValueError("unrecognized ReCoMER checkpoint bundle")
        mh = UGFModel(M6Config.from_dict(ck["mhnou_cfg"]))
        mh.load_state_dict(ck["mhnou"], strict=True)
        expert = CRBEFExpert.from_state(ck["expert"], ck["class_names"])
        fusion = LocalCurrentFusion(LocalCurrentConfig(**ck["fusion_cfg"]))
        fusion.load_state_dict(ck["fusion"], strict=True)
        return cls(mh, expert, fusion).to(device).eval()

    def save_bundle(self, path, provenance):
        if self.fusion is None:
            raise RuntimeError("cannot package an unfitted ReCoMER")
        cpu_state = lambda module: {k: v.detach().cpu() for k, v in module.state_dict().items()}
        torch.save(dict(format="ReCoMER_feature_bundle_v1", mhnou_cfg=self.mhnou.cfg.to_dict(),
                        mhnou=cpu_state(self.mhnou), expert=cpu_state(self.expert),
                        class_names=list(self.expert.class_names), fusion_cfg=self.fusion.cfg.__dict__,
                        fusion=cpu_state(self.fusion), provenance=provenance), path)

    @classmethod
    def from_files(cls, mhnou_checkpoint, expert_assets, fusion_checkpoint=None, device="cpu"):
        mh, cfg, _ = load_checkpoint(mhnou_checkpoint, device)
        expert = CRBEFExpert(expert_assets).to(device)
        fusion = None
        if fusion_checkpoint is not None:
            ck = torch.load(fusion_checkpoint, map_location="cpu", weights_only=True)
            if ck.get("class_names") != list(expert.class_names):
                raise ValueError("fusion checkpoint class order is missing or incompatible")
            fc = LocalCurrentConfig(**ck["cfg"])
            fusion = LocalCurrentFusion(fc)
            fusion.load_state_dict(ck["model"], strict=True)
            fusion.to(device).eval()
        return cls(mh, expert, fusion).to(device).eval()

    @torch.no_grad()
    def build_inputs(self, batch, expert_features):
        mh = mhnou_reference_outputs(self.mhnou, batch)
        cr = self.expert(expert_features)
        P, C = combine_inputs(mh, cr)
        return dict(P=P, **mh, **cr)

    @torch.no_grad()
    def forward(self, batch, expert_features):
        if self.fusion is None:
            raise RuntimeError("fit/load local_current before requesting final ReCoMER predictions")
        out = self.build_inputs(batch, expert_features)
        logp, details = self.fusion(out["P"], out["C"], return_details=True)
        return dict(**out, average=out["P"][:, :2].mean(1), fused=logp.exp(),
                    fusion_delta=details["delta"], candidate_mask=details["candidate_mask"])
