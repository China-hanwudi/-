"""Offline MELD features: Qwen3-Omni thinker text + librosa audio + torchvision video.

Text uses the frozen Qwen3-Omni-30B-A3B-Instruct thinker (talker disabled).
Mean-pooled last hidden states are saved at native width (2048); N3 learns
the 2048→d_model projection. Audio/video come from the *real* current and
historical mp4 files, never copies of the current clip.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

_CACHE_LOCK = threading.Lock()
_AUDIO_CACHE: dict[str, torch.Tensor] = {}
_VIDEO_CACHE: dict[str, torch.Tensor] = {}
_TEXT_CACHE: dict[str, torch.Tensor] = {}


def load_manifest(path: Path, limit: int | None = None) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8", errors="replace") as f:
        for r in csv.DictReader(f):
            if str(r.get("video_missing", "")).lower() in {"1", "true", "True"}:
                continue
            rows.append(r)
            if limit is not None and len(rows) >= limit:
                break
    return rows


def wav_from_mp4(mp4: Path, sr: int = 16000) -> np.ndarray:
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        cmd = ["ffmpeg", "-y", "-i", str(mp4), "-ac", "1", "-ar", str(sr), "-vn", tmp.name]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        try:
            import soundfile as sf
            audio, _ = sf.read(tmp.name)
        except Exception:
            audio = np.zeros(sr, dtype=np.float32)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio.astype(np.float32)


def audio_feat(mp4: Path, dim: int = 1536) -> torch.Tensor:
    key = str(mp4)
    with _CACHE_LOCK:
        if key in _AUDIO_CACHE:
            return _AUDIO_CACHE[key]
    try:
        import librosa
        y = wav_from_mp4(mp4)
        if y.size < 16:
            ret = torch.zeros(dim)
            with _CACHE_LOCK:
                _AUDIO_CACHE[key] = ret
            return ret
        mfcc = librosa.feature.mfcc(y=y, sr=16000, n_mfcc=40)
        spec = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=64)
        vec = np.concatenate([mfcc.mean(1), mfcc.std(1), spec.mean(1), spec.std(1)])
    except Exception:
        vec = np.zeros(dim, dtype=np.float32)
    out = np.zeros(dim, dtype=np.float32)
    n = min(dim, vec.size)
    out[:n] = vec[:n]
    ret = torch.from_numpy(out)
    with _CACHE_LOCK:
        _AUDIO_CACHE[key] = ret
    return ret


def video_feat(mp4: Path, dim: int = 768) -> torch.Tensor:
    key = str(mp4)
    with _CACHE_LOCK:
        if key in _VIDEO_CACHE:
            return _VIDEO_CACHE[key]
    out = torch.zeros(dim)
    try:
        cmd = [
            "ffmpeg", "-v", "error", "-i", str(mp4),
            "-vf", "fps=4,scale=64:64",
            "-frames:v", "8",
            "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1",
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False)
        raw = proc.stdout or b""
        n = len(raw) // (64 * 64 * 3)
        if n > 0:
            frames = torch.from_numpy(
                np.frombuffer(raw, dtype=np.uint8, count=n * 64 * 64 * 3).copy()
            ).reshape(n, 64, 64, 3).float() / 255.0
            frames = frames.permute(0, 3, 1, 2)
            mean_c = frames.mean(dim=(0, 2, 3))
            std_c = frames.std(dim=(0, 2, 3))
            spatial = F.adaptive_avg_pool2d(frames.mean(dim=0, keepdim=True), (8, 8)).reshape(-1)
            vec = torch.cat([mean_c, std_c, spatial])
            nfill = min(dim, vec.numel())
            out[:nfill] = vec[:nfill]
    except Exception:
        out = torch.zeros(dim)
    with _CACHE_LOCK:
        _VIDEO_CACHE[key] = out
    return out


def format_utterance(speaker: str, text: str) -> str:
    speaker = (speaker or "").strip()
    text = (text or "").strip()
    if speaker and text:
        return f"{speaker}: {text}"
    return text


class QwenTextExtractor:
    """Frozen Qwen3-Omni thinker text tower; talker disabled. Native hidden size."""

    def __init__(self, model_path: str, *, cpu: bool = False) -> None:
        from transformers import AutoTokenizer, Qwen3OmniMoeForConditionalGeneration

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        load_kwargs: dict = {
            "trust_remote_code": True,
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        }
        if not cpu:
            load_kwargs["device_map"] = "auto"
        print(f"Loading Qwen3-Omni from {model_path} ...", flush=True)
        model = Qwen3OmniMoeForConditionalGeneration.from_pretrained(model_path, **load_kwargs)
        if hasattr(model, "disable_talker"):
            model.disable_talker()
        thinker = getattr(model, "thinker", None)
        self.text_model = getattr(thinker, "model", None) or getattr(model, "model", None) or model
        self.text_model.eval()
        for p in self.text_model.parameters():
            p.requires_grad = False
        self.hidden_size = self._guess_hidden_size(model)
        try:
            first_device = next(self.text_model.parameters()).device
        except StopIteration:
            first_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if first_device.type == "meta":
            first_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.first_device = first_device
        print(
            f"Qwen text tower ready. hidden_size={self.hidden_size}, "
            f"first_device={first_device} (no random projection)",
            flush=True,
        )

    def _guess_hidden_size(self, model) -> int:
        cfg = getattr(model, "config", None)
        if cfg is not None:
            tcfg = getattr(cfg, "thinker_config", None)
            if tcfg is not None:
                text_cfg = getattr(tcfg, "text_config", tcfg)
                hidden = getattr(text_cfg, "hidden_size", None)
                if hidden:
                    return int(hidden)
            hidden = getattr(cfg, "hidden_size", None)
            if hidden:
                return int(hidden)
        return 2048

    def encode_texts(self, texts: list[str]) -> torch.Tensor:
        if not texts:
            return torch.zeros(0, self.hidden_size)
        encoded = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {k: v.to(self.first_device) for k, v in encoded.items()}
        with torch.no_grad():
            out = self.text_model(**encoded, use_cache=False)
            hidden = out.last_hidden_state
            # Causal LM: last non-pad token, not mean-pool (mean-pool is near-isotropic).
            last_idx = encoded["attention_mask"].sum(dim=1).clamp(min=1) - 1
            pooled = hidden[torch.arange(hidden.size(0), device=hidden.device), last_idx]
        vecs = pooled.float().cpu()
        for text, vec in zip(texts, vecs):
            _TEXT_CACHE[text] = vec
        return vecs


def extract_text_batch(
    qwen: QwenTextExtractor, rows: list[dict]
) -> dict[str, dict[str, torch.Tensor]]:
    to_encode: list[tuple[str, int, str]] = []
    for r in rows:
        sid = r["sample_id"]
        current = format_utterance(r.get("speaker", ""), r.get("text", ""))
        to_encode.append((sid, -1, current if current else " "))
        for slot, (spk_key, txt_key) in enumerate(
            (("h0_speaker", "h0_text"), ("h1_speaker", "h1_text"), ("h2_speaker", "h2_text"))
        ):
            raw = (r.get(txt_key) or "").strip()
            if not raw:
                continue
            to_encode.append((sid, slot, format_utterance(r.get(spk_key, ""), raw)))
    texts = [t for _, _, t in to_encode]
    embeddings = torch.zeros(len(texts), qwen.hidden_size)
    missing_idx = []
    missing_texts = []
    for i, t in enumerate(texts):
        cached = _TEXT_CACHE.get(t)
        if cached is not None:
            embeddings[i] = cached
        else:
            missing_idx.append(i)
            missing_texts.append(t)
    if missing_texts:
        fresh = qwen.encode_texts(missing_texts)
        for j, idx in enumerate(missing_idx):
            embeddings[idx] = fresh[j]
    dim = qwen.hidden_size
    results: dict[str, dict[str, torch.Tensor]] = {}
    for r in rows:
        sid = r["sample_id"]
        hist_n = int(r.get("history_n") or 0)
        hmask = torch.zeros(3, dtype=torch.float32)
        for i in range(min(hist_n, 3)):
            hmask[i] = 1.0
        results[sid] = {
            "T_t": torch.zeros(dim),
            "T_h": torch.zeros(3, dim),
            "history_mask": hmask,
        }
    idx = 0
    for sid, slot, _ in to_encode:
        emb = embeddings[idx]
        idx += 1
        if slot == -1:
            results[sid]["T_t"] = emb
        else:
            results[sid]["T_h"][slot] = emb
    return results


def extract_sample(
    row: dict,
    text_emb: dict[str, torch.Tensor],
    audio_dim: int,
    video_dim: int,
) -> dict[str, torch.Tensor] | None:
    mp4 = Path(row["video_path"])
    if not mp4.is_file():
        return None
    A_t = audio_feat(mp4, audio_dim)
    V_t = video_feat(mp4, video_dim)
    A_h = torch.zeros(3, audio_dim)
    V_h = torch.zeros(3, video_dim)
    hist_mod = torch.zeros(3, dtype=torch.float32)
    if text_emb["history_mask"].sum() > 0:
        hist_mod[0] = 1.0
    for i, key in enumerate(("h0_video_path", "h1_video_path", "h2_video_path")):
        h_path = (row.get(key) or "").strip()
        if h_path and Path(h_path).is_file() and text_emb["history_mask"][i] > 0:
            A_h[i] = audio_feat(Path(h_path), audio_dim)
            V_h[i] = video_feat(Path(h_path), video_dim)
            hist_mod[1] = 1.0
            hist_mod[2] = 1.0
    has_text = 1.0 if (row.get("text") or "").strip() else 0.0
    modality_mask = torch.tensor([has_text, 1.0, 1.0], dtype=torch.float32)
    return {
        "T_t": text_emb["T_t"],
        "T_h": text_emb["T_h"],
        "A_t": A_t,
        "A_h": A_h,
        "V_t": V_t,
        "V_h": V_h,
        "history_mask": text_emb["history_mask"],
        "modality_mask": modality_mask,
        "history_modality_mask": hist_mod,
        "vad": torch.zeros(3),
    }


def process_split(
    qwen: QwenTextExtractor | None,
    manifest: Path,
    out_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_manifest(manifest, args.limit)
    print(f"manifest {manifest} rows={len(rows)}", flush=True)
    if len(rows) == 0:
        return {"ok": 0, "fail": 0, "n": 0, "manifest": str(manifest), "out_dir": str(out_dir)}

    ok = fail = 0
    failures = []
    total = len(rows)
    sanity_done = False
    av_workers = max(1, int(getattr(args, "av_workers", 8)))

    def _existing_ok(path: Path) -> bool:
        try:
            feat = torch.load(path, map_location="cpu", weights_only=True)
            return int(feat["T_t"].shape[-1]) == int(qwen.hidden_size)
        except Exception:
            return False

    pending = []
    for r in rows:
        out_p = out_dir / f"{r['sample_id']}.pt"
        if out_p.is_file() and _existing_ok(out_p):
            ok += 1
        else:
            pending.append(r)
    print(f"resume skip={ok} pending={len(pending)}", flush=True)

    for start in range(0, len(pending), args.text_batch_size):
        batch_rows = pending[start : start + args.text_batch_size]
        text_embs = extract_text_batch(qwen, batch_rows)
        if not sanity_done and qwen is not None and batch_rows:
            sids = [r["sample_id"] for r in batch_rows[:2]]
            if len(sids) >= 1:
                a = text_embs[sids[0]]["T_t"]
                nrm = float(a.norm().item())
                if len(sids) == 2:
                    b = text_embs[sids[1]]["T_t"]
                    cos = float(torch.nn.functional.cosine_similarity(a, b, dim=0).item())
                    print(f"QWEN_SANITY cos(utt0,utt1)={cos:.4f} ||utt0||={nrm:.4f} dim={a.numel()}", flush=True)
                else:
                    print(f"QWEN_SANITY ||utt0||={nrm:.4f} dim={a.numel()}", flush=True)
                if nrm < 1e-6:
                    raise RuntimeError("Qwen text embeddings are zero; aborting extraction")
            sanity_done = True

        def _one(r: dict) -> tuple[str, dict[str, torch.Tensor] | None, str | None]:
            sid = r["sample_id"]
            try:
                feat = extract_sample(r, text_embs[sid], args.audio_dim, args.video_dim)
                if feat is None:
                    return sid, None, "missing current video"
                return sid, feat, None
            except Exception as e:
                return sid, None, str(e)

        with ThreadPoolExecutor(max_workers=av_workers) as pool:
            results = list(pool.map(_one, batch_rows))
        for sid, feat, err in results:
            if feat is None:
                fail += 1
                failures.append({"sample_id": sid, "error": err})
                continue
            torch.save(feat, out_dir / f"{sid}.pt")
            ok += 1
        done = ok + fail
        if done % 40 == 0 or start + len(batch_rows) >= len(pending):
            print(f"progress {done}/{total} ok={ok} fail={fail}", flush=True)
    report = {
        "ok": ok,
        "fail": fail,
        "n": total,
        "manifest": str(manifest),
        "out_dir": str(out_dir),
        "qwen_path": args.qwen_path,
        "failures": failures[:200],
        "text_encoder": "Qwen3-Omni-30B-A3B-Instruct-thinker",
        "text_dim": int(qwen.hidden_size) if qwen is not None else args.text_dim,
        "projection": "none_native_hidden",
        "text_pooling": "last_nonpad_token_causal",
    }
    (out_dir / "extraction_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("EXTRACT_DONE", manifest.name, report["ok"], report["fail"])
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--manifests", type=Path, nargs="+", default=None)
    ap.add_argument("--out-dirs", type=Path, nargs="+", default=None)
    ap.add_argument("--qwen-path", type=str, default="/data/shared/qwen/Qwen3-Omni-30B-A3B-Instruct")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--text-batch-size", type=int, default=8)
    ap.add_argument("--av-workers", type=int, default=8)
    ap.add_argument("--text-dim", type=int, default=2048)
    ap.add_argument("--audio-dim", type=int, default=1536)
    ap.add_argument("--video-dim", type=int, default=768)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    if args.manifests and args.out_dirs:
        manifests = list(args.manifests)
        out_dirs = list(args.out_dirs)
    elif args.manifest and args.out_dir:
        manifests = [args.manifest]
        out_dirs = [args.out_dir]
    else:
        raise SystemExit("provide either --manifest + --out-dir or --manifests + --out-dirs")
    if len(manifests) != len(out_dirs):
        raise SystemExit("--manifests and --out-dirs must have same length")

    qwen = QwenTextExtractor(args.qwen_path, cpu=args.cpu)
    args.text_dim = qwen.hidden_size

    all_reports = []
    for manifest, out_dir in zip(manifests, out_dirs):
        all_reports.append(process_split(qwen, manifest, out_dir, args))
    summary = {
        "reports": all_reports,
        "total_ok": sum(r["ok"] for r in all_reports),
        "total_fail": sum(r["fail"] for r in all_reports),
    }
    print("ALL_EXTRACT_DONE", summary["total_ok"], summary["total_fail"])
    return 0 if summary["total_fail"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
