from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import shutil
import sys
import tarfile
import tempfile
import time
import wave
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from scipy.signal import resample_poly
from transformers import AutoFeatureExtractor, AutoImageProcessor, AutoModel


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hva_affect.data_contract import write_json_atomic  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark frozen EmotionTalk encoders")
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--wavlm", type=Path, required=True)
    parser.add_argument("--dinov2", type=Path, required=True)
    parser.add_argument("--audio-tar", type=Path, required=True)
    parser.add_argument("--video-tar", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=4)
    return parser.parse_args()


def load_index(path: Path, keys: set[str]) -> dict[tuple[str, str], dict]:
    result: dict[tuple[str, str], dict] = {}
    with gzip.open(path, "rt", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["key"] in keys:
                result[(row["key"], row["modality"])] = row
    expected = {(key, modality) for key in keys for modality in ("audio", "video")}
    if set(result) != expected:
        raise RuntimeError(f"benchmark index incomplete: {sorted(expected - set(result))}")
    return result


def read_wav(handle) -> tuple[np.ndarray, int, dict]:
    with wave.open(handle, "rb") as wav:
        channels = wav.getnchannels()
        rate = wav.getframerate()
        width = wav.getsampwidth()
        frames = wav.getnframes()
        if width != 2:
            raise RuntimeError(f"expected 16-bit PCM, got {width} bytes")
        data = np.frombuffer(wav.readframes(frames), dtype="<i2").reshape(-1, channels)
    mono = data.astype(np.float32).mean(axis=1) / 32768.0
    if rate != 16000:
        divisor = math.gcd(rate, 16000)
        mono = resample_poly(mono, 16000 // divisor, rate // divisor).astype(np.float32)
    return mono, 16000, {
        "source_rate": rate,
        "source_channels": channels,
        "source_frames": frames,
        "resampled_samples": len(mono),
        "duration_seconds": frames / rate,
    }


def audio_embedding(model, processor, waveform: np.ndarray, device: torch.device) -> np.ndarray:
    inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")
    values = inputs.input_values.to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        hidden = model(values).last_hidden_state.float()[0]
    return torch.cat([hidden.mean(dim=0), hidden.std(dim=0, unbiased=False)]).cpu().numpy()


def uniform_frames(video_path: Path, count: int) -> tuple[list[np.ndarray], dict]:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV cannot open {video_path}")
    total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if total <= 0:
        capture.release()
        raise RuntimeError(f"invalid frame count for {video_path}")
    fractions = np.linspace(0.1, 0.9, count)
    indices = sorted({min(total - 1, max(0, int(round(value * (total - 1))))) for value in fractions})
    frames: list[np.ndarray] = []
    for index in indices:
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if ok and frame is not None:
            frames.append(frame)
    capture.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {video_path}")
    return frames, {"total_frames": total, "fps": fps, "sample_indices": indices, "decoded_frames": len(frames)}


def load_face_detector() -> cv2.CascadeClassifier:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    # OpenCV's Windows file loader cannot reliably open non-ASCII paths. The
    # project venv lives under a Japanese/Chinese OneDrive path, so load the
    # exact packaged XML through an ASCII temporary path instead.
    temporary_xml: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as handle:
            temporary_xml = Path(handle.name)
            handle.write(cascade_path.read_bytes())
        detector = cv2.CascadeClassifier(str(temporary_xml))
    finally:
        if temporary_xml is not None:
            temporary_xml.unlink(missing_ok=True)
    if detector.empty():
        raise RuntimeError("OpenCV Haar face detector unavailable")
    return detector


def face_or_full_frames(
    frames: list[np.ndarray], detector: cv2.CascadeClassifier | None = None
) -> tuple[list[Image.Image], dict]:
    detector = detector or load_face_detector()
    images: list[Image.Image] = []
    detected = 0
    crop_fractions: list[float] = []
    for frame in frames:
        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(64, 64))
        crop = frame
        if len(faces):
            x, y, w, h = max(faces, key=lambda box: int(box[2]) * int(box[3]))
            margin = int(round(0.3 * max(w, h)))
            x0, y0 = max(0, x - margin), max(0, y - margin)
            x1, y1 = min(width, x + w + margin), min(height, y + h + margin)
            crop = frame[y0:y1, x0:x1]
            detected += 1
            crop_fractions.append(float(crop.shape[0] * crop.shape[1] / (height * width)))
        else:
            crop_fractions.append(1.0)
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(rgb))
    return images, {
        "face_detected_frames": detected,
        "face_detection_rate": detected / len(frames),
        "mean_crop_area_fraction": float(np.mean(crop_fractions)),
    }


def video_embedding(model, processor, images: list[Image.Image], device: torch.device) -> np.ndarray:
    inputs = processor(images=images, return_tensors="pt")
    values = inputs.pixel_values.to(device)
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        hidden = model(pixel_values=values).last_hidden_state[:, 0, :].float()
    return torch.cat([hidden.mean(dim=0), hidden.std(dim=0, unbiased=False)]).cpu().numpy()


def copy_member(tar: tarfile.TarFile, member_name: str, suffix: str) -> Path:
    member = tar.getmember(member_name)
    source = tar.extractfile(member)
    if source is None:
        raise RuntimeError(f"unreadable tar member {member_name}")
    handle = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    path = Path(handle.name)
    try:
        with handle:
            shutil.copyfileobj(source, handle, length=1024 * 1024)
        if path.stat().st_size != member.size:
            raise RuntimeError(f"temporary extraction size mismatch for {member_name}")
        return path
    except Exception:
        path.unlink(missing_ok=True)
        raise


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the frozen encoder benchmark")
    audit = json.loads(args.audit.read_text(encoding="utf-8"))
    keys = sorted(set(audit["sample_probes"]["audio"]) & set(audit["sample_probes"]["video"]))
    if len(keys) < 3:
        raise RuntimeError("audit report has too few aligned benchmark samples")
    index = load_index(args.index, set(keys))
    device = torch.device("cuda")

    started = time.perf_counter()
    audio_processor = AutoFeatureExtractor.from_pretrained(args.wavlm, local_files_only=True)
    audio_model = AutoModel.from_pretrained(args.wavlm, local_files_only=True).eval().to(device)
    audio_load_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    audio_rows: dict[str, dict] = {}
    audio_started = time.perf_counter()
    with tarfile.open(args.audio_tar, "r:") as tar:
        for key in keys:
            member = tar.getmember(index[(key, "audio")]["outer_member"])
            source = tar.extractfile(member)
            if source is None:
                raise RuntimeError(f"unreadable WAV {key}")
            waveform, _, metadata = read_wav(source)
            before = time.perf_counter()
            embedding = audio_embedding(audio_model, audio_processor, waveform, device)
            audio_rows[key] = {
                **metadata,
                "embedding_dim": int(embedding.size),
                "finite": bool(np.isfinite(embedding).all()),
                "l2_norm": float(np.linalg.norm(embedding)),
                "inference_seconds": time.perf_counter() - before,
            }
    first_key = keys[0]
    with tarfile.open(args.audio_tar, "r:") as tar:
        source = tar.extractfile(tar.getmember(index[(first_key, "audio")]["outer_member"]))
        waveform, _, _ = read_wav(source)
    repeated_a = audio_embedding(audio_model, audio_processor, waveform, device)
    repeated_b = audio_embedding(audio_model, audio_processor, waveform, device)
    audio_total_seconds = time.perf_counter() - audio_started
    audio_peak = torch.cuda.max_memory_allocated() / 2**20
    del audio_model
    torch.cuda.empty_cache()

    started = time.perf_counter()
    video_processor = AutoImageProcessor.from_pretrained(args.dinov2, local_files_only=True, use_fast=False)
    video_model = AutoModel.from_pretrained(args.dinov2, local_files_only=True).eval().to(device)
    video_load_seconds = time.perf_counter() - started
    torch.cuda.reset_peak_memory_stats()
    video_rows: dict[str, dict] = {}
    video_started = time.perf_counter()
    first_images: list[Image.Image] | None = None
    with tarfile.open(args.video_tar, "r:") as tar:
        for key in keys:
            temporary = copy_member(tar, index[(key, "video")]["outer_member"], ".mp4")
            try:
                before = time.perf_counter()
                frames, decode = uniform_frames(temporary, args.frames)
                images, faces = face_or_full_frames(frames)
                embedding = video_embedding(video_model, video_processor, images, device)
                video_rows[key] = {
                    **decode,
                    **faces,
                    "embedding_dim": int(embedding.size),
                    "finite": bool(np.isfinite(embedding).all()),
                    "l2_norm": float(np.linalg.norm(embedding)),
                    "decode_and_inference_seconds": time.perf_counter() - before,
                }
                if key == first_key:
                    first_images = images
            finally:
                temporary.unlink(missing_ok=True)
    if first_images is None:
        raise RuntimeError("first deterministic video sample was not decoded")
    repeated_v_a = video_embedding(video_model, video_processor, first_images, device)
    repeated_v_b = video_embedding(video_model, video_processor, first_images, device)
    video_total_seconds = time.perf_counter() - video_started
    video_peak = torch.cuda.max_memory_allocated() / 2**20

    report = {
        "dataset_revision": audit["revision"],
        "benchmark_keys": keys,
        "device": torch.cuda.get_device_name(0),
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "opencv": cv2.__version__,
        },
        "audio": {
            "model_path": str(args.wavlm),
            "model_revision": args.wavlm.name,
            "aggregation": "last_hidden_state temporal mean + std",
            "load_seconds": audio_load_seconds,
            "total_seconds": audio_total_seconds,
            "samples_per_second_excluding_load": len(keys) / audio_total_seconds,
            "peak_cuda_mib": audio_peak,
            "repeat_max_abs_difference": float(np.max(np.abs(repeated_a - repeated_b))),
            "samples": audio_rows,
        },
        "video": {
            "model_path": str(args.dinov2),
            "model_revision": args.dinov2.name,
            "frames_per_video": args.frames,
            "frame_positions": "evenly spaced from 10% to 90%",
            "crop": "largest Haar frontal face + 30% margin; otherwise full frame",
            "aggregation": "CLS-token frame mean + std",
            "load_seconds": video_load_seconds,
            "total_seconds": video_total_seconds,
            "samples_per_second_excluding_load": len(keys) / video_total_seconds,
            "peak_cuda_mib": video_peak,
            "repeat_max_abs_difference": float(np.max(np.abs(repeated_v_a - repeated_v_b))),
            "samples": video_rows,
        },
        "label_fields_read": False,
        "temporary_media_retained": False,
    }
    write_json_atomic(report, args.output.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
