"""Extract conservative speaker-associated face CLIP features from MELD.

Reads only train/valid. Each shard saves target-face evidence separately from
the immutable scene pack. The speaker prototype was fitted on train metadata
and is not emotion-supervised. UNKNOWN faces produce zero features/masks.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from transformers import CLIPModel, CLIPProcessor

from n7_visual.face_identity import FaceTools, match_target


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_unique(indices: np.ndarray, scene_valid: np.ndarray) -> np.ndarray:
    """Keep the first successful read of each *source frame index*.

    Feature cosine similarity is not an identity test for decoded frames.
    """
    unique = np.zeros(len(indices), dtype=np.bool_)
    seen = set()
    for j, (idx, valid) in enumerate(zip(indices, scene_valid)):
        if valid and int(idx) not in seen:
            seen.add(int(idx))
            unique[j] = True
    return unique


def extract_item(item, original, tools, clip, processor, device):
    indices = original["frame_index"]
    frame_valid = original["scene_valid"]
    unique = source_unique(indices, frame_valid)
    features = np.zeros((len(indices), 512), dtype=np.float32)
    face_valid = np.zeros(len(indices), dtype=np.bool_)
    similarity = np.zeros(len(indices), dtype=np.float32)
    margin = np.zeros(len(indices), dtype=np.float32)
    face_count = np.zeros(len(indices), dtype=np.int16)
    crop_rows, crop_positions = [], []
    cap = cv2.VideoCapture(item["video_path"])
    try:
        if cap.isOpened():
            current_index = -1
            for j, idx in enumerate(indices):
                if not unique[j]:
                    continue
                # Frame indices are ascending. Grab once through the clip and
                # decode only selected frames; repeated random seeks are slow.
                if int(idx) < current_index:
                    raise ValueError("frame indices must be monotonic")
                ok = True
                while current_index < int(idx):
                    ok = cap.grab()
                    current_index += 1
                    if not ok:
                        break
                if not ok:
                    break
                ok, frame = cap.retrieve()
                if not ok or frame is None:
                    continue
                faces = tools.detect(frame)
                face_count[j] = len(faces)
                target, score, gap = match_target(
                    faces, item["speaker"], tools.prototypes)
                similarity[j], margin[j] = score, gap
                if target is None:
                    continue
                # Crop remains face-focused while preserving expression context.
                crop = faces[target].crop
                if min(crop.shape[:2]) < 12:
                    continue
                crop_rows.append(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                crop_positions.append(j)
    finally:
        cap.release()
    if crop_rows:
        from PIL import Image
        with torch.no_grad():
            for start in range(0, len(crop_rows), 16):
                part = crop_rows[start:start + 16]
                inp = processor(
                    images=[Image.fromarray(x) for x in part],
                    return_tensors="pt").to(device)
                out = clip.get_image_features(**inp)
                out = out / out.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                for j, vector in zip(crop_positions[start:start + len(part)],
                                     out.cpu().numpy()):
                    features[j] = vector
                    face_valid[j] = True
    return features, face_valid, similarity, margin, face_count, unique


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "valid", "test"], required=True)
    ap.add_argument("--meld", required=True)
    ap.add_argument("--base-pack", required=True)
    ap.add_argument("--prototypes", required=True)
    ap.add_argument("--yunet", required=True)
    ap.add_argument("--sface", required=True)
    ap.add_argument("--clip-snapshot", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shard-index", type=int, required=True)
    ap.add_argument("--shard-count", type=int, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    assert 0 <= args.shard_index < args.shard_count
    assert args.clip_snapshot != "" and Path(args.clip_snapshot).is_dir()
    meld = Path(args.meld)
    base_file = Path(args.base_pack) / (args.split + ".npz")
    base = np.load(base_file, allow_pickle=False)
    ids = [str(x) for x in base["uid"]]
    manifest = json.loads(
        (meld / "manifests" / (args.split + ".json")).read_text())["items"]
    by_id = {str(row["id"]): row for row in manifest}
    assert set(ids) == set(by_id), "feature/manifest UID mismatch"
    rows = [i for i in range(len(ids)) if i % args.shard_count == args.shard_index]
    if args.limit > 0:
        rows = rows[:args.limit]
    proto_file = Path(args.prototypes)
    with np.load(proto_file, allow_pickle=False) as d:
        prototypes = {k: d[k].astype(np.float32) for k in d.files}
    tools = FaceTools(args.yunet, args.sface)
    tools.prototypes = prototypes
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(2)
    clip = CLIPModel.from_pretrained(args.clip_snapshot,
                                     local_files_only=True).to(device).eval()
    processor = CLIPProcessor.from_pretrained(args.clip_snapshot,
                                              local_files_only=True)
    pack = {name: [] for name in (
        "uid", "source_row", "face_feat", "face_valid", "similarity", "margin",
        "n_faces", "unique_frame")}
    for position, source_row in enumerate(rows):
        uid = ids[source_row]
        item = by_id[uid]
        original = {k: base[k][source_row] for k in
                    ("frame_index", "scene_valid")}
        feat, valid, sim, gap, count, unique = extract_item(
            item, original, tools, clip, processor, device)
        for name, value in zip(pack, (uid, source_row, feat, valid, sim, gap,
                                      count, unique)):
            pack[name].append(value)
        if (position + 1) % 200 == 0:
            print(args.split, args.shard_index, position + 1, "/", len(rows),
                  flush=True)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / ("%s_part%d.npz" % (args.split, args.shard_index))
    if output.exists():
        raise FileExistsError(output)
    arrays = {k: np.asarray(v) for k, v in pack.items()}
    np.savez_compressed(output, **arrays)
    meta = {"split": args.split, "shard": args.shard_index,
            "n_shards": args.shard_count, "rows": len(rows),
            "n_face_valid": int(arrays["face_valid"].sum()),
            "base_pack_sha256": digest(base_file),
            "prototypes_sha256": digest(proto_file),
            "yunet_sha256": digest(Path(args.yunet)),
            "sface_sha256": digest(Path(args.sface)),
            "clip_snapshot": args.clip_snapshot,
            "part_sha256": digest(output), "test_read": False}
    output.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
