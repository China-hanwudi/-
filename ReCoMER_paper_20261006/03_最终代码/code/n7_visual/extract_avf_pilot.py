"""Matched sparse-face video/static/CLIP controls, train/valid only.

Uses the longest run of adjacent valid sampling slots (at least four distinct
source frames). Slots are sparse in source time: this is not dense tracking.
Repeated padding to 16 encoder frames is recorded, not counted as new evidence.
"""
import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from n7_visual.avf_encoder import build_encoder, encode
from n7_visual.face_identity import FaceTools, match_target
from n7_visual.stamp_run_inputs import sha


def longest_run(mask):
    best, current = [], []
    for j, value in enumerate(mask):
        if value:
            current.append(j)
            if len(current) > len(best):
                best = current[:]
        else:
            current = []
    return best


def read_crops(item, source_indices, tools, prototypes):
    wanted = set(int(x) for x in source_indices)
    cap = cv2.VideoCapture(item["video_path"])
    found, boxes = {}, {}
    try:
        if not cap.isOpened():
            return None, None
        for idx in range(max(wanted) + 1):
            if not cap.grab():
                break
            if idx not in wanted:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                continue
            faces = tools.detect(frame)
            target, _, _ = match_target(faces, item["speaker"], prototypes)
            if target is None:
                continue
            crop = faces[target].crop
            if min(crop.shape[:2]) < 12:
                continue
            found[idx] = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB),
                                    (160, 160), interpolation=cv2.INTER_LINEAR)
            boxes[idx] = faces[target].box
    finally:
        cap.release()
    if set(found) != wanted:
        return None, None
    return (np.stack([found[int(k)] for k in source_indices]),
            np.stack([boxes[int(k)] for k in source_indices]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--meld", type=Path, required=True)
    ap.add_argument("--split", choices=("train", "valid", "test"), required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, default=6)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    torch.set_num_threads(2)
    pack_path = args.root / "features/vf16_face_v2/face" / (args.split + ".npz")
    base = np.load(pack_path, allow_pickle=False)
    manifest = json.loads((args.meld / "manifests" / (args.split + ".json")).read_text())["items"]
    items = {str(x["id"]): x for x in manifest}
    proto_path = args.root / "audit/train_prototypes_4000.npz"
    proto = np.load(proto_path, allow_pickle=False)
    prototypes = {k: proto[k] for k in proto.files}
    tools = FaceTools(str(args.root / "assets/yunet/face_detection_yunet_2023mar.onnx"),
                      str(args.root / "assets/sface/face_recognition_sface_2021dec.onnx"))
    weights = args.root / "assets/avfmae/visual_encoder.pt"
    device = torch.device("cuda")
    encoder = build_encoder(args.root / "vendor/AVF-MAE", weights, device)
    indices = [i for i in range(len(base["uid"])) if i % args.nshards == args.shard]
    if args.limit:
        indices = indices[:args.limit]
    n = len(indices)
    output = {"source_row": np.array(indices), "uid": base["uid"][indices],
              "dynamic": np.zeros((n, 512), np.float32),
              "static": np.zeros((n, 512), np.float32),
              "clip": np.zeros((n, 512), np.float32),
              "valid": np.zeros(n, bool), "unique_count": np.zeros(n, np.int16),
              "source_indices": np.full((n, 16), -1, np.int32),
              "boxes": np.zeros((n, 16, 4), np.int32)}
    queue, queued_rows = [], []
    mean = torch.tensor([.485, .456, .406], device=device).view(1, 3, 1, 1, 1)
    std = torch.tensor([.229, .224, .225], device=device).view(1, 3, 1, 1, 1)

    def flush():
        if not queue:
            return
        # Dynamic and repeated central-frame controls share the same crops.
        x = torch.from_numpy(np.stack(queue)).to(device).float().permute(0, 4, 1, 2, 3) / 255.
        x = (x - mean) / std
        static = x[:, :, 8:9].expand(-1, -1, 16, -1, -1)
        emb = encode(encoder, torch.cat((x, static), 0)).cpu().numpy()
        assert np.isfinite(emb).all()
        for j, row in enumerate(queued_rows):
            output["dynamic"][row] = emb[j]
            output["static"][row] = emb[j + len(queue)]
            output["valid"][row] = True
        queue.clear()
        queued_rows.clear()

    for row, source_row in enumerate(indices):
        obs = base["face_valid"][source_row] & base["unique_frame"][source_row]
        selected = longest_run(obs)
        if len(selected) >= 4:
            source = base["frame_index"][source_row, selected]
            assert len(np.unique(source)) == len(source)
            crops, boxes = read_crops(items[str(base["uid"][source_row])], source,
                                      tools, prototypes)
            if crops is not None:
                padding = np.rint(np.linspace(0, len(selected) - 1, 16)).astype(int)
                queue.append(crops[padding])
                queued_rows.append(row)
                output["source_indices"][row] = source[padding]
                output["boxes"][row] = boxes[padding]
                output["unique_count"][row] = len(selected)
                clip = base["face_feat"][source_row, selected].mean(0)
                output["clip"][row] = clip / max(float(np.linalg.norm(clip)), 1e-8)
        if len(queue) >= 8:
            flush()
        if (row + 1) % 200 == 0:
            print(args.split, args.shard, row + 1, n, flush=True)
    flush()
    assert np.all(output["unique_count"][output["valid"]] >= 4)
    args.out.mkdir(parents=True, exist_ok=True)
    dest = args.out / ("%s_part%d.npz" % (args.split, args.shard))
    if dest.exists():
        raise FileExistsError(dest)
    np.savez_compressed(dest, **output)
    meta = {"split": args.split, "rows": n, "valid_rows": int(output["valid"].sum()),
            "pack_sha256": sha(pack_path), "prototypes_sha256": sha(proto_path),
            "encoder_sha256": sha(weights), "output_sha256": sha(dest),
            "script_sha256": sha(Path(__file__)),
            "test_read": False,
            "limits": "sparse source-time frames; metadata prototypes not human-audited"}
    dest.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta), flush=True)


if __name__ == "__main__":
    main()
