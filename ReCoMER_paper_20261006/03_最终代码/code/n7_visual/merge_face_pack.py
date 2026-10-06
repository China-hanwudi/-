"""Merge verified face shards with the existing scene pack, preserving IDs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["train", "valid", "test"], required=True)
    ap.add_argument("--base-pack", required=True)
    ap.add_argument("--shards", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--scene-out-dir", required=True)
    ap.add_argument("--duplicate-scene-out-dir", required=True)
    ap.add_argument("--shard-count", type=int, required=True)
    args = ap.parse_args()
    base_file = Path(args.base_pack) / (args.split + ".npz")
    base = np.load(base_file, allow_pickle=False)
    ids = [str(x) for x in base["uid"]]
    n = len(ids)
    face = np.zeros_like(base["face_feat"])
    face_valid = np.zeros_like(base["face_valid"])
    sim = np.zeros((n, 16), dtype=np.float32)
    margin = np.zeros((n, 16), dtype=np.float32)
    n_faces = np.zeros((n, 16), dtype=np.int16)
    unique = np.zeros_like(base["unique_frame"])
    seen = np.zeros(n, dtype=np.bool_)
    shard_hashes = {}
    for j in range(args.shard_count):
        part = Path(args.shards) / ("%s_part%d.npz" % (args.split, j))
        with np.load(part, allow_pickle=False) as d:
            ix = d["source_row"].astype(np.int64)
            assert np.all((ix >= 0) & (ix < n))
            assert not seen[ix].any(), "duplicate source row"
            assert [str(x) for x in d["uid"]] == [ids[k] for k in ix]
            seen[ix] = True
            face[ix] = d["face_feat"]
            face_valid[ix] = d["face_valid"]
            sim[ix] = d["similarity"]
            margin[ix] = d["margin"]
            n_faces[ix] = d["n_faces"]
            unique[ix] = d["unique_frame"]
        shard_hashes[part.name] = sha(part)
    assert seen.all(), "missing source rows: %d" % int((~seen).sum())
    assert np.all(face_valid <= (base["scene_valid"] & unique))
    assert np.isfinite(face).all()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / (args.split + ".npz")
    if out_file.exists():
        raise FileExistsError(out_file)
    output = {k: base[k] for k in base.files}
    output.update({"face_feat": face, "face_valid": face_valid,
                   "unique_frame": unique, "match_confidence": sim,
                   "match_margin": margin, "n_faces": n_faces})
    np.savez_compressed(out_file, **output)
    scene_dir = Path(args.scene_out_dir)
    scene_dir.mkdir(parents=True, exist_ok=True)
    scene_file = scene_dir / (args.split + ".npz")
    if scene_file.exists():
        raise FileExistsError(scene_file)
    scene_output = dict(output)
    scene_output["face_feat"] = np.zeros_like(face)
    scene_output["face_valid"] = np.zeros_like(face_valid)
    np.savez_compressed(scene_file, **scene_output)
    duplicate_dir = Path(args.duplicate_scene_out_dir)
    duplicate_dir.mkdir(parents=True, exist_ok=True)
    duplicate_file = duplicate_dir / (args.split + ".npz")
    if duplicate_file.exists():
        raise FileExistsError(duplicate_file)
    duplicate_output = dict(output)
    duplicate_output["face_feat"] = np.where(
        face_valid[..., None], base["scene_feat"], 0.0).astype(np.float32)
    np.savez_compressed(duplicate_file, **duplicate_output)
    report = {"split": args.split, "rows": n,
              "source_pack_sha256": sha(base_file),
              "shard_sha256": shard_hashes,
              "output_sha256": sha(out_file),
              "scene_only_sha256": sha(scene_file),
              "duplicate_scene_sha256": sha(duplicate_file),
              "face_valid_frames": int(face_valid.sum()),
              "face_valid_rows": int(face_valid.any(axis=1).sum()),
              "source_unique_frames": int(unique.sum()),
              "test_read": False}
    out_file.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
