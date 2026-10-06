"""Verify and join sparse video evidence onto the immutable scene feature pack."""
import argparse
import json
from pathlib import Path

import numpy as np

from n7_visual.stamp_run_inputs import sha


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--split", choices=("train", "valid", "test"), required=True)
    args = ap.parse_args()
    base_file = args.root / "features/vf16_face_v2/scene" / (args.split + ".npz")
    base = np.load(base_file, allow_pickle=False)
    n = len(base["uid"])
    merged = {k: base[k] for k in base.files}
    merged.update({k: np.zeros((n, 512), np.float32) for k in ("dynamic", "static", "clip")})
    merged["evidence_valid"] = np.zeros(n, bool)
    merged["unique_count"] = np.zeros(n, np.int16)
    merged["evidence_source_indices"] = np.full((n, 16), -1, np.int32)
    seen = np.zeros(n, bool)
    hashes = {}
    for shard in range(6):
        f = args.root / "features/avf_pilot_v1/shards" / ("%s_part%d.npz" % (args.split, shard))
        data = np.load(f, allow_pickle=False)
        ix = data["source_row"]
        assert np.all(ix >= 0) and np.all(ix < n) and not seen[ix].any()
        assert np.array_equal(base["uid"][ix], data["uid"])
        seen[ix] = True
        for key in ("dynamic", "static", "clip"):
            assert np.isfinite(data[key]).all()
            assert np.all(data[key][~data["valid"]] == 0)
            norms = np.linalg.norm(data[key][data["valid"]], axis=1)
            assert np.allclose(norms, 1, atol=1e-5)
            merged[key][ix] = data[key]
        merged["evidence_valid"][ix] = data["valid"]
        merged["unique_count"][ix] = data["unique_count"]
        merged["evidence_source_indices"][ix] = data["source_indices"]
        hashes[f.name] = sha(f)
    assert seen.all()
    valid = merged["evidence_valid"]
    assert np.all(merged["unique_count"][valid] >= 4)
    output = args.root / "features/avf_pilot_v1/packed" / (args.split + ".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    np.savez_compressed(output, **merged)
    similarities = np.sum(merged["dynamic"][valid] * merged["static"][valid], axis=1)
    report = {"split": args.split, "rows": n, "valid_rows": int(valid.sum()),
              "dynamic_static_cosine_mean": float(similarities.mean()),
              "dynamic_static_cosine_quantiles": np.quantile(similarities, [.1,.5,.9]).tolist(),
              "source_scene_sha256": sha(base_file), "shard_sha256": hashes,
              "output_sha256": sha(output), "test_read": False}
    output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
