"""Conservative face identity evidence for MELD train/valid.

YuNet detects faces and SFace embeds them. MELD's utterance speaker field is
used only to build train-side identity prototypes. A detected face is marked
as the target only when similarity and separation both pass locked thresholds.
Otherwise the target face is UNKNOWN; scene evidence remains available.

The prototypes are weakly supervised because a single visible face can be a
listener. They require manual audit before a target-speaker method claim.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np


@dataclass
class FaceObservation:
    box: np.ndarray
    score: float
    embedding: np.ndarray
    crop: np.ndarray


def unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / max(float(np.linalg.norm(x)), 1e-8)


class FaceTools:
    def __init__(self, detector_path: str, recognizer_path: str,
                 detection_score: float = 0.75):
        # Several parallel shards otherwise spawn tens of OpenCV threads each
        # and spend most wall time in CPU oversubscription.
        cv2.setNumThreads(2)
        for path in (detector_path, recognizer_path):
            if not Path(path).is_file():
                raise FileNotFoundError(path)
        self.detector = cv2.FaceDetectorYN_create(
            detector_path, "", (320, 320), score_threshold=detection_score,
            nms_threshold=0.3, top_k=1000)
        self.recognizer = cv2.FaceRecognizerSF_create(recognizer_path, "")

    def detect(self, frame: np.ndarray) -> List[FaceObservation]:
        height, width = frame.shape[:2]
        _, faces = self.detector.detect(cv2.resize(frame, (320, 320)))
        if faces is None:
            return []
        output = []
        for raw in faces:
            face = raw.copy()
            face[[0, 2, 4, 6, 8, 10, 12]] *= width / 320.0
            face[[1, 3, 5, 7, 9, 11, 13]] *= height / 320.0
            x, y, w, h = face[:4]
            if w <= 0 or h <= 0:
                continue
            x0 = max(0, int(x - 0.15 * w))
            y0 = max(0, int(y - 0.15 * h))
            x1 = min(width, int(x + 1.15 * w))
            y1 = min(height, int(y + 1.15 * h))
            if x1 <= x0 or y1 <= y0:
                continue
            try:
                aligned = self.recognizer.alignCrop(frame, face)
                embedding = unit(self.recognizer.feature(aligned).reshape(-1))
            except cv2.error:
                continue
            output.append(FaceObservation(
                box=np.array([x0, y0, x1, y1], dtype=np.int32),
                score=float(face[-1]), embedding=embedding,
                crop=frame[y0:y1, x0:x1]))
        return output


def read_frame(video_path: str, fraction: float = 0.5) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    try:
        if not cap.isOpened():
            return None
        count = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        cap.set(cv2.CAP_PROP_POS_FRAMES, round((count - 1) * fraction))
        ok, frame = cap.read()
        return frame if ok else None
    finally:
        cap.release()


def robust_prototypes(rows: List[Tuple[str, np.ndarray]], min_rows: int = 12,
                      cluster_cosine: float = 0.58) -> Dict[str, np.ndarray]:
    """Use the densest same-label cluster, not the mean of noisy singleton rows."""
    by_speaker: Dict[str, List[np.ndarray]] = {}
    for speaker, emb in rows:
        by_speaker.setdefault(speaker, []).append(unit(emb))
    out = {}
    for speaker, vectors in by_speaker.items():
        if len(vectors) < min_rows:
            continue
        x = np.stack(vectors)
        sims = x @ x.T
        neighbors = sims >= cluster_cosine
        medoid = int(neighbors.sum(1).argmax())
        members = x[neighbors[medoid]]
        if len(members) < min_rows:
            continue
        out[speaker] = unit(members.mean(0))
    return out


def match_target(faces: List[FaceObservation], speaker: str,
                 prototypes: Dict[str, np.ndarray],
                 min_similarity: float = 0.55,
                 min_margin: float = 0.08) -> Tuple[Optional[int], float, float]:
    """Return UNKNOWN on weak or ambiguous evidence.

    The margin is against both another face for the requested speaker and the
    best *different* speaker prototype for the selected face.
    """
    if not faces or speaker not in prototypes:
        return None, 0.0, 0.0
    scores = np.array([float(f.embedding @ prototypes[speaker]) for f in faces])
    best = int(scores.argmax())
    target_score = float(scores[best])
    other_face = max((float(v) for j, v in enumerate(scores) if j != best),
                     default=-1.0)
    other_identity = max(
        (float(faces[best].embedding @ proto) for name, proto in
         prototypes.items() if name != speaker), default=-1.0)
    margin = target_score - max(other_face, other_identity)
    if target_score < min_similarity or margin < min_margin:
        return None, target_score, margin
    return best, target_score, margin
