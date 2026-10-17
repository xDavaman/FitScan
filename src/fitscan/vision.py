"""Bildanalyse pro Videobild: Körperpunkte, Personenmaske, Blickrichtung.

Im Windows-Prototyp übernimmt das MediaPipe (Pose Landmarker). Auf dem iPhone ersetzt
Apples Vision-Framework diesen Teil; die Schnittstelle (FrameObservation) bleibt gleich.
"""

from __future__ import annotations

import dataclasses
import math
import os
import urllib.request
from pathlib import Path

import numpy as np

POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
)

# MediaPipe-Indizes
NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST = 0, 11, 12, 13, 14, 15, 16
L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANKLE, R_ANKLE, L_HEEL, R_HEEL = 23, 24, 25, 26, 27, 28, 29, 30
L_FOOT_INDEX, R_FOOT_INDEX = 31, 32  # Zehenspitze
FACE = list(range(0, 11))


def fitscan_home() -> Path:
    home = Path(os.environ.get("FITSCAN_HOME", Path.cwd() / "daten"))
    home.mkdir(parents=True, exist_ok=True)
    return home


@dataclasses.dataclass
class FrameObservation:
    index: int
    time_s: float
    width: int
    height: int
    persons: int = 0
    keypoints: np.ndarray | None = None  # (33, 3): x, y in Pixeln, Sichtbarkeit 0..1
    world: np.ndarray | None = None  # (33, 3) Meter, Hüftmitte als Ursprung
    mask: np.ndarray | None = None  # (H, W) float32 0..1

    @property
    def yaw_estimate(self) -> float | None:
        """Grobe Drehung der Person (Radiant): 0 = schaut in die Kamera, +90° = rechte Seite sichtbar."""
        if self.world is None:
            return None
        d = self.world[L_SHOULDER] - self.world[R_SHOULDER]
        yaw = math.atan2(d[2], d[0])
        if self.keypoints is not None and np.mean(self.keypoints[FACE, 2]) < 0.3 and abs(yaw) < math.pi / 2:
            yaw = yaw + math.pi if yaw <= 0 else yaw - math.pi  # Gesicht unsichtbar: Rückenansicht
        return yaw


def ensure_pose_model() -> Path:
    target = fitscan_home() / "modelle" / "pose_landmarker_heavy.task"
    if not target.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".part")
        urllib.request.urlretrieve(POSE_MODEL_URL, tmp)
        tmp.replace(target)
    return target


class PoseAnalyzer:
    """Körperpunkte und Maske pro Bild mit MediaPipe."""

    def __init__(self, video_mode: bool = False):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_pose_model())),
            running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
            num_poses=2,
            output_segmentation_masks=True,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
        )
        self._video_mode = video_mode
        self._landmarker = vision.PoseLandmarker.create_from_options(options)

    def analyze(self, rgb: np.ndarray, index: int = 0, time_s: float = 0.0) -> FrameObservation:
        h, w = rgb.shape[:2]
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        if self._video_mode:
            result = self._landmarker.detect_for_video(image, int(time_s * 1000))
        else:
            result = self._landmarker.detect(image)
        obs = FrameObservation(index=index, time_s=time_s, width=w, height=h)
        obs.persons = len(result.pose_landmarks)
        if not result.pose_landmarks:
            return obs
        # Bei mehreren Personen die größte nehmen (die Prüfung meldet das trotzdem)
        sizes = [max(p.y for p in lm) - min(p.y for p in lm) for lm in result.pose_landmarks]
        best = int(np.argmax(sizes))
        lms = result.pose_landmarks[best]
        obs.keypoints = np.array([[p.x * w, p.y * h, p.visibility or 0.0] for p in lms], np.float32)
        obs.world = np.array([[p.x, p.y, p.z] for p in result.pose_world_landmarks[best]], np.float32)
        if result.segmentation_masks:
            obs.mask = result.segmentation_masks[best].numpy_view().astype(np.float32).reshape(h, w)
        return obs

    def close(self) -> None:
        self._landmarker.close()
