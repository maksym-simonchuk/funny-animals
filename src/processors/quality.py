"""Frame quality assessment: sharpness, brightness and resolution gating."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2

if TYPE_CHECKING:
    from src.config import Config


@dataclass(frozen=True)
class QualityResult:
    sharpness: float
    brightness: float
    ok: bool
    reason: str | None


def assess(frames: list[Path], cfg: "Config") -> QualityResult:
    """Average sharpness (Laplacian variance) / brightness (HSV V-channel) across `frames`.

    Checks run in a fixed order (sharpness -> brightness -> resolution); the first
    violated threshold from cfg.processing.quality becomes `reason`.
    """
    sharpness_values: list[float] = []
    brightness_values: list[float] = []
    heights: list[int] = []

    for frame_path in frames:
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        sharpness_values.append(float(cv2.Laplacian(gray, cv2.CV_64F).var()))

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        brightness_values.append(float(hsv[:, :, 2].mean()))

        heights.append(image.shape[0])

    sharpness = sum(sharpness_values) / len(sharpness_values) if sharpness_values else 0.0
    brightness = sum(brightness_values) / len(brightness_values) if brightness_values else 0.0

    quality_cfg = cfg.processing.quality
    reason: str | None = None
    if sharpness < quality_cfg.min_sharpness:
        reason = "low_sharpness"
    elif brightness < quality_cfg.min_brightness:
        reason = "dark"
    elif not heights or min(heights) < quality_cfg.min_resolution:
        reason = "low_resolution"

    return QualityResult(sharpness=sharpness, brightness=brightness, ok=reason is None, reason=reason)
