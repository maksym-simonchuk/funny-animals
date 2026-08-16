"""Animal detection via YOLOv8n (ultralytics)."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ultralytics import YOLO

if TYPE_CHECKING:
    from src.config import Config


@dataclass(frozen=True)
class DetectionResult:
    ts_s: float
    class_name: str
    conf: float
    x: float
    y: float
    w: float
    h: float


class AnimalDetector:
    """Wraps a YOLOv8n model. The model is loaded once, at construction time — create a
    single instance per `process` run and reuse it across every video."""

    def __init__(self, cfg: "Config") -> None:
        weights = cfg.storage.model_path / cfg.processing.animal_detection.model
        weights.parent.mkdir(parents=True, exist_ok=True)
        self._model = YOLO(str(weights))  # downloads to `weights` on first use if missing
        self._classes = set(cfg.processing.animal_detection.classes)
        self._confidence_threshold = cfg.processing.animal_detection.confidence_threshold

    def detect(self, frames: list[tuple[float, Path]]) -> list[DetectionResult]:
        """Run inference on each (timestamp, frame_path), keeping only configured animal
        classes above the confidence threshold. Boxes are normalized to frame pixels."""
        results: list[DetectionResult] = []
        for ts_s, frame_path in frames:
            for prediction in self._model(str(frame_path), verbose=False):
                boxes = prediction.boxes
                if boxes is None:
                    continue
                names = prediction.names
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf < self._confidence_threshold:
                        continue
                    class_name = names[int(box.cls[0])]
                    if class_name not in self._classes:
                        continue
                    x1, y1, x2, y2 = (float(v) for v in box.xyxy[0])
                    results.append(
                        DetectionResult(
                            ts_s=ts_s, class_name=class_name, conf=conf,
                            x=x1, y=y1, w=x2 - x1, h=y2 - y1,
                        )
                    )
        return results

    def summarize(self, results: list[DetectionResult]) -> tuple[str | None, float]:
        """Most frequent class_name across frames, with its average confidence. Empty -> (None, 0.0)."""
        if not results:
            return None, 0.0
        top_class, _ = Counter(r.class_name for r in results).most_common(1)[0]
        confs = [r.conf for r in results if r.class_name == top_class]
        return top_class, sum(confs) / len(confs)
