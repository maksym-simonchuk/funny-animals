"""Tests for src.processors.detector.AnimalDetector. YOLO is mocked - no real inference,
no network, no model download. Uses the tmp_cfg fixture from tests/conftest.py.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from src.processors.detector import AnimalDetector


class _FakeBox:
    def __init__(self, cls_idx: int, conf: float, xyxy: tuple[float, float, float, float]) -> None:
        self.cls = [cls_idx]
        self.conf = [conf]
        self.xyxy = [xyxy]


class _FakePrediction:
    def __init__(self, names: dict[int, str], boxes: list[_FakeBox]) -> None:
        self.names = names
        self.boxes = boxes


class _FakeYOLO:
    """Stand-in for ultralytics.YOLO. Tracks instantiation count and returns scripted predictions."""

    instances = 0
    predictions: list[_FakePrediction] = []

    def __init__(self, weights: str) -> None:
        _FakeYOLO.instances += 1
        self.weights = weights

    def __call__(self, frame_path: str, verbose: bool = False):
        return _FakeYOLO.predictions


@pytest.fixture(autouse=True)
def _reset_fake_yolo(monkeypatch: pytest.MonkeyPatch) -> None:
    _FakeYOLO.instances = 0
    _FakeYOLO.predictions = []
    monkeypatch.setattr("src.processors.detector.YOLO", _FakeYOLO)


_NAMES = {0: "cat", 1: "car", 2: "dog"}


def test_detect_filters_by_confidence_threshold(tmp_cfg, tmp_path: Path) -> None:
    _FakeYOLO.predictions = [
        _FakePrediction(_NAMES, [_FakeBox(0, 0.9, (10, 10, 50, 60)), _FakeBox(0, 0.2, (0, 0, 5, 5))])
    ]
    detector = AnimalDetector(tmp_cfg)
    results = detector.detect([(0.0, tmp_path / "f1.jpg")])
    assert len(results) == 1
    assert results[0].conf == 0.9


def test_detect_filters_by_class_allowlist(tmp_cfg, tmp_path: Path) -> None:
    _FakeYOLO.predictions = [
        _FakePrediction(_NAMES, [_FakeBox(1, 0.9, (0, 0, 10, 10))])  # "car" is not an animal class
    ]
    detector = AnimalDetector(tmp_cfg)
    results = detector.detect([(0.0, tmp_path / "f1.jpg")])
    assert results == []


def test_detect_converts_bbox_to_pixel_xywh(tmp_cfg, tmp_path: Path) -> None:
    _FakeYOLO.predictions = [_FakePrediction(_NAMES, [_FakeBox(0, 0.9, (10, 20, 50, 80))])]
    detector = AnimalDetector(tmp_cfg)
    [result] = detector.detect([(1.5, tmp_path / "f1.jpg")])
    assert result.class_name == "cat"
    assert result.ts_s == 1.5
    assert (result.x, result.y, result.w, result.h) == (10.0, 20.0, 40.0, 60.0)


def test_detect_no_boxes_returns_empty(tmp_cfg, tmp_path: Path) -> None:
    _FakeYOLO.predictions = [_FakePrediction(_NAMES, [])]
    detector = AnimalDetector(tmp_cfg)
    assert detector.detect([(0.0, tmp_path / "f1.jpg")]) == []


def test_summarize_most_common_class_and_avg_conf(tmp_cfg) -> None:
    detector = AnimalDetector(tmp_cfg)
    results = [
        _make_result("cat", 0.8),
        _make_result("cat", 0.6),
        _make_result("dog", 0.9),
    ]
    category, conf = detector.summarize(results)
    assert category == "cat"
    assert conf == pytest.approx(0.7)


def test_summarize_empty_returns_none_and_zero(tmp_cfg) -> None:
    detector = AnimalDetector(tmp_cfg)
    assert detector.summarize([]) == (None, 0.0)


def test_model_instantiated_once_across_many_videos(tmp_cfg, tmp_path: Path) -> None:
    _FakeYOLO.predictions = [_FakePrediction(_NAMES, [_FakeBox(0, 0.9, (0, 0, 10, 10))])]
    detector = AnimalDetector(tmp_cfg)

    for i in range(20):
        detector.detect([(0.0, tmp_path / f"video{i}_frame.jpg")])

    assert _FakeYOLO.instances == 1


def _make_result(class_name: str, conf: float):
    from src.processors.detector import DetectionResult

    return DetectionResult(ts_s=0.0, class_name=class_name, conf=conf, x=0.0, y=0.0, w=1.0, h=1.0)
