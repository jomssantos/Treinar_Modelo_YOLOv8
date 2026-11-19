import sys
import types
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture
def make_dataset(tmp_path):
    """Cria um dataset pequeno e válido em uma pasta temporária."""

    def _make(phases=("train", "val"), per_phase=2, with_labels=True):
        root = tmp_path / "dataset"
        for phase in phases:
            (root / "images" / phase).mkdir(parents=True)
            (root / "labels" / phase).mkdir(parents=True)
            for i in range(per_phase):
                cv2.imwrite(str(root / "images" / phase / f"img{i}.png"), np.zeros((100, 200, 3), np.uint8))
                if with_labels:
                    (root / "labels" / phase / f"img{i}.txt").write_text("0 0.5 0.5 0.2 0.4\n", encoding="utf-8")
        return root

    return _make


@pytest.fixture
def fake_ultralytics(monkeypatch):
    """Troca a biblioteca ultralytics por uma versão falsa que só registra as chamadas."""
    calls = {}

    class FakeBoxes:
        cls = [0, 1]
        conf = [0.91, 0.55]

    class FakeResult:
        boxes = FakeBoxes()
        save_dir = "runs/detect/predict"

        def plot(self):
            return np.zeros((10, 10, 3), np.uint8)

    class FakeYOLO:
        names = {0: "gato", 1: "cachorro"}

        def __init__(self, path):
            calls["model"] = path
            self.trainer = types.SimpleNamespace(save_dir="runs/detect/modelo")

        def train(self, **kwargs):
            calls["train"] = kwargs

        def predict(self, **kwargs):
            calls["predict"] = kwargs
            return [FakeResult()]

    module = types.ModuleType("ultralytics")
    module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", module)
    return calls
