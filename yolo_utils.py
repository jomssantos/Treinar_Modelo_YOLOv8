from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import yaml

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
PHASES = ("train", "val")
MIN_BOX_PIXELS = 3


def list_images(folder: str | Path) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)


def pending_images(images_dir: str | Path, labels_dir: str | Path) -> list[Path]:
    # imagens sem .txt de anotação
    labels_dir = Path(labels_dir)
    return [img for img in list_images(images_dir) if not (labels_dir / f"{img.stem}.txt").exists()]


def to_yolo_line(class_id: int, p1: tuple, p2: tuple, width: int, height: int) -> str | None:
    # "classe cx cy w h", tudo entre 0 e 1
    x_min, x_max = sorted((max(0.0, min(p1[0], width)), max(0.0, min(p2[0], width))))
    y_min, y_max = sorted((max(0.0, min(p1[1], height)), max(0.0, min(p2[1], height))))

    # clique sem arrastar não vira caixa
    if (x_max - x_min) < MIN_BOX_PIXELS or (y_max - y_min) < MIN_BOX_PIXELS:
        return None

    cx = (x_min + x_max) / 2 / width
    cy = (y_min + y_max) / 2 / height
    w = (x_max - x_min) / width
    h = (y_max - y_min) / height
    return f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"


def load_classes(data_yaml: str | Path) -> list[str]:
    path = Path(data_yaml)
    if not path.exists():
        return []
    names = (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("names", [])
    if isinstance(names, dict):
        return [str(names[k]) for k in sorted(names, key=int)]
    return [str(n) for n in names]


def save_data_yaml(data_yaml: str | Path, classes: list[str], dataset_dir: str | Path = "dataset") -> None:
    base = Path(dataset_dir).as_posix()
    content = {"train": f"{base}/images/train", "val": f"{base}/images/val", "nc": len(classes), "names": list(classes)}
    Path(data_yaml).write_text(
        yaml.safe_dump(content, allow_unicode=True, sort_keys=False, default_flow_style=None), encoding="utf-8"
    )


@dataclass
class DatasetReport:
    images: dict = field(default_factory=dict)
    unlabeled: dict = field(default_factory=dict)
    orphan_labels: dict = field(default_factory=dict)
    empty_labels: dict = field(default_factory=dict)  # "sem objetos", válidas
    invalid_lines: list = field(default_factory=list)
    class_counts: Counter = field(default_factory=Counter)

    @property
    def errors(self) -> list[str]:
        found = list(self.invalid_lines)
        for phase, names in self.orphan_labels.items():
            found += [f"[{phase}] anotação sem imagem: {n}" for n in names]
        for phase in PHASES:
            if self.images.get(phase, 0) == 0:
                found.append(f"[{phase}] nenhuma imagem em dataset/images/{phase}")
        return found

    @property
    def warnings(self) -> list[str]:
        return [
            f"[{phase}] {len(names)} imagem(ns) sem anotação (viram fundo)"
            for phase, names in self.unlabeled.items()
            if names
        ]

    @property
    def ok(self) -> bool:
        return not self.errors


def parse_label_line(line: str, num_classes: int | None = None) -> tuple[int, float, float, float, float]:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"esperava 5 valores e achou {len(parts)}")
    try:
        class_id = int(parts[0])
        cx, cy, w, h = (float(p) for p in parts[1:])
    except ValueError:
        raise ValueError("valores não numéricos")
    if class_id < 0 or (num_classes is not None and class_id >= num_classes):
        raise ValueError(f"classe {class_id} fora do intervalo do data.yaml")
    if not all(0 <= v <= 1 for v in (cx, cy, w, h)):
        raise ValueError("coordenadas fora do intervalo 0 a 1")
    if w <= 0 or h <= 0:
        raise ValueError("caixa sem largura ou altura")
    return class_id, cx, cy, w, h


def validate_dataset(dataset_dir: str | Path = "dataset", num_classes: int | None = None) -> DatasetReport:
    root = Path(dataset_dir)
    report = DatasetReport()

    for phase in PHASES:
        images = list_images(root / "images" / phase)
        labels_dir = root / "labels" / phase
        label_files = sorted(labels_dir.glob("*.txt")) if labels_dir.exists() else []
        stems = {img.stem for img in images}

        report.images[phase] = len(images)
        report.unlabeled[phase] = [img.name for img in images if not (labels_dir / f"{img.stem}.txt").exists()]
        report.orphan_labels[phase] = [lf.name for lf in label_files if lf.stem not in stems]
        report.empty_labels[phase] = 0

        for label_file in label_files:
            lines = [ln for ln in label_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
            if not lines and label_file.stem in stems:
                report.empty_labels[phase] += 1
            for number, line in enumerate(lines, start=1):
                try:
                    class_id, *_ = parse_label_line(line, num_classes)
                    report.class_counts[class_id] += 1
                except ValueError as exc:
                    report.invalid_lines.append(f"[{phase}] {label_file.name} linha {number}: {exc}")
    return report
