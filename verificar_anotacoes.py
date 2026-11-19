# uso: python verificar_anotacoes.py [--ver train] [--salvar pasta]
import argparse
import sys
from pathlib import Path

import cv2

from yolo_utils import PHASES, list_images, load_classes, parse_label_line, validate_dataset

COLORS = [(75, 25, 230), (75, 180, 60), (25, 225, 255), (83, 113, 224), (49, 130, 245), (180, 30, 145)]


def print_report(dataset: str, classes: list[str]) -> bool:
    report = validate_dataset(dataset, len(classes) or None)

    for phase in PHASES:
        print(f"{phase}: {report.images.get(phase, 0)} imagem(ns), {report.empty_labels.get(phase, 0)} sem objetos")

    print("\nCaixas por classe:")
    for class_id, total in sorted(report.class_counts.items()):
        name = classes[class_id] if class_id < len(classes) else f"classe {class_id}"
        print(f"  {name}: {total}")
    if not report.class_counts:
        print("  nenhuma")

    for warning in report.warnings:
        print(f"AVISO: {warning}")
    for error in report.errors:
        print(f"ERRO: {error}")

    print("\n" + ("Dataset pronto para treinar." if report.ok else "Corrija os erros antes de treinar."))
    return report.ok


def draw_boxes(image, label_file: Path, classes: list[str]):
    height, width = image.shape[:2]
    for line in label_file.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            class_id, cx, cy, w, h = parse_label_line(line)
        except ValueError:
            continue
        x0, y0 = int((cx - w / 2) * width), int((cy - h / 2) * height)
        x1, y1 = int((cx + w / 2) * width), int((cy + h / 2) * height)
        color = COLORS[class_id % len(COLORS)]
        name = classes[class_id] if class_id < len(classes) else str(class_id)
        cv2.rectangle(image, (x0, y0), (x1, y1), color, 2)
        cv2.putText(image, name, (x0, max(15, y0 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return image


def review(dataset: str, phase: str, classes: list[str], output: Path | None) -> None:
    labels_dir = Path(dataset) / "labels" / phase
    if output:
        output.mkdir(parents=True, exist_ok=True)

    for image_path in list_images(Path(dataset) / "images" / phase):
        label_file = labels_dir / f"{image_path.stem}.txt"
        image = cv2.imread(str(image_path))
        if image is None or not label_file.exists():
            continue

        draw_boxes(image, label_file, classes)
        if output:
            cv2.imwrite(str(output / image_path.name), image)
        else:
            cv2.imshow(f"{phase}: {image_path.name} (tecla = próxima, ESC = sair)", image)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:
                return

    if output:
        print(f"Imagens salvas em: {output}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Confere as anotações do dataset")
    parser.add_argument("--dataset", default="dataset")
    parser.add_argument("--data-yaml", default="data.yaml")
    parser.add_argument("--ver", choices=PHASES, help="mostra as caixas de uma fase")
    parser.add_argument("--salvar", type=Path, help="grava as imagens com caixas nesta pasta")
    args = parser.parse_args(argv)

    classes = load_classes(args.data_yaml)
    ok = print_report(args.dataset, classes)
    if args.ver or args.salvar:
        review(args.dataset, args.ver or "train", classes, args.salvar)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
