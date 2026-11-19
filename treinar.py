# uso: python treinar.py [--nome meu_detector --epocas 100 --imgsz 416 --device cpu]
import argparse
import sys

from yolo_utils import load_classes, validate_dataset


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Treina um modelo YOLOv8")
    parser.add_argument("--modelo", default="yolov8n.pt", help="modelo base, baixado na primeira vez")
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--dataset", default="dataset", help="pasta usada na verificação prévia")
    parser.add_argument("--epocas", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=320)
    parser.add_argument("--batch", type=int, default=16, help="diminua se faltar memória")
    parser.add_argument("--device", default=None, help="cpu ou número da GPU (0)")
    parser.add_argument("--nome", default="modelo", help="pasta de resultados em runs/detect")
    parser.add_argument("--pular-verificacao", action="store_true")
    args = parser.parse_args(argv)

    if not args.pular_verificacao:
        report = validate_dataset(args.dataset, len(load_classes(args.data)) or None)
        if not report.ok:
            print("Dataset com problemas, treino não iniciado:\n")
            for error in report.errors[:20]:
                print(f"  - {error}")
            print("\nRode `python verificar_anotacoes.py` para o relatório completo.")
            return 1

    # import aqui deixa o --help rápido
    from ultralytics import YOLO

    train_args = dict(data=args.data, epochs=args.epocas, imgsz=args.imgsz, batch=args.batch, name=args.nome)
    if args.device is not None:
        train_args["device"] = args.device

    model = YOLO(args.modelo)
    model.train(**train_args)
    print(f"\nPronto. Melhor modelo: {model.trainer.save_dir}/weights/best.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
