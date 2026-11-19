# uso: python testar_modelo.py --imagem foto.jpg [--conf 0.3] [--mostrar]
import argparse
import sys

DEFAULT_MODEL = "runs/detect/modelo/weights/best.pt"


def summarize(result, names: dict) -> list[str]:
    return [f"{names[int(c)]} {float(s):.2f}" for c, s in zip(result.boxes.cls, result.boxes.conf)]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Testa o modelo em uma imagem")
    parser.add_argument("--imagem", required=True)
    parser.add_argument("--modelo", default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.5, help="confiança mínima de 0 a 1")
    parser.add_argument("--mostrar", action="store_true", help="abre janela com o resultado")
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(args.modelo)
    for result in model.predict(source=args.imagem, conf=args.conf, save=True, show=args.mostrar):
        detections = summarize(result, model.names)
        print(f"{len(detections)} detecção(ões):")
        for line in detections:
            print(f"  {line}")
        print(f"Salvo em: {result.save_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
