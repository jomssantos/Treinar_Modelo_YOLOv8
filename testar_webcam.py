# uso: python testar_webcam.py [--conf 0.4]  |  C salva foto.jpg, ESC ou Q sai
import argparse
import sys

import cv2

DEFAULT_MODEL = "runs/detect/modelo/weights/best.pt"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Testa o modelo na webcam")
    parser.add_argument("--modelo", default=DEFAULT_MODEL)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args(argv)

    from ultralytics import YOLO

    model = YOLO(args.modelo)
    capture = cv2.VideoCapture(args.camera)
    if not capture.isOpened():
        print(f"Não abriu a câmera {args.camera}")
        return 1

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            result = model.predict(source=frame, conf=args.conf, verbose=False)[0]
            cv2.imshow("C = foto, ESC = sair", result.plot())

            key = cv2.waitKey(1) & 0xFF
            if key == ord("c"):
                cv2.imwrite("foto.jpg", frame)  # sem as caixas
                print("Salvo: foto.jpg")
            elif key in (27, ord("q")):
                break
    finally:
        capture.release()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
