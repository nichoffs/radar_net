from argparse import ArgumentParser

from ultralytics import YOLO

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--weights", type=str)
    args = parser.parse_args()

    model = YOLO(args.weights)

    model.export(format="onnx")
