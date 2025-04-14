from ultralytics import YOLO


def main():
    MODEL_PTH = "$HOME/Downloads/yolo11n-obb.pt"
    YAML_PTH = "dataset.yaml"

    model = YOLO(MODEL_PTH)

    train_results = model.train(data=YAML_PTH, epochs=100, imgsz=800, device="cuda")

    model.export(format="onnx")


if __name__ == "__main__":
    main()
