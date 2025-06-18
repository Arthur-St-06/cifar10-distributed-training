from ultralytics import YOLO
import torch

if torch.cuda.is_available():
    print("PyTorch is using a GPU")
else:
    print("PyTorch is not using a GPU")

# Load a smaller model (you’re already using yolov8n – that's good)
model = YOLO("yolov8n.yaml")

if __name__ == '__main__':
    results = model.train(
        data="config.yaml",
        epochs=500,
        device='0',
        workers=4
    )
