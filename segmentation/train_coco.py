from ultralytics import YOLO
import os

# Load downloaded model
model = YOLO("yolo11m-seg.pt")

# Train the seg model
results = model.train(data="data.yaml", epochs=20, imgsz=640, batch=16, workers=8, device="mps", save=True, freeze=10, lr0=0.0005)
