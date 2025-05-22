from ultralytics import YOLO

model = YOLO('./runs/detect/train10/weights/best.pt')
model.export(format="engine", data='/datasets/data.yaml', simplify=True, nms=True)  # or format="onnx"