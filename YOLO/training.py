from ultralytics import YOLO

model = YOLO('yolov12n.pt')

# Train the model
results = model.train(
  data='./datasets/iv.yaml',
  epochs=100, 
  batch=16,
  imgsz=800,
  mosaic=1.0,
  mixup=0,  # S:0.05; M:0.15; L:0.15; X:0.2
  device="0",
  optimizer='SGD',
  momentum=0.937,
  weight_decay=0.0005,
  lr0=0.001,
  lrf=0.0001,
  degrees=90,
  translate=1,
  scale=0.5,
  shear=10,
  perspective=0.001,
  cos_lr=True,
  patience=30,
  fliplr=0,
)

# Evaluate model performance on the validation set
metrics = model.val()