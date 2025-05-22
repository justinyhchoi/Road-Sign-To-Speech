import cv2
import numpy as np
import onnxruntime

# Initialize ONNX runtime sessions for YOLO and MicroNet
yolo_session = onnxruntime.InferenceSession("yolov12n.onnx", providers=['CUDAExecutionProvider''CPUExecutionProvider'])
micronet_session = onnxruntime.InferenceSession("micronnet_gtsrb.onnx", providers=['CUDAExecutionProvider','CPUExecutionProvider'])

# Grab input/output names for both models
yolo_input_name  = yolo_session.get_inputs()[0].name
yolo_output_name = yolo_session.get_outputs()[0].name
micro_input_name = micronet_session.get_inputs()[0].name
micro_output_name= micronet_session.get_outputs()[0].name

# Preprocessing functions
def preprocess_yolo(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (800, 800))
    tensor = img_resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return tensor

def preprocess_micronet(crop):
    img_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (48, 48))
    tensor = img_resized.astype(np.float32) / 255.0
    tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    return tensor

# Detection threshold
CONF_THRESH = 0.7

# Open video file instead of webcam
cap = cv2.VideoCapture("input1.mp4")
if not cap.isOpened():
    raise RuntimeError("Cannot open video file input1.mp4")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h0, w0 = frame.shape[:2]

    # 1) Run YOLO detection
    yolo_input = preprocess_yolo(frame)
    preds = yolo_session.run([yolo_output_name], {yolo_input_name: yolo_input})[0][0]

    for x1, y1, x2, y2, score, cls in preds:
        if score < CONF_THRESH:
            continue

        # Rescale coords to original frame
        x1o = int(x1 / 800 * w0)
        y1o = int(y1 / 800 * h0)
        x2o = int(x2 / 800 * w0)
        y2o = int(y2 / 800 * h0)

        # 2) Crop and classify with MicroNet
        crop = frame[y1o:y2o, x1o:x2o]
        if crop.size == 0:
            continue

        micro_in = preprocess_micronet(crop)
        micro_out = micronet_session.run([micro_output_name], {micro_input_name: micro_in})[0]
        class_id = int(np.argmax(micro_out, axis=1)[0])
        conf_cls  = float(np.max(micro_out, axis=1)[0])

        # Draw results
        label = f"{class_id} ({conf_cls:.2f})"
        cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), (0, 255, 0), 2)
        cv2.putText(frame, label, (x1o, y1o - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Detection + Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
