"""
pipeline.py  –  Real‑time traffic‑sign detection (YOLO‑ONNX) + recognition (Micronet‑PyTorch)

Usage
-----
$ python pipeline.py --onnx yolov12s_gtsrb.onnx --micronet micronet_gtsrb.pth --classes gtsrb.yaml \
                     --source 0  # 0 = webcam, or path to video file

The script loads a YOLO object‑detector exported to ONNX and a Micronet classifier
(.pth) trained on 43 GTSRB classes.  For every frame it:
  1. runs YOLO → bounding‑boxes of candidate signs;
  2. crops each detected sign region and feeds it to Micronet → class label;
  3. overlays results on the frame and shows / writes to disk.

Requirements
------------
Python 3.9+, onnxruntime‑gpu (or cpu), torch, torchvision, opencv‑python,
pyyaml (for the class‑names file).

Tip:  For best speed install onnxruntime‑gpu ≥1.16 and enable CUDA.
"""

from pathlib import Path
import argparse
import time
import yaml

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torchvision.transforms as T

# ---------------------------------------------
# Utils
# ---------------------------------------------

def letterbox(img, new_size=640, color=(114, 114, 114)):
    """Resize & pad image while meeting stride-multiple constraints."""
    h, w = img.shape[:2]
    scale = new_size / max(h, w)
    nh, nw = int(round(h * scale)), int(round(w * scale))
    img_resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top = (new_size - nh) // 2
    bottom = new_size - nh - top
    left = (new_size - nw) // 2
    right = new_size - nw - left
    img_padded = cv2.copyMakeBorder(img_resized, top, bottom, left, right,
                                    cv2.BORDER_CONSTANT, value=color)
    return img_padded, scale, left, top


def xyxy_rescale(box, scale, pad_left, pad_top):
    """Rescale padded box back to original image coordinates."""
    x1, y1, x2, y2 = box
    x1 = (x1 - pad_left) / scale
    y1 = (y1 - pad_top) / scale
    x2 = (x2 - pad_left) / scale
    y2 = (y2 - pad_top) / scale
    return np.array([x1, y1, x2, y2])


def non_max_suppression(boxes, scores, iou_thresh=0.45):
    """Simple NMS for xyxy boxes on CPU using torchvision."""
    import torchvision.ops as ops
    keep = ops.nms(torch.tensor(boxes), torch.tensor(scores), iou_thresh)
    return keep.numpy()

# ---------------------------------------------
# Main Pipeline Class
# ---------------------------------------------

class TSRPipeline:
    def __init__(self, onnx_path: Path, micronet_path: Path, classes_yaml: Path,
                 det_imgsz: int = 640, conf_thresh: float = 0.25, iou_thresh: float = 0.45,
                 cuda: bool = True):
        self.device = 'cuda' if cuda and torch.cuda.is_available() else 'cpu'
        self.det_size = det_imgsz
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh

        # Load class names
        with open(classes_yaml, 'r') as f:
            names = yaml.safe_load(f)
            self.class_names = names['names'] if 'names' in names else names

        # ------------------------
        # Detection model (ONNX)
        # ------------------------
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
        self.det_session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.det_input_name = self.det_session.get_inputs()[0].name

        # ------------------------
        # Classification model (Micronet)
        # ------------------------
        self.cls_model = torch.load(micronet_path, map_location=self.device)
        self.cls_model.eval()
        self.cls_transforms = T.Compose([
            T.ToTensor(),
            T.Resize((32, 32)),  # Micronet default input size
            T.Normalize((0.3403, 0.3121, 0.3214), (0.2724, 0.2608, 0.2669))  # GTSRB mean/std
        ])

    # ----------------------------
    # Frame processing functions
    # ----------------------------
    def detect(self, frame_bgr):
        img, scale, pad_left, pad_top = letterbox(frame_bgr, self.det_size)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        blob = img_rgb.astype(np.float32) / 255.0
        blob = blob.transpose(2, 0, 1)[None]  # NCHW, 1x3x640x640

        # ONNX inference
        preds = self.det_session.run(None, {self.det_input_name: blob})[0]  # shape (N, 85)

        # Parse predictions (assuming YOLO export: x1,y1,x2,y2, obj_conf, cls_conf*)
        boxes = preds[:, :4]
        obj_conf = preds[:, 4]
        cls_conf = preds[:, 5:]
        cls_ids = cls_conf.argmax(1)
        scores = obj_conf * cls_conf.max(1)

        # Filter by conf threshold
        keep = scores > self.conf_thresh
        boxes, scores, cls_ids = boxes[keep], scores[keep], cls_ids[keep]

        # NMS
        if len(boxes):
            keep_idx = non_max_suppression(boxes, scores, self.iou_thresh)
            boxes, scores, cls_ids = boxes[keep_idx], scores[keep_idx], cls_ids[keep_idx]

        # Rescale boxes back to original image coords
        rescaled = [xyxy_rescale(b, scale, pad_left, pad_top) for b in boxes]
        return np.array(rescaled), scores, cls_ids

    def classify(self, crops_bgr):
        if not len(crops_bgr):
            return []
        tensors = torch.stack([self.cls_transforms(cv2.cvtColor(c, cv2.COLOR_BGR2RGB))
                               for c in crops_bgr]).to(self.device)
        with torch.no_grad():
            logits = self.cls_model(tensors)
        probs = torch.softmax(logits, dim=1)
        conf, pred = probs.max(1)
        return pred.cpu().numpy(), conf.cpu().numpy()

    def annotate(self, frame, boxes, labels, confs):
        for box, lab, conf in zip(boxes, labels, confs):
            x1, y1, x2, y2 = map(int, box)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            text = f"{lab}: {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(frame, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
            cv2.putText(frame, text, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)
        return frame

# ---------------------------------------------
# CLI entry‑point
# ---------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Traffic‑Sign Detection + Recognition Pipeline")
    parser.add_argument('--onnx', required=True, type=Path, default='yolon12n.onnx')
    parser.add_argument('--micronet', required=True, type=Path, default='micronnet.pth',)
    parser.add_argument('--classes', required=True, type=Path, help='YAML with class names list')
    parser.add_argument('--source', default=0, help='0 = webcam or path to video file')
    parser.add_argument('--out', type=Path, default=None, help='Optional path to save annotated video')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--iou', type=float, default=0.45)
    parser.add_argument('--cpu', action='store_true', help='Force CPU inference')
    args = parser.parse_args()

    pipeline = TSRPipeline(args.onnx, args.micronet, args.classes,
                           det_imgsz=args.imgsz, conf_thresh=args.conf,
                           iou_thresh=args.iou, cuda=not args.cpu)

    # --- Video IO ---
    cap = cv2.VideoCapture(int(args.source) if str(args.source).isdigit() else str(args.source))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    out_writer = None
    if args.out:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_writer = cv2.VideoWriter(str(args.out), fourcc, fps, (w, h))

    # --- Main loop ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        t0 = time.time()
        boxes, det_scores, _ = pipeline.detect(frame)
        crops = [frame[int(y1):int(y2), int(x1):int(x2)] for x1, y1, x2, y2 in boxes]
        cls_ids, cls_conf = pipeline.classify(crops) if crops else ([], [])
        labels = [pipeline.class_names[i] for i in cls_ids] if crops else []
        annotated = pipeline.annotate(frame.copy(), boxes, labels, cls_conf)
        fps_curr = 1 / (time.time() - t0)
        cv2.putText(annotated, f"FPS: {fps_curr:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

        cv2.imshow('Traffic‑Sign Recognition', annotated)
        if out_writer:
            out_writer.write(annotated)
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
            break

    cap.release()
    if out_writer:
        out_writer.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
