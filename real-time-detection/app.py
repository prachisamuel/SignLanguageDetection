from flask import Flask, render_template, request, redirect, url_for, Response
import os
import cv2
import torch
import numpy as np
import numpy as np
import pathlib
import platform

# 🛠 Fix for Windows and PosixPath
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

from werkzeug.utils import secure_filename
from pathlib import Path
import sys

# Add yolov5 folder to system path
sys.path.append(str(Path(__file__).resolve().parent / 'yolov5'))

from utils.general import non_max_suppression
from utils.augmentations import letterbox
from models.common import DetectMultiBackend
from utils.torch_utils import select_device

# Initialize Flask app
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model
device = select_device('')
weights_path = 'models/best.pt'  # path to your custom model
model = DetectMultiBackend(weights_path, device=device)
model.eval()

# Route: Home
@app.route('/')
def index():
    return render_template('index.html')

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:
        # Calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2)  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    return coords

# Route: File Upload
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    # Load and preprocess the image
    img0 = cv2.imread(file_path)  # original image (BGR)
    img = letterbox(img0, new_shape=(640, 640))[0]  # resize with padding
    img = img.transpose((2, 0, 1))  # HWC to CHW
    img = np.ascontiguousarray(img)

    # Convert to torch tensor
    img = torch.from_numpy(img).to(device)
    img = img.float() / 255.0  # normalize to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    # 🔥 Set custom thresholds here
    conf_thres = 0.3   # Set this higher if you want stricter detection (try 0.5 or 0.6)
    iou_thres = 0.45   # You can try increasing to 0.6 to reduce overlapping boxes

    # Model Inference
    pred = model(img, augment=False)[0]

    # Apply NMS with thresholds
    det = non_max_suppression(pred, conf_thres=conf_thres, iou_thres=iou_thres)[0]

    # Draw boxes
    # for det in pred:
    if det is not None and len(det):
        det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
        for *xyxy, conf, cls in det:
            label = f'{model.names[int(cls)]} {conf:.2f}'
            cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
            cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

    # Save result image
    output_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{filename}')
    cv2.imwrite(output_path, img0)

    return redirect(url_for('index', result=f'result_{filename}'))

# Route: Video Stream
def generate_frames():
    print("[INFO] Webcam stream started...")
    cap = cv2.VideoCapture(0)
    print(cap.isOpened())  # should print True
    while True:
        success, frame = cap.read()
        if not success:
            break

        img0 = frame.copy()
        img = letterbox(img0, new_shape=(640, 640))[0]
        img = img.transpose((2, 0, 1))
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device)
        img = img.float() / 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # 🔥 Apply model
        pred = model(img, augment=False)[0]
        det = non_max_suppression(pred, conf_thres=0.3, iou_thres=0.45)[0]

        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Encode image to stream
        ret, buffer = cv2.imencode('.jpg', img0)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run app
if __name__ == '__main__':
    app.run(debug=True)
