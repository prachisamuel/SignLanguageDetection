from flask import Flask, render_template, request, redirect, url_for, jsonify
import os
import cv2
import torch
import numpy as np
import pathlib
import platform
from werkzeug.utils import secure_filename
from pathlib import Path
import sys
import base64

# Windows path fix
if platform.system() == 'Windows':
    pathlib.PosixPath = pathlib.WindowsPath

# Setup
sys.path.append(str(Path(__file__).resolve().parent / 'yolov5'))

from models.common import DetectMultiBackend
from utils.general import non_max_suppression
from utils.augmentations import letterbox
from utils.torch_utils import select_device

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load YOLOv5 model
device = select_device('')
model = DetectMultiBackend('models/best.pt', device=device)
model.eval()

# Resize & detect function
def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])
        pad = ((img1_shape[1] - img0_shape[1] * gain) / 2,
               (img1_shape[0] - img0_shape[0] * gain) / 2)
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    coords[:, 0].clamp_(0, img0_shape[1])
    coords[:, 1].clamp_(0, img0_shape[0])
    coords[:, 2].clamp_(0, img0_shape[1])
    coords[:, 3].clamp_(0, img0_shape[0])
    return coords

def detect_frame(frame):
    img0 = frame.copy()
    img = letterbox(img0, new_shape=(640, 640))[0]
    img = img.transpose((2, 0, 1))[::-1]
    img = np.ascontiguousarray(img)

    img = torch.from_numpy(img).to(device).float() / 255.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    pred = model(img, augment=False, visualize=False)[0]
    pred = non_max_suppression(pred, conf_thres=0.4, iou_thres=0.5)

    for det in pred:
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], img0.shape).round()
            for *xyxy, conf, cls in det:
                label = f'{model.names[int(cls)]} {conf:.2f}'
                cv2.rectangle(img0, (int(xyxy[0]), int(xyxy[1])), 
                              (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
                cv2.putText(img0, label, (int(xyxy[0]), int(xyxy[1]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    return img0

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    frame = cv2.imread(file_path)
    result_img = detect_frame(frame)
    result_path = os.path.join(app.config['UPLOAD_FOLDER'], f'result_{filename}')
    cv2.imwrite(result_path, result_img)

    return redirect(url_for('index', filename=filename))

@app.route('/api/stream', methods=['POST'])
def api_stream():
    data = request.json.get('frame')
    if not data:
        return jsonify({'error': 'No frame received'}), 400

    # Decode base64
    img_data = base64.b64decode(data.split(',')[1])
    np_arr = np.frombuffer(img_data, np.uint8)
    frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

    result_img = detect_frame(frame)

    _, buffer = cv2.imencode('.jpg', result_img)
    encoded_img = base64.b64encode(buffer).decode('utf-8')

    return jsonify({'frame': f'data:image/jpeg;base64,{encoded_img}'})

# Run
if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)
