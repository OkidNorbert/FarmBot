# inference_pi.py
import time
import cv2
import numpy as np
import serial
from ultralytics import YOLO
from ik_solver import solve_2link_ik
import argparse
import os

# --------- Config -------------
MODEL_PATH = 'best.pt'  # set to your trained weights or exported
VIDEO_SOURCE = 0        # 0 for default camera, or /dev/video0 or rtsp url
SERIAL_PORT = '/dev/ttyUSB0'  # adjust for Pi connection to Arduino
SERIAL_BAUD = 115200
HOMOGRAPHY_PATH = 'homography.npy'  # produced by calibration step
L1 = 10.0  # link1 length in cm (adjust)
L2 = 10.0  # link2 length in cm (adjust)
CONF_THRESH = 0.35

# class mapping from training
CLASS_NAMES = ['not_ready', 'ready', 'spoilt']

# --------------------------------

def load_homography(path):
    if os.path.exists(path):
        return np.load(path)
    else:
        # identity fallback (no calibration)
        return None

def pixel_to_world(px, py, H):
    # Accepts pixel coords (px,py), returns (wx,wy) in same units used for IK.
    # Uses homography H (3x3). If H is None, apply identity scale (user must calibrate).
    if H is None:
        # simple scaling fallback; user should calibrate
        return px / 10.0, py / 10.0
    denom = H[2,0]*px + H[2,1]*py + H[2,2]
    wx = (H[0,0]*px + H[0,1]*py + H[0,2]) / denom
    wy = (H[1,0]*px + H[1,1]*py + H[1,2]) / denom
    return wx, wy

def send_angles(ser, a1, a2, a3=90):
    # send in format "ANGLE a1 a2 a3\n"
    cmd = f'ANGLE {int(a1)} {int(a2)} {int(a3)}\n'
    ser.write(cmd.encode())
    print('Sent:', cmd.strip())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=MODEL_PATH)
    parser.add_argument('--source', default=VIDEO_SOURCE)
    parser.add_argument('--serial', default=SERIAL_PORT)
    parser.add_argument('--baud', type=int, default=SERIAL_BAUD)
    parser.add_argument('--homography', default=HOMOGRAPHY_PATH)
    args = parser.parse_args()

    # Load model
    model = YOLO(args.model)

    # Serial
    try:
        ser = serial.Serial(args.serial, args.baud, timeout=1)
        time.sleep(2)
        print('Serial connected to', args.serial)
    except Exception as e:
        print('Serial error:', e)
        ser = None

    # Homography
    H = load_homography(args.homography)
    if H is None:
        print('Warning: no homography loaded; pixel->world will be approximate.')

    # Video capture
    cap = cv2.VideoCapture(int(args.source)) if str(args.source).isdigit() else cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise RuntimeError('Unable to open camera source')

    print('Starting inference loop. Press q to quit.')
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Ultralytics predict
        results = model.predict(source=frame, conf=CONF_THRESH, verbose=False)

        # results is a list with one item per image (we passed a single frame)
        if len(results) > 0:
            res = results[0]
            boxes = res.boxes  # Boxes object
            if boxes is not None and len(boxes) > 0:
                # choose target priority: ready > not_ready > spoilt
                # or choose highest confidence of class 'ready' first
                chosen = None
                for cls_priority in [1, 0, 2]:  # prefer ready, then not_ready, then spoilt
                    for i, b in enumerate(boxes):
                        cls = int(b.cls[0])
                        conf = float(b.conf[0])
                        if cls == cls_priority:
                            chosen = (b.xyxy[0].tolist(), cls, conf)
                            break
                    if chosen:
                        break

                if chosen:
                    xyxy, cls, conf = chosen
                    x1, y1, x2, y2 = list(map(float, xyxy))
                    cx = (x1 + x2) / 2.0
                    cy = (y1 + y2) / 2.0
                    # Map pixel -> world (cm)
                    wx, wy = pixel_to_world(cx, cy, H)
                    print(f'Detected class={CLASS_NAMES[cls]} conf={conf:.2f} pixel=({cx:.1f},{cy:.1f}) world=({wx:.2f},{wy:.2f})')

                    # Solve IK for planar 2-link
                    ik = solve_2link_ik(wx, wy, L1, L2)
                    if ik is not None:
                        theta1, theta2 = ik
                        # Map geometry to servo angles (apply offsets as needed)
                        servo1 = 90 + theta1  # adjust per your mechanical offsets
                        servo2 = 90 + theta2
                        servo3 = 90 if cls == 1 else 0  # example: gripper open/close (customize)
                        if ser:
                            send_angles(ser, servo1, servo2, servo3)
                    else:
                        print('Target unreachable by IK.')

        # Draw for debug
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()

if __name__ == '__main__':
    main()
