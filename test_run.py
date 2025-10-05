# test_run.py
import csv, time
from ultralytics import YOLO
import cv2

model = YOLO('best.pt')
cap = cv2.VideoCapture(0)

N = 100
out = open('test_log.csv', 'w', newline='')
writer = csv.writer(out)
writer.writerow(['timestamp','frame_idx','class','conf','x1','y1','x2','y2'])

for i in range(N):
    ret, frame = cap.read()
    if not ret:
        break
    res = model.predict(source=frame, conf=0.35, verbose=False)[0]
    boxes = res.boxes
    if boxes is not None and len(boxes) > 0:
        for b in boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            x1,y1,x2,y2 = map(float, b.xyxy[0].tolist())
            writer.writerow([time.time(), i, cls, conf, x1, y1, x2, y2])
    else:
        writer.writerow([time.time(), i, 'none', 0, 0,0,0,0])
out.close()
cap.release()
print('Done logging to test_log.csv')
