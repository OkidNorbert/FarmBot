# calibrate_homography.py
import cv2
import numpy as np
import json

# Define real-world corners in cm (order must match the pixel points)
# Example: a 30cm x 20cm rectangle corners in clockwise order
world_pts = np.array([
    [0.0, 0.0],
    [30.0, 0.0],
    [30.0, 20.0],
    [0.0, 20.0]
], dtype=np.float32)

# Collect pixel points (interactive)
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
if not ret:
    raise RuntimeError('Failed to read camera')

pts = []

def click(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        pts.append([x, y])
        print('Added pixel:', x, y)

cv2.namedWindow('frame')
cv2.setMouseCallback('frame', click)

print('Click the 4 rectangle corners in the camera image in the SAME ORDER as world points.')
while True:
    r, frame = cap.read()
    for p in pts:
        cv2.circle(frame, (int(p[0]), int(p[1])), 5, (0,255,0), -1)
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if len(pts) >= 4:
        break

cap.release()
cv2.destroyAllWindows()

if len(pts) < 4:
    raise RuntimeError('Not enough points selected')

pixel_pts = np.array(pts[:4], dtype=np.float32)
H, status = cv2.findHomography(pixel_pts, world_pts)
np.save('homography.npy', H)
print('Saved homography.npy')
