# train.py
import os
from ultralytics import YOLO
import argparse
import datetime

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data', default='data.yaml', help='path to data.yaml')
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--project', default='runs/tomato', help='save folder')
    p.add_argument('--name', default=None, help='run name')
    p.add_argument('--device', default='0', help='cuda device or cpu')
    return p.parse_args()

def main():
    args = parse_args()
    run_name = args.name or f'tomato_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}'
    os.makedirs(args.project, exist_ok=True)

    # model: start from yolov8n pretrained weights (nano)
    model = YOLO('yolov8n.pt')  # will download weights if missing

    # Train
    model.train(data=args.data,
                epochs=args.epochs,
                imgsz=args.imgsz,
                batch=args.batch,
                project=args.project,
                name=run_name,
                device=args.device,
                workers=4,
                patience=20)  # early-stopping patience

    # After training, best weights path:
    best_path = os.path.join(args.project, run_name, 'weights', 'best.pt')
    print('Best weights expected at:', best_path)

    # Export to ONNX and TFLite (may require extra packages)
    try:
        print('Exporting to ONNX...')
        model.export(format='onnx', imgsz=args.imgsz, simplify=True)
    except Exception as e:
        print('ONNX export failed:', e)

    try:
        print('Exporting to TFLite...')
        model.export(format='tflite', imgsz=args.imgsz)
    except Exception as e:
        print('TFLite export failed:', e)

if __name__ == '__main__':
    main()
