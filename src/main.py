# src/main.py
import cv2
import os
from pathlib import Path
from tqdm import tqdm
import numpy as np
import pytesseract


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

from detector import YoloDetector
from reid_tracker import PlayerTracker

# Based on model.names: {0:'ball',1:'goalkeeper',2:'player',3:'referee'}
PLAYER_CLASS_INDEX = 2
VIDEO_IN  = "clips/15sec_input_720p.mp4"
VIDEO_OUT = "output/output_with_reid.mp4"

OCR_CONFIG = '--psm 10 -c tessedit_char_whitelist=0123456789'
# Minimum bounding box dimensions for player detection
MIN_BOX_W, MIN_BOX_H = 50, 80


def extract_jersey_number(crop):
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)
    kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]])
    gray = cv2.filter2D(gray, -1, kernel)
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 2
    )
    text = pytesseract.image_to_string(thresh, config=OCR_CONFIG)
    digits = ''.join(filter(str.isdigit, text))
    return digits if digits else None


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)
    det = YoloDetector(
        weight_path="weights/best.pt",
        conf=0.25,
        classes=[PLAYER_CLASS_INDEX]
    )
    tracker = PlayerTracker(
        extract_fn=extract_jersey_number,
        min_box_w=MIN_BOX_W,
        min_box_h=MIN_BOX_H
    )

    cap = cv2.VideoCapture(VIDEO_IN)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    out = cv2.VideoWriter(
        VIDEO_OUT,
        cv2.VideoWriter_fourcc(*'mp4v'),
        fps, (w, h)
    )

    for _ in tqdm(range(total), desc="Processing"):
        ret, frame = cap.read()
        if not ret:
            break

        boxes = det(frame)
        tracks = tracker.update(boxes, frame)

        for t in tracks:
            x1, y1, x2, y2 = map(int, t['bbox'])
            label = f"ID {t['track_id']}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(
                frame, label,
                (x1, y1 - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0,255,0), 2
            )

        out.write(frame)

    cap.release()
    out.release()
    print(f"Done! Annotated video saved to {Path(VIDEO_OUT).resolve()}")
