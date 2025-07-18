# Player Re-Identification Pipeline

This repository implements a robust player re-identification system for single-feed sports videos (e.g., a 15-second soccer clip). It detects players, extracts appearance and jersey cues, tracks them across frames, and assigns consistent IDs even through occlusions and entry/exit.

## Features

* **YOLOv11-based Detection**: Fine-tuned to return only the "player" class.
* **Appearance Embeddings**: MobileNetV2 backbone produces 1280-D L2-normalized features.
* **Jersey OCR**: Tesseract in single-character mode with 5-frame majority voting.
* **Kalman Filtering**: Predicts motion and velocity for gating.
* **Data Association**: Hungarian algorithm on combined IoU, appearance, and jersey costs.
* **Motion Gating**: Disallows implausible jumps based on predicted speed.
* **Track Management**:

  * *Tentative* → *Confirmed* after 3 hits.
  * Survives up to 50 missed frames.
  * Only outputs tracks seen ≥5 frames.
  * 50-frame cooldown before reusing deleted IDs.

## Installation

1. **Clone the repo**

   ```bash
   git clone https://github.com/ashutoshrabia/Player-identification
   cd Player-identification
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate    # macOS/Linux
   venv\Scripts\activate     # Windows
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Install Tesseract OCR** (Windows example):

   * Download and run the installer from [UB Mannheim builds](https://github.com/UB-Mannheim/tesseract/wiki).
   * Ensure `C:\Program Files\Tesseract-OCR` is in your PATH or update `pytesseract.pytesseract.tesseract_cmd` in `main.py`.

## Configuration

* **Video paths**: Edit `VIDEO_IN` and `VIDEO_OUT` in `src/main.py`.
* **Model weights**: Place `best.pt` in `weights/` .
* **OCR settings**: Adjust `--psm` or whitelist in `main.py`’s `OCR_CONFIG`.
* **Tracker parameters**: Tweak weights and thresholds in `PlayerTracker` constructor.

## Running

```bash
python src/main.py
```

This will process the input clip, annotate each frame with bounding boxes and IDs, and save the output video.

## Directory Structure

```
project/
├── clips/                # Input video(s)
├── output/               # Annotated output video(s)
├── src/
│   ├── main.py           # Entry point
│   ├── detector.py       # YOLOv11 wrapper
│   ├── feature.py        # MobileNetV2 feature extractor
│   ├── reid_tracker.py   # Tracking & association logic
│   └── track.py          # Kalman-filtered Track class
├── weights/
│   └── best.pt           # YOLOv11 weights for player detection
├── requirements.txt
└── README.md
```

## Customization & Tuning

* **Detection confidence**: Change `conf` in `YoloDetector`.
* **Appearance vs. IoU weights**: Adjust `app_w`, `iou_w` in `PlayerTracker`.
* **Motion gating**: Tune `gating_factor` and `min_speed_thresh`.
* **Jersey cooldown**: Modify `dead_cooldown` to prevent ID reuse.
* **Output threshold**: Set `output_thresh` to control minimum visible frames.

