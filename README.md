# Facial Analysis: Blink Rate & Face Dimensions

Computer vision pipeline for analysing a long-form face recording.

- **Part A** — estimate eye blink rate across the recording
- **Part B** — estimate dimensions of face, eyes, nose, and mouth

## Approach

Uses **MediaPipe Face Mesh** (468 3D facial landmarks) per frame. Blink
detection uses the **Eye Aspect Ratio (EAR)** formula from Soukupova & Cech
(2016), averaged across both eyes, with a small consecutive-frames state
machine to avoid spurious detections.

Dimensions are measured directly from landmark coordinates:

| Measurement     | Landmarks used              |
|-----------------|-----------------------------|
| Face width      | 234 (L cheek) → 454 (R)     |
| Face height     | 10 (forehead) → 152 (chin)  |
| Eye width/height | 6-point eye landmarks      |
| Nose width      | 129 → 358                   |
| Nose height     | 168 (bridge) → 2 (tip)      |
| Mouth width     | 61 → 291                    |
| Mouth height    | 13 → 14                     |

## Setup

```bash
pip install -r requirements.txt
```

## Usage

### Run analysis

```bash
python src/analyze.py --video path/to/recording.mp4 --out results --skip 2
```

Outputs:
- `results/results.json` — summary statistics
- `results/per_frame.csv` — per-frame measurements

### Generate annotated preview clip

```bash
python src/make_clip.py --video path/to/recording.mp4 --out results/annotated_sample.mp4 --duration 60
```

## Parameters

- `--skip N`: analyse every Nth frame (default 2). Use 1 for max accuracy.
- `EAR_THRESHOLD = 0.21`: eye considered closed below this (see analyze.py).
- `CONSEC_FRAMES = 2`: minimum consecutive closed frames to register a blink.

## Output schema

`results.json` contains:
- `blink`: total count, per-second, per-minute, threshold info
- `dimensions_px_median`: median measurements in pixels
- `processing`: runtime statistics
# CV-mini-project-Assignment
