"""
Facial Analysis for Blink Rate & Face Dimensions
-------------------------------------------------
Part A: Estimate eye blink rate over a long recording
Part B: Estimate dimensions of eyes, face, nose, mouth

Approach: MediaPipe Face Mesh (468 3D landmarks) for real per-frame geometry.
Blink detection via Eye Aspect Ratio (EAR) from the six canonical eye landmarks
(Soukupova & Cech, 2016).
"""

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

# --- MediaPipe face mesh landmark indices (standard reference points) ---
# Left eye 6-point EAR landmarks
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Right eye 6-point EAR landmarks
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Face bounding landmarks
FACE_TOP    = 10    # forehead
FACE_BOTTOM = 152   # chin
FACE_LEFT   = 234   # left cheek (ear side)
FACE_RIGHT  = 454   # right cheek (ear side)

# Nose
NOSE_TOP    = 168
NOSE_BOTTOM = 2
NOSE_LEFT   = 129
NOSE_RIGHT  = 358

# Mouth (outer)
MOUTH_LEFT   = 61
MOUTH_RIGHT  = 291
MOUTH_TOP    = 13
MOUTH_BOTTOM = 14

# Blink thresholds
EAR_THRESHOLD = 0.21     # below this = eye considered closed
CONSEC_FRAMES = 2        # need this many closed frames to count as blink


def eye_aspect_ratio(landmarks, eye_idx, w, h):
    """Standard 6-point EAR formula."""
    pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in eye_idx])
    # vertical distances
    a = np.linalg.norm(pts[1] - pts[5])
    b = np.linalg.norm(pts[2] - pts[4])
    # horizontal distance
    c = np.linalg.norm(pts[0] - pts[3])
    return (a + b) / (2.0 * c) if c > 0 else 0.0


def dist(landmarks, i, j, w, h):
    p1 = np.array([landmarks[i].x * w, landmarks[i].y * h])
    p2 = np.array([landmarks[j].x * w, landmarks[j].y * h])
    return float(np.linalg.norm(p1 - p2))


def analyze_video(video_path: str, out_dir: Path, skip: int = 2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps          = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_s   = total_frames / fps
    w_frame      = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_frame      = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    print(f"Video: {total_frames} frames @ {fps:.2f} fps = {duration_s/60:.1f} min")
    print(f"Resolution: {w_frame} x {h_frame}")

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    rows            = []
    blink_count     = 0
    closed_streak   = 0
    blink_timestamps = []

    t0 = time.time()
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip != 0:
            frame_idx += 1
            continue

        ts = frame_idx / fps
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        row = {
            "frame": frame_idx, "time_s": round(ts, 3),
            "face_detected": False, "ear": 0.0, "is_blink": False,
            "face_w": 0.0, "face_h": 0.0,
            "left_eye_w": 0.0, "left_eye_h": 0.0,
            "right_eye_w": 0.0, "right_eye_h": 0.0,
            "nose_w": 0.0, "nose_h": 0.0,
            "mouth_w": 0.0, "mouth_h": 0.0,
        }

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark
            row["face_detected"] = True

            left_ear  = eye_aspect_ratio(lm, LEFT_EYE,  w_frame, h_frame)
            right_ear = eye_aspect_ratio(lm, RIGHT_EYE, w_frame, h_frame)
            ear = (left_ear + right_ear) / 2.0
            row["ear"] = round(ear, 4)

            # blink state machine
            if ear < EAR_THRESHOLD:
                closed_streak += 1
            else:
                if closed_streak >= CONSEC_FRAMES:
                    blink_count += 1
                    blink_timestamps.append(round(ts, 3))
                    row["is_blink"] = True
                closed_streak = 0

            # dimensions (pixels)
            row["face_w"]      = round(dist(lm, FACE_LEFT,  FACE_RIGHT,  w_frame, h_frame), 1)
            row["face_h"]      = round(dist(lm, FACE_TOP,   FACE_BOTTOM, w_frame, h_frame), 1)
            row["left_eye_w"]  = round(dist(lm, LEFT_EYE[0],  LEFT_EYE[3],  w_frame, h_frame), 1)
            row["left_eye_h"]  = round(dist(lm, LEFT_EYE[1],  LEFT_EYE[5],  w_frame, h_frame), 1)
            row["right_eye_w"] = round(dist(lm, RIGHT_EYE[0], RIGHT_EYE[3], w_frame, h_frame), 1)
            row["right_eye_h"] = round(dist(lm, RIGHT_EYE[1], RIGHT_EYE[5], w_frame, h_frame), 1)
            row["nose_w"]      = round(dist(lm, NOSE_LEFT,   NOSE_RIGHT,   w_frame, h_frame), 1)
            row["nose_h"]      = round(dist(lm, NOSE_TOP,    NOSE_BOTTOM,  w_frame, h_frame), 1)
            row["mouth_w"]     = round(dist(lm, MOUTH_LEFT,  MOUTH_RIGHT,  w_frame, h_frame), 1)
            row["mouth_h"]     = round(dist(lm, MOUTH_TOP,   MOUTH_BOTTOM, w_frame, h_frame), 1)

        rows.append(row)
        frame_idx += 1

        if frame_idx % 500 == 0:
            pct = frame_idx / total_frames * 100
            print(f"  {pct:5.1f}%  frame {frame_idx}/{total_frames}  blinks={blink_count}")

    cap.release()
    face_mesh.close()

    elapsed = time.time() - t0
    print(f"\nProcessing done in {elapsed:.1f}s")

    # --- aggregate ---
    detected = [r for r in rows if r["face_detected"]]

    def median_of(key):
        vals = [r[key] for r in detected if r[key] > 0]
        return float(np.median(vals)) if vals else 0.0

    summary = {
        "video_duration_s":   round(duration_s, 1),
        "total_frames":       total_frames,
        "frames_analyzed":    len(rows),
        "frames_with_face":   len(detected),
        "face_detection_pct": round(len(detected) / max(len(rows), 1) * 100, 2),
        "blink": {
            "total_blinks":          blink_count,
            "blinks_per_second":     round(blink_count / duration_s, 5) if duration_s else 0,
            "blinks_per_minute":     round(blink_count / duration_s * 60, 3) if duration_s else 0,
            "ear_threshold":         EAR_THRESHOLD,
            "consecutive_frames":    CONSEC_FRAMES,
            "first_20_timestamps":   blink_timestamps[:20],
        },
        "dimensions_px_median": {
            "face_width_ear_to_ear":    round(median_of("face_w"), 1),
            "face_height_head_to_chin": round(median_of("face_h"), 1),
            "left_eye_width":           round(median_of("left_eye_w"), 1),
            "left_eye_height":          round(median_of("left_eye_h"), 1),
            "right_eye_width":          round(median_of("right_eye_w"), 1),
            "right_eye_height":         round(median_of("right_eye_h"), 1),
            "nose_width":               round(median_of("nose_w"), 1),
            "nose_height":              round(median_of("nose_h"), 1),
            "mouth_width":              round(median_of("mouth_w"), 1),
            "mouth_height":             round(median_of("mouth_h"), 1),
        },
        "processing": {
            "elapsed_seconds":  round(elapsed, 1),
            "effective_fps":    round(frame_idx / elapsed, 1) if elapsed else 0,
            "frame_skip":       skip,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {out_dir / 'results.json'}")

    with open(out_dir / "per_frame.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {out_dir / 'per_frame.csv'}")

    return summary


def main():
    ap = argparse.ArgumentParser(description="Blink rate + face dimension analysis")
    ap.add_argument("--video", required=True, help="Path to input video")
    ap.add_argument("--out",   default="results", help="Output directory")
    ap.add_argument("--skip",  type=int, default=2,
                    help="Analyze every Nth frame (1=every frame)")
    args = ap.parse_args()

    summary = analyze_video(args.video, Path(args.out), skip=args.skip)

    print("\n=== SUMMARY ===")
    print(f"Duration:     {summary['video_duration_s']:.0f} s")
    print(f"Total blinks: {summary['blink']['total_blinks']}")
    print(f"Blinks/min:   {summary['blink']['blinks_per_minute']}")
    d = summary["dimensions_px_median"]
    print(f"Face:  {d['face_width_ear_to_ear']} x {d['face_height_head_to_chin']} px")
    print(f"Eye:   {d['left_eye_width']} x {d['left_eye_height']} px")
    print(f"Nose:  {d['nose_width']} x {d['nose_height']} px")
    print(f"Mouth: {d['mouth_width']} x {d['mouth_height']} px")


if __name__ == "__main__":
    main()
