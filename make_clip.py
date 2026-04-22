"""
Generate an annotated preview clip showing face mesh landmarks and blink events.
Run after analyze.py to produce a short visualisation video.
"""

import argparse
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from analyze import (
    LEFT_EYE, RIGHT_EYE,
    FACE_TOP, FACE_BOTTOM, FACE_LEFT, FACE_RIGHT,
    NOSE_TOP, NOSE_BOTTOM, NOSE_LEFT, NOSE_RIGHT,
    MOUTH_TOP, MOUTH_BOTTOM, MOUTH_LEFT, MOUTH_RIGHT,
    EAR_THRESHOLD, CONSEC_FRAMES,
    eye_aspect_ratio,
)


def annotate(video_in: str, video_out: str, duration_s: float = 60.0):
    cap = cv2.VideoCapture(video_in)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {video_in}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # cap output width at 1280 for size
    if w > 1280:
        out_w = 1280
        out_h = int(h * 1280 / w)
    else:
        out_w, out_h = w, h

    writer = cv2.VideoWriter(
        video_out,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (out_w, out_h),
    )

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    n_frames = int(fps * duration_s)
    closed_streak = 0
    blink_flash   = 0  # frames remaining to flash BLINK label

    for idx in range(n_frames):
        ret, frame = cap.read()
        if not ret:
            break

        if (out_w, out_h) != (w, h):
            frame = cv2.resize(frame, (out_w, out_h))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            lm = result.multi_face_landmarks[0].landmark

            def pt(i):
                return int(lm[i].x * out_w), int(lm[i].y * out_h)

            # face box
            left, right = pt(FACE_LEFT), pt(FACE_RIGHT)
            top, bot    = pt(FACE_TOP),  pt(FACE_BOTTOM)
            cv2.rectangle(frame, (left[0], top[1]), (right[0], bot[1]),
                          (0, 220, 120), 2)

            # eye points
            for i in LEFT_EYE + RIGHT_EYE:
                cv2.circle(frame, pt(i), 2, (100, 200, 255), -1)

            # nose & mouth
            cv2.line(frame, pt(NOSE_TOP),   pt(NOSE_BOTTOM),  (255, 160, 50), 1)
            cv2.line(frame, pt(NOSE_LEFT),  pt(NOSE_RIGHT),   (255, 160, 50), 1)
            cv2.line(frame, pt(MOUTH_LEFT), pt(MOUTH_RIGHT),  (50, 160, 255), 1)
            cv2.line(frame, pt(MOUTH_TOP),  pt(MOUTH_BOTTOM), (50, 160, 255), 1)

            # EAR + blink detection
            ear = (eye_aspect_ratio(lm, LEFT_EYE, out_w, out_h) +
                   eye_aspect_ratio(lm, RIGHT_EYE, out_w, out_h)) / 2.0
            cv2.putText(frame, f"EAR {ear:.3f}", (left[0], bot[1] + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

            if ear < EAR_THRESHOLD:
                closed_streak += 1
            else:
                if closed_streak >= CONSEC_FRAMES:
                    blink_flash = int(fps * 0.4)  # flash for 0.4s
                closed_streak = 0

        if blink_flash > 0:
            cv2.putText(frame, "BLINK", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.2, (0, 80, 255), 5)
            blink_flash -= 1

        cv2.putText(frame, f"t={idx/fps:6.2f}s", (14, out_h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        writer.write(frame)

    writer.release()
    cap.release()
    face_mesh.close()
    print(f"Wrote annotated clip: {video_out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out",   default="results/annotated_sample.mp4")
    ap.add_argument("--duration", type=float, default=60.0)
    args = ap.parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    annotate(args.video, args.out, args.duration)


if __name__ == "__main__":
    main()
