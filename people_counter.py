"""Count people in a video clip and save the counts to a CSV for the dashboard.

Run on YOUR laptop or in Google Colab (needs internet once to download the model):
    pip install ultralytics opencv-python
    python people_counter.py my_clip.mp4
    python people_counter.py my_clip.mp4 --every 1 --save-video annotated.mp4

Privacy: film only your own team, with everyone's consent. A real deployment would
count heads only and store no images or faces. This script saves counts, not faces.
"""
import argparse
import csv

import cv2
from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--every", type=float, default=1.0, help="analyse one frame every N seconds")
    ap.add_argument("--out", default="people_counts.csv")
    ap.add_argument("--conf", type=float, default=0.35, help="detection confidence")
    ap.add_argument("--save-video", default=None, help="optional annotated output video")
    a = ap.parse_args()

    model = YOLO("yolov8n.pt")            # small pre-trained model, class 0 = person
    cap = cv2.VideoCapture(a.video)
    if not cap.isOpened():
        raise SystemExit(f"Cannot open {a.video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    step = max(1, int(round(fps * a.every)))
    writer, i, rows = None, 0, []

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i % step == 0:
            res = model(frame, classes=[0], conf=a.conf, verbose=False)[0]
            people = len(res.boxes)
            rows.append((round(i / fps, 2), people))
            if a.save_video:
                shown = res.plot()
                cv2.putText(shown, f"People: {people}", (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                if writer is None:
                    h, w = shown.shape[:2]
                    writer = cv2.VideoWriter(a.save_video, cv2.VideoWriter_fourcc(*"mp4v"),
                                             max(1.0, 1 / a.every), (w, h))
                writer.write(shown)
        i += 1

    cap.release()
    if writer:
        writer.release()
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["second", "people"])
        w.writerows(rows)
    print(f"Saved {len(rows)} counts to {a.out}")


if __name__ == "__main__":
    main()
