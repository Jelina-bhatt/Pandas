# trace_prisoner.py
# Requirements:
# pip install facenet-pytorch opencv-python numpy pandas torch scipy

import os
import cv2
import numpy as np
import pandas as pd
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
from pathlib import Path
from scipy.spatial.distance import cosine
from datetime import timedelta

# ---------------- CONFIG ----------------
# Folder containing known prisoner images
KNOWN_DIR = "known_prisoners"   # images named like ID_Name.jpg

# Camera videos to scan
CAMERAS = [
    ("GateCam", "cameras/gate.mp4"),
    ("YardCam", "cameras/yard.mp4"),
    ("StreetCam", "cameras/street.mp4"),
]

# Folder to save alerts/cropped faces
OUTPUT = "alerts"

FRAME_SKIP = 10       # analyze every Nth frame to save CPU time
FACE_THRESH = 0.45    # smaller = stricter match
# ----------------------------------------

# Create output folder if not exists
os.makedirs(OUTPUT, exist_ok=True)

# Use GPU if available, otherwise CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

# Initialize face detection and recognition models
mtcnn = MTCNN(keep_all=True, device=device)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# ---------------- 1) Load known prisoners ----------------
known = []  # list of dicts: id, name, emb
for img_path in Path(KNOWN_DIR).glob("*.*"):
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, _ = mtcnn.detect(img_rgb)
    if boxes is None:
        print(f"No face detected in {img_path.name}")
        continue

    # pick largest face
    areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
    b = boxes[int(np.argmax(areas))]
    try:
        face_tensor = mtcnn.extract(img_rgb, [b])[0].to(device)
        with torch.no_grad():
            emb = resnet(face_tensor.unsqueeze(0)).squeeze(0).cpu().numpy()
        # parse filename for ID and name
        fname = img_path.stem
        known.append({"id": fname.split("_")[0], "name": fname, "emb": emb})
        print(f"Added known prisoner: {fname}")
    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")

if not known:
    print(f"No known prisoner images found in {KNOWN_DIR}. Exiting...")
    raise SystemExit

# ---------------- 2) Scan camera videos ----------------
events = []

for cam_name, vid_path in CAMERAS:
    if not Path(vid_path).exists():
        print(f"Video not found: {vid_path}. Skipping...")
        continue

    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_idx = 0

    print(f"Scanning video: {vid_path} ({cam_name})")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = mtcnn.detect(img_rgb)
        if boxes is None:
            continue

        for b in boxes:
            try:
                x1, y1, x2, y2 = map(int, b)
                face_tensor = mtcnn.extract(img_rgb, [b])[0].to(device)
                with torch.no_grad():
                    emb = resnet(face_tensor.unsqueeze(0)).squeeze(0).cpu().numpy()

                # Compare with known embeddings
                best = None
                for k in known:
                    d = cosine(k["emb"], emb)
                    if best is None or d < best[0]:
                        best = (d, k)

                if best and best[0] < FACE_THRESH:
                    t_s = frame_idx / fps
                    timestamp = str(timedelta(seconds=int(t_s)))
                    print(f"[MATCH] {cam_name} at {timestamp} -> {best[1]['name']} (dist={best[0]:.3f})")

                    # Save cropped face
                    outdir = Path(OUTPUT)/cam_name
                    outdir.mkdir(parents=True, exist_ok=True)
                    fname = outdir / f"{best[1]['id']}_{frame_idx}.jpg"
                    cv2.imwrite(str(fname), frame[y1:y2, x1:x2])

                    events.append({
                        "camera": cam_name,
                        "time_s": t_s,
                        "frame": frame_idx,
                        "match_id": best[1]['id'],
                        "name": best[1]['name'],
                        "score": float(best[0]),
                        "path": str(fname)
                    })
            except Exception as e:
                print("Error processing face:", e)

    cap.release()

# ---------------- 3) Save events ----------------
df = pd.DataFrame(events)
if not df.empty:
    df.to_csv(Path(OUTPUT)/"trace_events.csv", index=False)
    print(f"Saved {len(df)} events -> {Path(OUTPUT)/'trace_events.csv'}")
else:
    print("No matches found. Adjust FACE_THRESH or provide better images.")
