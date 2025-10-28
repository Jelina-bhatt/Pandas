import cv2
from ultralytics import YOLO
from pathlib import Path
from datetime import datetime

# ---------------- CONFIG ----------------
OUTPUT_DIR = "alerts"
FRAME_SKIP = 5
MODEL = "yolov8n.pt"  # YOLOv8 small model
# ----------------------------------------

Path(OUTPUT_DIR).mkdir(exist_ok=True)

# Load YOLO model
model = YOLO(MODEL)

# Open webcam (0 = default camera)
cap = cv2.VideoCapture(0)
frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % FRAME_SKIP != 0:
        continue

    results = model(frame)
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            if cls != 0:  # 0 = person in COCO
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0,0,255), 2)
            cv2.putText(frame, "INTRUDER!", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # Save snapshot
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{OUTPUT_DIR}/intruder_{frame_idx}_{timestamp}.jpg"
            cv2.imwrite(fname, frame[y1:y2, x1:x2])
            print("Saved intruder snapshot:", fname)

    cv2.imshow("Intruder Alert", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
