import os
import csv
import cv2
import logging
import numpy as np
from datetime import datetime

# ==============================
# 1. FORCE TENSORFLOW SILENT MODE
# ==============================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

import tensorflow as tf
tf.get_logger().setLevel(logging.ERROR)

from mtcnn import MTCNN
from keras_facenet import FaceNet

# ==============================
# 2. INITIALIZE MODELS
# ==============================
print("⏳ System starting...")
detector = MTCNN()
embedder = FaceNet()

# ==============================
# 3. LOAD FACE EMBEDDINGS SAFELY
# ==============================
model_path = "student_embeddings.npz"

if not os.path.exists(model_path):
    print("❌ Error: student_embeddings.npz not found")
    exit()

data = np.load(model_path, allow_pickle=True)
print("🔍 NPZ keys found:", data.files)

if "embeddings" in data.files and "labels" in data.files:
    embeddings = data["embeddings"]
    labels = data["labels"]
else:
    embeddings = data[data.files[0]]
    labels = data[data.files[1]]

print(f"✅ Brain Loaded: {len(labels)} records")

# ==============================
# 4. DATE-WISE ATTENDANCE SETUP
# ==============================
from datetime import datetime

today = datetime.now().strftime("%Y-%m-%d")
attendance_dir = "attendance"
os.makedirs(attendance_dir, exist_ok=True)

attendance_file = f"{attendance_dir}/attendance_{today}.csv"
marked_names = set()

if not os.path.exists(attendance_file):
    with open(attendance_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Date", "Time"])

def mark_attendance(name):
    if name in marked_names:
        return

    now = datetime.now()
    date = now.strftime("%Y-%m-%d")
    time = now.strftime("%H:%M:%S")

    with open(attendance_file, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([name, date, time])

    marked_names.add(name)
    print(f"📝 Attendance marked: {name}")


# ==============================
# 5. CAMERA SETUP
# ==============================
cap = cv2.VideoCapture(0)

window_name = "Real-Time Attendance"
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window_name, 800, 600)

# ==============================
# 6. MAIN LOOP
# ==============================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(rgb)

    for res in results:
        x, y, w, h = res["box"]
        x, y = max(0, x), max(0, y)

        face = rgb[y:y+h, x:x+w]
        if face.size == 0:
            continue

        face = cv2.resize(face, (160, 160))
        face = np.expand_dims(face, axis=0)

        emb = embedder.embeddings(face)[0]

        distances = np.linalg.norm(embeddings - emb, axis=1)
        min_idx = np.argmin(distances)
        confidence = distances[min_idx]

        if confidence < 0.8:
            name = labels[min_idx].upper()
            color = (0, 255, 0)
            mark_attendance(name)
        else:
            name = "UNKNOWN"
            color = (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(
            frame,
            f"{name}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    cv2.imshow(window_name, frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ==============================
# 7. CLEAN EXIT
# ==============================
cap.release()
cv2.destroyAllWindows()
print("👋 System Closed")
