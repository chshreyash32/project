import os
import cv2
import numpy as np
import tkinter as tk
from tkinter import simpledialog, messagebox
from mtcnn import MTCNN

def collect_data():
    detector = MTCNN()
    cap = cv2.VideoCapture(0)

    # --- Ask student name ---
    root = tk.Tk()
    root.withdraw()
    name = simpledialog.askstring("Student Registration", "Enter Student Name:")

    if not name:
        messagebox.showerror("Error", "Name cannot be empty")
        return

    save_path = os.path.join("dataset", name)
    os.makedirs(save_path, exist_ok=True)

    count = 0
    print("📸 Collecting face data...")

    while count < 30:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = detector.detect_faces(rgb)

        for face in faces:
            x, y, w, h = face["box"]
            x, y = max(0, x), max(0, y)

            face_img = rgb[y:y+h, x:x+w]
            if face_img.size == 0:
                continue

            face_img = cv2.resize(face_img, (160, 160))
            cv2.imwrite(f"{save_path}/{count}.jpg", cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR))
            count += 1

            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
            cv2.putText(frame, f"{count}/30", (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        cv2.imshow("Face Registration", frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

    cap.release()
    cv2.destroyAllWindows()
    messagebox.showinfo("Success", f"Face data collected for {name}")

if __name__ == "__main__":
    collect_data()
