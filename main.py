import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import cv2
import os
import pandas as pd
import numpy as np
import json
from datetime import datetime
import time

# ==========================================
# 1. SYSTEM CONFIGURATION
# ==========================================
GRIET_BLUE = "#0056b3"
BG_COLOR = "#f4f6f9"

# Files
CSV_FILE = "attendance.csv"
MAPPING_FILE = "student_map.json" # Maps "23241A6613" -> ID 1 for AI

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(BASE_DIR, "trainer")
HAAR_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# The 9 Subjects
SUBJECTS = [
    "Machine Learning (ML)", "Big Data Analytics (BDA)", "Software Engineering (SE)", 
    "DSRP", "Java Programming (JCP)", "ML Lab", "BDA Lab", "Mini Project (MPS)", "Constitution of India (COI)"
]

# Create Folders
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)

# Create Database if missing
if not os.path.exists(CSV_FILE):
    df = pd.DataFrame(columns=["RollNo", "Name", "Subject", "Held", "Attended", "LastUpdated"])
    df.to_csv(CSV_FILE, index=False)

# Create Mapping File if missing
if not os.path.exists(MAPPING_FILE):
    with open(MAPPING_FILE, 'w') as f: json.dump({}, f)

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================

def get_next_id():
    """Gets the next integer ID for the AI."""
    with open(MAPPING_FILE, 'r') as f:
        data = json.load(f)
    if not data: return 1
    ids = [int(k) for k in data.keys()]
    return max(ids) + 1

def save_mapping(internal_id, real_roll_no):
    """Saves the link between ID 1 and Roll 23241A6613."""
    with open(MAPPING_FILE, 'r') as f:
        data = json.load(f)
    
    # Check if roll already exists
    for k, v in data.items():
        if v == real_roll_no:
            return int(k)
            
    data[str(internal_id)] = real_roll_no
    with open(MAPPING_FILE, 'w') as f:
        json.dump(data, f)
    return internal_id

def get_roll_from_id(internal_id):
    """Gets real roll number back from ID."""
    with open(MAPPING_FILE, 'r') as f:
        data = json.load(f)
    return data.get(str(internal_id), "Unknown")

# ==========================================
# 3. GUI APPLICATION
# ==========================================
class GRIETPortal(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Gokaraju Rangaraju Institute of Engineering and Technology - CSE(AIML)")
        self.state("zoomed")
        self.configure(bg=BG_COLOR)
        
        self.current_user = None
        
        # Main Container
        self.container = tk.Frame(self, bg=BG_COLOR)
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        self.frames = {}
        for F in (LoginPage, StudentDashboard, FacultyDashboard):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
            
        self.show_frame("LoginPage")

    def show_frame(self, page_name):
        frame = self.frames[page_name]
        frame.tkraise()
        if hasattr(frame, 'refresh'): frame.refresh()

    def logout(self):
        self.current_user = None
        self.show_frame("LoginPage")

# --- LOGIN PAGE ---
class LoginPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG_COLOR)
        self.controller = controller
        
        # Banner
        banner = tk.Frame(self, bg="white", pady=20)
        banner.pack(fill="x")
        tk.Label(banner, text="GOKARAJU RANGARAJU", font=("Times New Roman", 30, "bold"), fg="#c0392b", bg="white").pack()
        tk.Label(banner, text="Institute of Engineering and Technology", font=("Arial", 14, "bold"), fg=GRIET_BLUE, bg="white").pack()
        
        # Login Box
        box = tk.Frame(self, bg=BG_COLOR)
        box.pack(expand=True)
        
        # Faculty
        fr_fac = tk.LabelFrame(box, text="FACULTY", font=("Segoe UI", 12, "bold"), bg="white", padx=30, pady=30)
        fr_fac.grid(row=0, column=0, padx=20)
        tk.Label(fr_fac, text="Admin Login", bg="white").pack(pady=5)
        self.e_fuser = tk.Entry(fr_fac); self.e_fuser.pack(pady=5)
        self.e_fpass = tk.Entry(fr_fac, show="*"); self.e_fpass.pack(pady=5)
        tk.Button(fr_fac, text="LOGIN", bg=GRIET_BLUE, fg="white", width=15, command=self.fac_login).pack(pady=10)

        # Student
        fr_stu = tk.LabelFrame(box, text="STUDENT", font=("Segoe UI", 12, "bold"), bg="white", padx=30, pady=30)
        fr_stu.grid(row=0, column=1, padx=20)
        tk.Label(fr_stu, text="Enter Roll Number", bg="white").pack(pady=5)
        self.e_suser = tk.Entry(fr_stu); self.e_suser.pack(pady=5)
        self.e_suser.insert(0, "23241A6613") # Demo value
        tk.Button(fr_stu, text="CHECK ATTENDANCE", bg=GRIET_BLUE, fg="white", width=20, command=self.stu_login).pack(pady=10)
        
        tk.Button(fr_stu, text="Register New Face", bg="white", fg="#c0392b", bd=0, cursor="hand2", 
                  command=self.register_window).pack()

    def fac_login(self):
        if self.e_fuser.get() == "admin" and self.e_fpass.get() == "admin":
            self.controller.show_frame("FacultyDashboard")
        else:
            messagebox.showerror("Error", "Invalid Admin Password")

    def stu_login(self):
        if self.e_suser.get():
            self.controller.current_user = self.e_suser.get()
            self.controller.show_frame("StudentDashboard")

    def register_window(self):
        # We use a Toplevel window for stable camera performance
        reg = tk.Toplevel(self)
        reg.title("Register Face")
        reg.geometry("400x350")
        reg.configure(bg="white")
        
        tk.Label(reg, text="1. Enter Details", font=("bold", 12), bg="white").pack(pady=10)
        tk.Label(reg, text="Roll Number:", bg="white").pack()
        e_roll = tk.Entry(reg); e_roll.pack(pady=5)
        tk.Label(reg, text="Name:", bg="white").pack()
        e_name = tk.Entry(reg); e_name.pack(pady=5)
        
        def start_capture():
            roll = e_roll.get()
            name = e_name.get()
            if not roll: return messagebox.showerror("Error", "Enter Roll No")
            
            # Map Roll Number to Integer ID
            internal_id = get_next_id()
            final_id = save_mapping(internal_id, roll)
            
            # Initialize DB Rows for this student
            self.init_db_for_student(roll, name)

            # Open Camera Window
            cam = cv2.VideoCapture(0)
            detector = cv2.CascadeClassifier(HAAR_FILE)
            count = 0
            
            while True:
                ret, frame = cam.read()
                if not ret: break
                
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = detector.detectMultiScale(gray, 1.3, 5)
                
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    count += 1
                    cv2.imwrite(f"dataset/User.{final_id}.{count}.jpg", gray[y:y+h,x:x+w])
                    cv2.putText(frame, f"Captured: {count}/50", (x,y-20), cv2.FONT_HERSHEY_DUPLEX, 0.8, (0,255,0), 2)

                cv2.imshow("Registration (Look at Camera)", frame)
                
                # Take 50 photos or stop on 'q'
                if cv2.waitKey(1) & 0xFF == ord('q') or count >= 50:
                    break
            
            cam.release()
            cv2.destroyAllWindows()
            
            # Train Immediately
            self.train_model()
            reg.destroy()
            messagebox.showinfo("Success", "Student Registered & Trained Successfully!")

        tk.Button(reg, text="2. START CAMERA & CAPTURE", bg="green", fg="white", font=("bold", 10), command=start_capture).pack(pady=20)

    def train_model(self):
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)]
        faces = []; ids = []
        for path in paths:
            try:
                img = Image.open(path).convert('L')
                img_np = np.array(img, 'uint8')
                id = int(os.path.split(path)[-1].split(".")[1])
                faces.append(img_np)
                ids.append(id)
            except: pass
        if faces:
            recognizer.train(faces, np.array(ids))
            recognizer.write(os.path.join(TRAIN_DIR, "trainer.yml"))

    def init_db_for_student(self, roll, name):
        df = pd.read_csv(CSV_FILE)
        # Check if student exists, if so, don't delete, just ensure rows exist (or reset)
        # Here we reset to ensure clean state
        df = df[df.RollNo.astype(str) != str(roll)]
        new_data = []
        for sub in SUBJECTS:
            new_data.append({"RollNo": roll, "Name": name, "Subject": sub, "Held": 0, "Attended": 0, "LastUpdated": ""})
        df = pd.concat([df, pd.DataFrame(new_data)], ignore_index=True)
        df.to_csv(CSV_FILE, index=False)


# --- FACULTY DASHBOARD ---
class FacultyDashboard(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG_COLOR)
        self.controller = controller
        
        # Header
        head = tk.Frame(self, bg=GRIET_BLUE, height=60)
        head.pack(fill="x")
        tk.Label(head, text="Faculty Dashboard", fg="white", bg=GRIET_BLUE, font=("Segoe UI", 16, "bold")).pack(side="left", padx=20)
        tk.Button(head, text="LOGOUT", bg="red", fg="white", command=controller.logout).pack(side="right", padx=20)
        
        # Controls
        ctrl = tk.Frame(self, bg="white", pady=20)
        ctrl.pack(fill="x", padx=20, pady=20)
        
        tk.Label(ctrl, text="Select Subject:").pack(side="left", padx=10)
        self.cb_sub = ttk.Combobox(ctrl, values=SUBJECTS, width=30, state="readonly")
        self.cb_sub.current(0)
        self.cb_sub.pack(side="left")
        
        tk.Button(ctrl, text="START LIVE ATTENDANCE", bg="green", fg="white", font=("bold", 11), 
                  command=self.start_attendance).pack(side="left", padx=20)
        
        self.lbl_status = tk.Label(self, text="System Ready", font=("Segoe UI", 14), bg=BG_COLOR)
        self.lbl_status.pack(pady=20)
        tk.Label(self, text="(Camera will open in a new window. Press 'q' to stop)", fg="grey", bg=BG_COLOR).pack()

    def start_attendance(self):
        subject = self.cb_sub.get()
        trainer_path = os.path.join(TRAIN_DIR, "trainer.yml")
        if not os.path.exists(trainer_path):
            return messagebox.showerror("Error", "Model not trained yet.")

        # LOAD RESOURCES
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read(trainer_path)
        face_cascade = cv2.CascadeClassifier(HAAR_FILE)
        
        cam = cv2.VideoCapture(0)
        
        self.lbl_status.config(text=f"Taking Attendance for: {subject}...", fg="blue")
        
        last_marked_time = {} # Key: RollNo, Value: timestamp
        confirmation_frames = 0 # To show "Marked" text for a few seconds
        confirmed_roll = ""

        while True:
            ret, frame = cam.read()
            if not ret: break
            
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                id_internal, conf = recognizer.predict(gray[y:y+h,x:x+w])
                
                if conf < 65: # Confidence Threshold
                    real_roll = get_roll_from_id(id_internal)
                    
                    # LOGIC TO MARK ATTENDANCE
                    now = datetime.now()
                    
                    # Check Cooldown (Don't mark same person twice in 1 minute)
                    if real_roll not in last_marked_time or (now - last_marked_time[real_roll]).seconds > 60:
                        self.mark_csv(real_roll, subject)
                        last_marked_time[real_roll] = now
                        
                        # Trigger Visual Feedback
                        confirmation_frames = 40 # Show text for 40 frames
                        confirmed_roll = real_roll
                        print(f"Attendance Marked for {real_roll}")

                    cv2.putText(frame, f"Roll: {real_roll}", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                else:
                    cv2.putText(frame, "Unknown", (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

            # VISUAL CONFIRMATION OVERLAY
            if confirmation_frames > 0:
                cv2.putText(frame, f"ATTENDANCE MARKED: {confirmed_roll}", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                cv2.putText(frame, f"Subject: {subject}", (50, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                confirmation_frames -= 1

            cv2.imshow("Live Attendance", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cam.release()
        cv2.destroyAllWindows()
        self.lbl_status.config(text="Session Ended", fg="black")

    def mark_csv(self, roll, subject):
        try:
            df = pd.read_csv(CSV_FILE)
            df['RollNo'] = df['RollNo'].astype(str)
            
            mask = (df['RollNo'] == str(roll)) & (df['Subject'] == subject)
            
            if df.loc[mask].any():
                df.loc[mask, 'Held'] += 1
                df.loc[mask, 'Attended'] += 1
                df.loc[mask, 'LastUpdated'] = datetime.now().strftime("%H:%M:%S")
                df.to_csv(CSV_FILE, index=False)
            else:
                print(f"Error: Row not found for {roll} in {subject}")
        except Exception as e: print(e)


# --- STUDENT DASHBOARD ---
class StudentDashboard(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent, bg=BG_COLOR)
        self.controller = controller
        
        nav = tk.Frame(self, bg=GRIET_BLUE, height=60)
        nav.pack(fill="x")
        tk.Label(nav, text="My Attendance Profile", fg="white", bg=GRIET_BLUE, font=("bold", 16)).pack(side="left", padx=20)
        tk.Button(nav, text="LOGOUT", bg="red", fg="white", command=controller.logout).pack(side="right", padx=20)
        
        self.lbl_welcome = tk.Label(self, text="", font=("Segoe UI", 14), bg=BG_COLOR)
        self.lbl_welcome.pack(pady=20)
        
        cols = ("Subject", "Held", "Attended", "Percentage")
        self.tree = ttk.Treeview(self, columns=cols, show='headings', height=15)
        for c in cols: 
            self.tree.heading(c, text=c)
            self.tree.column(c, anchor="center")
        self.tree.column("Subject", width=300, anchor="w")
        self.tree.pack(pady=10)

    def refresh(self):
        roll = self.controller.current_user
        self.lbl_welcome.config(text=f"Roll Number: {roll}")
        for i in self.tree.get_children(): self.tree.delete(i)
        
        try:
            df = pd.read_csv(CSV_FILE)
            df['RollNo'] = df['RollNo'].astype(str)
            data = df[df['RollNo'] == str(roll)]
            
            if data.empty:
                self.lbl_welcome.config(text=f"No data found for {roll}. Please Register first.")
                return

            for _, row in data.iterrows():
                h = int(row['Held']); a = int(row['Attended'])
                p = f"{(a/h)*100:.1f}%" if h > 0 else "0%"
                self.tree.insert("", "end", values=(row['Subject'], h, a, p))
        except: pass

if __name__ == "__main__":
    app = GRIETPortal()
    app.mainloop()