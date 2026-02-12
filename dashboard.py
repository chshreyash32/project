import tkinter as tk
import os

# ---------------- FUNCTION DEFINITIONS ----------------

def open_collect():
    os.system("python collect_data.py")

def train_model():
    os.system("python train_model.py")

def start_attendance():
    os.system("python attendance_system.py")

def attendance_summary():
    os.system("python attendance_summary.py")

def view_attendance():
    os.startfile("attendance")

# ---------------- MAIN WINDOW ----------------

root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("520x500")
root.configure(bg="#1e1e2f")

# ---------------- TITLE ----------------

title = tk.Label(
    root,
    text="Face Recognition\nAttendance System",
    font=("Segoe UI", 20, "bold"),
    fg="#00ffd5",
    bg="#1e1e2f"
)
title.pack(pady=20)

# ---------------- BUTTON STYLE FUNCTION ----------------

def styled_button(text, command, color):
    return tk.Button(
        root,
        text=text,
        command=command,
        font=("Segoe UI", 12, "bold"),
        width=30,
        height=2,
        bg=color,
        fg="white",
        activebackground="#333333",
        bd=0,
        cursor="hand2"
    )

# ---------------- BUTTONS ----------------

styled_button("➕ Register New Face", open_collect, "#007acc").pack(pady=7)
styled_button("🧠 Train AI Model", train_model, "#00a86b").pack(pady=7)
styled_button("📸 Start Attendance", start_attendance, "#ff9800").pack(pady=7)
styled_button("📊 Attendance Summary", attendance_summary, "#3f51b5").pack(pady=7)
styled_button("📂 View Attendance Records", view_attendance, "#9c27b0").pack(pady=7)
styled_button("❌ Exit System", root.quit, "#e53935").pack(pady=15)

# ---------------- FOOTER ----------------

footer = tk.Label(
    root,
    text="Developed by Shreyash | CSE (AIML)",
    font=("Segoe UI", 9),
    fg="gray",
    bg="#1e1e2f"
)
footer.pack(side="bottom", pady=10)

# ---------------- RUN APP ----------------

root.mainloop()
