import os
import csv
import tkinter as tk
from tkinter import simpledialog, messagebox

ATTENDANCE_DIR = "attendance"

def show_summary():
    if not os.path.exists(ATTENDANCE_DIR):
        messagebox.showerror("Error", "No attendance records found")
        return

    root = tk.Tk()
    root.withdraw()

    student_name = simpledialog.askstring(
        "Attendance Summary",
        "Enter Student Name:"
    )

    if not student_name:
        messagebox.showwarning("Warning", "Student name cannot be empty")
        return

    student_name = student_name.upper()
    present_days = 0
    total_days = 0

    for file in os.listdir(ATTENDANCE_DIR):
        if not file.endswith(".csv"):
            continue

        total_days += 1
        file_path = os.path.join(ATTENDANCE_DIR, file)

        with open(file_path, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header

            for row in reader:
                if row and row[0].upper() == student_name:
                    present_days += 1
                    break

    messagebox.showinfo(
        "Attendance Summary",
        f"Student Name: {student_name}\n\n"
        f"Days Present: {present_days}\n"
        f"Total Working Days: {total_days}"
    )

if __name__ == "__main__":
    show_summary()
