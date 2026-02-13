import streamlit as st
import cv2
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from PIL import Image

# ==========================================
# 1. CONFIGURATION & STATE MANAGEMENT
# ==========================================
st.set_page_config(layout="wide", page_title="CSE (AIML) Attendance", page_icon="📷")

# -- PATHS --
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(BASE_DIR, "trainer")
CSV_FILE = os.path.join(BASE_DIR, "attendance.csv")
MAPPING_FILE = os.path.join(BASE_DIR, "student_map.json")
# Fix for opencv path issues
HAAR_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# -- CONFIG LISTS --
SUBJECT_LIST = [
    "Machine Learning", 
    "Big Data Analytics", 
    "Software Engineering", 
    "DSRP", 
    "JCP", 
    "ML Lab", 
    "BDA Lab", 
    "Mini Project", 
    "Constitution of India"
]

SECTION_LIST = ["Section A", "Section B", "Section C"]

# -- INIT DIRECTORIES --
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)

# -- INIT DATABASE --
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["RollNo", "Name", "Subject", "Section", "Held", "Attended", "LastUpdated"]).to_csv(CSV_FILE, index=False)

if not os.path.exists(MAPPING_FILE):
    with open(MAPPING_FILE, 'w') as f: json.dump({}, f)

# -- SESSION STATE --
if 'page' not in st.session_state: st.session_state['page'] = 'home'
if 'capture_count' not in st.session_state: st.session_state['capture_count'] = 0
if 'attendance_log' not in st.session_state: st.session_state['attendance_log'] = []

def navigate_to(page):
    st.session_state['page'] = page
    st.rerun()

# ==========================================
# 2. CUSTOM CSS (PREMIUM MINIMALIST DARK)
# ==========================================
st.markdown("""
<style>
    /* GLOBAL THEME */
    .stApp {
        background-color: #000000;
        color: #e0e0e0;
    }
    
    /* INPUT FIELDS - CLEAN & DARK */
    div[data-baseweb="input"] {
        background-color: #111111;
        border: 1px solid #333;
        border-radius: 8px;
        color: white;
    }
    div[data-baseweb="base-input"] {
        background-color: #111111;
    }
    
    /* BUTTONS */
    .stButton > button {
        background-color: #007bff;
        color: #ffffff !important;
        border: none;
        border-radius: 6px;
        padding: 12px 20px;
        font-weight: 600;
        width: 100%;
    }
    .stButton > button:hover {
        background-color: #0056b3;
        color: #ffffff !important;
    }
    
    /* TABLE STYLING */
    div[data-testid="stDataFrame"] {
        background-color: #111;
        border: 1px solid #333;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 3. BACKEND LOGIC
# ==========================================
def get_next_id():
    with open(MAPPING_FILE, 'r') as f: data = json.load(f)
    if not data: return 1
    return max([int(k) for k in data.keys()]) + 1

def save_mapping(internal_id, roll):
    with open(MAPPING_FILE, 'r') as f: data = json.load(f)
    for k, v in data.items():
        if v == roll: return int(k)
    data[str(internal_id)] = roll
    with open(MAPPING_FILE, 'w') as f: json.dump(data, f)
    return internal_id

def get_roll(internal_id):
    with open(MAPPING_FILE, 'r') as f: data = json.load(f)
    return data.get(str(internal_id), "Unknown")

def train_model():
    # Check if cv2.face exists (requires opencv-contrib-python)
    if not hasattr(cv2, 'face'):
        st.error("❌ 'opencv-contrib-python' is missing. Please add it to requirements.txt.")
        return False

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)]
    faces, ids = [], []
    for path in paths:
        try:
            img = Image.open(path).convert('L')
            faces.append(np.array(img, 'uint8'))
            ids.append(int(os.path.split(path)[-1].split(".")[1]))
        except: pass
    if faces:
        recognizer.train(faces, np.array(ids))
        recognizer.write(os.path.join(TRAIN_DIR, "trainer.yml"))
        return True
    return False

# ==========================================
# 4. PAGES
# ==========================================

# --- HOME ---
def page_home():
    st.markdown("<h1 style='text-align: center;'>CSE (AIML) Portal</h1>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🎓 Student Portal")
        if st.button("Continue as Student"): navigate_to("student_hub")
    with c2:
        st.markdown("### 👨‍🏫 Faculty Login")
        if st.button("Continue as Faculty"): navigate_to("faculty_login")

# --- STUDENT HUB ---
def page_student_hub():
    if st.button("← Back"): navigate_to("home")
    st.markdown("<h2>Student Dashboard</h2>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("View Attendance"): navigate_to("student_view")
    with c2:
        if st.button("Register Face"): navigate_to("student_register")

# --- STUDENT REGISTER (MANUAL CAPTURE) ---
def page_student_register():
    if st.button("← Back"): 
        st.session_state['capture_count'] = 0
        st.session_state['reg_step'] = None
        navigate_to("student_hub")
    
    if st.session_state.get('reg_step') != 'capture':
        st.markdown("<h3>New Student Registration</h3>", unsafe_allow_html=True)
        with st.container():
            col_form, col_space = st.columns([2, 1])
            with col_form:
                name = st.text_input("Full Name")
                roll = st.text_input("Roll Number")
                section = st.selectbox("Section", SECTION_LIST)
                
                if st.button("Next: Capture Face"):
                    if name and roll:
                        st.session_state.update({
                            'reg_name': name, 'reg_roll': roll, 'reg_sec': section, 
                            'reg_step': 'capture', 'capture_count': 0
                        })
                        st.rerun()
                    else: st.error("Please fill all details.")
    else:
        st.markdown(f"<h3>📸 Manual Capture: {st.session_state['reg_name']}</h3>", unsafe_allow_html=True)
        final_id = save_mapping(get_next_id(), st.session_state['reg_roll'])
        detector = cv2.CascadeClassifier(HAAR_FILE)
        
        c_cam, c_txt = st.columns([2, 1])
        with c_txt:
            st.markdown(f"### Progress: {st.session_state.get('capture_count', 0)}/30")
            st.progress(st.session_state.get('capture_count', 0) / 30)
            if st.session_state.get('capture_count', 0) >= 30:
                st.success("✅ All captures done!")
        
        with c_cam:
            if st.session_state.get('capture_count', 0) < 30:
                img_file = st.camera_input(f"Photo {st.session_state.get('capture_count', 0) + 1}", key=f"cam_{st.session_state.get('capture_count', 0)}")
                if img_file is not None:
                    image = Image.open(img_file)
                    gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
                    faces = detector.detectMultiScale(gray, 1.3, 5)
                    if len(faces) > 0:
                        for (x, y, w, h) in faces:
                            st.session_state['capture_count'] += 1
                            cv2.imwrite(f"{DATA_DIR}/User.{final_id}.{st.session_state['capture_count']}.jpg", gray[y:y+h, x:x+w])
                        st.rerun()
                    else:
                        st.error("❌ No face detected!")
            else:
                if train_model():
                    df = pd.read_csv(CSV_FILE)
                    df = df[df['RollNo'].astype(str) != st.session_state['reg_roll']]
                    new_rows = []
                    for sub in SUBJECT_LIST:
                        new_rows.append({"RollNo": st.session_state['reg_roll'], "Name": st.session_state['reg_name'], "Subject": sub, "Section": st.session_state['reg_sec'], "Held": 0, "Attended": 0, "LastUpdated": "-"})
                    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                    df.to_csv(CSV_FILE, index=False)
                    st.success("✅ Registration Successful!")
                    if st.button("Finish"):
                        st.session_state['reg_step'] = None
                        st.session_state['capture_count'] = 0
                        navigate_to("student_hub")

# --- STUDENT VIEW ---
def page_student_view():
    if st.button("← Back"): navigate_to("student_hub")
    roll = st.text_input("Enter Roll Number")
    if st.button("Check"):
        df = pd.read_csv(CSV_FILE)
        data = df[df['RollNo'].astype(str) == roll]
        if not data.empty:
            st.dataframe(data)
        else:
            st.error("No records found.")

# --- FACULTY LOGIN ---
def page_faculty_login():
    if st.button("← Home"): navigate_to("home")
    u = st.text_input("Faculty ID")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if len(u) == 4 and u.isdigit() and u == p: navigate_to("faculty_dashboard")
        else: st.error("Invalid Credentials.")

# --- FACULTY DASHBOARD ---
def page_faculty_dashboard():
    if st.button("← Logout"): navigate_to("home")
    st.markdown("<h2>Faculty Dashboard</h2>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1: sub = st.selectbox("Subject", SUBJECT_LIST)
    with c2: sec = st.selectbox("Section", SECTION_LIST)
    periods = st.selectbox("No. of Periods", [1, 2, 3, 4])
    if st.button("🚀 Start Live Attendance"):
        st.session_state.update({'live_sub': sub, 'live_sec': sec, 'live_periods': periods, 'attendance_log': []})
        navigate_to("live_attendance")

# --- LIVE ATTENDANCE (FIXED) ---
def page_live_attendance():
    if st.button("← Back"): navigate_to("faculty_dashboard")
    sub = st.session_state.get('live_sub')
    periods = st.session_state.get('live_periods')
    
    st.markdown(f"<h3>📸 Live Attendance: {sub}</h3>", unsafe_allow_html=True)

    col_cam, col_log = st.columns([2, 1])
    
    with col_log:
        st.markdown("#### Log")
        for log in st.session_state['attendance_log'][:10]:
            st.markdown(f"{log}")

    with col_cam:
        if not os.path.exists(os.path.join(TRAIN_DIR, "trainer.yml")):
            st.error("❌ Model not trained. Register students first.")
            return # STOP EXECUTION HERE IF MODEL MISSING

        # SAFETY CHECK FOR CV2.FACE
        if not hasattr(cv2, 'face'):
             st.error("❌ Library Error: 'opencv-contrib-python-headless' is missing from requirements.txt")
             return

        img_file = st.camera_input("📸 Capture Attendance Photo", key="attendance_photo")
        
        if img_file is not None:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(os.path.join(TRAIN_DIR, "trainer.yml"))
            face_cascade = cv2.CascadeClassifier(HAAR_FILE)
            
            frame = cv2.cvtColor(np.array(Image.open(img_file)), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.2, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 3)
                try:
                    id_internal, conf = recognizer.predict(gray[y:y+h, x:x+w])
                    if conf < 65:
                        real_roll = get_roll(id_internal)
                        df = pd.read_csv(CSV_FILE)
                        df['RollNo'] = df['RollNo'].astype(str)
                        mask = (df['RollNo'] == str(real_roll)) & (df['Subject'] == sub)
                        if not df.loc[mask].empty:
                            idx = df.index[mask].tolist()[0]
                            df.at[idx, 'Held'] = int(df.at[idx, 'Held']) + periods
                            df.at[idx, 'Attended'] = int(df.at[idx, 'Attended']) + periods
                            df.at[idx, 'LastUpdated'] = datetime.now().strftime("%H:%M:%S")
                            df.to_csv(CSV_FILE, index=False)
                            st.session_state['attendance_log'].insert(0, f"✅ {real_roll} Marked")
                            cv2.putText(frame, f"{real_roll}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        else:
                            st.session_state['attendance_log'].insert(0, f"⚠️ {real_roll} Wrong Sub")
                    else:
                        cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                except: pass
            
            st.image(frame, channels="BGR")

# ==========================================
# 5. ROUTER
# ==========================================
if st.session_state['page'] == "home": page_home()
elif st.session_state['page'] == "student_hub": page_student_hub()
elif st.session_state['page'] == "student_register": page_student_register()
elif st.session_state['page'] == "student_view": page_student_view()
elif st.session_state['page'] == "faculty_login": page_faculty_login()
elif st.session_state['page'] == "faculty_dashboard": page_faculty_dashboard()
elif st.session_state['page'] == "live_attendance": page_live_attendance()
