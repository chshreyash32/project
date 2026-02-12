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
st.set_page_config(layout="wide", page_title="Intelligent Attendance", page_icon="📷")

# -- PATHS --
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(BASE_DIR, "trainer") # Folder for trainer.yml
CSV_FILE = os.path.join(BASE_DIR, "attendance.csv")
MAPPING_FILE = os.path.join(BASE_DIR, "student_map.json")
HAAR_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

# -- GLOBAL SUBJECT LIST --
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
if 'live_run' not in st.session_state: st.session_state['live_run'] = False

def navigate_to(page):
    st.session_state['page'] = page
    st.rerun()

# ==========================================
# 2. CUSTOM CSS (DARK MODE & STYLING)
# ==========================================
st.markdown("""
<style>
    .stApp { background-color: #050505; color: #e0e0e0; }
    .custom-card { background-color: #121212; border: 1px solid #333; border-radius: 12px; padding: 25px; margin-bottom: 20px; }
    .card-title { font-size: 22px; font-weight: 600; color: #ffffff; margin-bottom: 8px; }
    .card-desc { color: #a0a0a0; font-size: 14px; margin-bottom: 20px; }
    .stButton > button { background-color: #007bff; color: white; border: none; border-radius: 8px; padding: 12px; font-weight: 600; width: 100%; }
    .stButton > button:hover { background-color: #0056b3; }
    div[data-testid="stDataFrame"] { background-color: #1e1e1e; border-radius: 10px; padding: 10px; }
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
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)]
    faces, ids = [], []
    for path in paths:
        try:
            img = Image.open(path).convert('L')
            faces.append(np.array(img, 'uint8'))
            # Extract ID from filename format: User.ID.Count.jpg
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
    st.markdown("<h1 style='text-align: center;'>Intelligent Attendance System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Gokaraju Rangaraju Institute of Engineering and Technology</p>", unsafe_allow_html=True)
    st.write("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="custom-card"><div class="card-title">🎓 Student Portal</div><div class="card-desc">Login to view attendance or Register</div></div>""", unsafe_allow_html=True)
        if st.button("Continue as Student"): navigate_to("student_hub")
    with c2:
        st.markdown("""<div class="custom-card"><div class="card-title">👨‍🏫 Faculty Login</div><div class="card-desc">Mark attendance and view reports</div></div>""", unsafe_allow_html=True)
        if st.button("Continue as Faculty"): navigate_to("faculty_login")

# --- STUDENT HUB ---
def page_student_hub():
    if st.button("← Back"): navigate_to("home")
    st.markdown("<h2>Student Portal</h2>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""<div class="custom-card"><div class="card-title">📊 Check Attendance</div><div class="card-desc">View your attendance table</div></div>""", unsafe_allow_html=True)
        if st.button("View Attendance"): navigate_to("student_view")
    with c2:
        st.markdown("""<div class="custom-card"><div class="card-title">👤 Register</div><div class="card-desc">Register your face</div></div>""", unsafe_allow_html=True)
        if st.button("Register Face"): navigate_to("student_register")

# --- STUDENT REGISTER ---
def page_student_register():
    if st.button("← Back"): navigate_to("student_hub")
    
    if st.session_state.get('reg_step') != 'capture':
        st.markdown("<h3>Register New Student</h3>", unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="custom-card">', unsafe_allow_html=True)
            name = st.text_input("Full Name")
            roll = st.text_input("Roll Number")
            section = st.selectbox("Section", ["Section A", "Section B", "CSE-AIML"])
            if st.button("Next: Face Capture"):
                if name and roll:
                    st.session_state.update({'reg_name': name, 'reg_roll': roll, 'reg_sec': section, 'reg_step': 'capture'})
                    st.rerun()
                else: st.error("Please fill all details.")
            st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown(f"<h3>Capturing: {st.session_state['reg_name']}</h3>", unsafe_allow_html=True)
        c_cam, c_txt = st.columns([2, 1])
        with c_cam: cam_ph = st.image([])
        with c_txt: 
            st.info("Capturing 30 images...")
            status = st.empty()
            
        cap = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier(HAAR_FILE)
        final_id = save_mapping(get_next_id(), st.session_state['reg_roll'])
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret: break
            frame = cv2.flip(frame, 1)
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x,y,w,h) in faces:
                cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                count += 1
                cv2.imwrite(f"{DATA_DIR}/User.{final_id}.{count}.jpg", gray[y:y+h,x:x+w])
            cam_ph.image(frame, channels="BGR")
            status.markdown(f"**Progress: {count}/30**")
            if count >= 30: break
        cap.release()
        status.success("Processing...")
        
        if train_model():
            # INIT DB
            df = pd.read_csv(CSV_FILE)
            # Remove old records for this student to prevent duplicates
            df = df[df['RollNo'].astype(str) != st.session_state['reg_roll']]
            
            new_rows = []
            for sub in SUBJECT_LIST:
                new_rows.append({
                    "RollNo": st.session_state['reg_roll'], "Name": st.session_state['reg_name'],
                    "Subject": sub, "Section": st.session_state['reg_sec'],
                    "Held": 0, "Attended": 0, "LastUpdated": "-"
                })
            df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
            df.to_csv(CSV_FILE, index=False)
            
            st.success("Registration Successful! You can now use Live Attendance.")
            if st.button("Finish"):
                st.session_state['reg_step'] = None
                navigate_to("student_hub")

# --- STUDENT VIEW ---
def page_student_view():
    if st.button("← Back"): navigate_to("student_hub")
    st.markdown("<h3>Attendance Records</h3>", unsafe_allow_html=True)
    roll = st.text_input("Enter Roll Number")
    
    if st.button("Check Attendance"):
        try:
            df = pd.read_csv(CSV_FILE)
            df['RollNo'] = df['RollNo'].astype(str)
            data = df[df['RollNo'] == roll]
            if not data.empty:
                st.markdown(f"#### Student: {data.iloc[0]['Name']} ({roll})")
                table_data = []
                for _, row in data.iterrows():
                    h, a = int(row['Held']), int(row['Attended'])
                    p = (a/h)*100 if h > 0 else 0
                    table_data.append({
                        "Subject": row['Subject'], "Section": row['Section'],
                        "Held": h, "Attended": a, "Percentage": f"{p:.1f}%",
                        "Last Updated": row['LastUpdated']
                    })
                st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
            else: st.error("No records found. Please Register first.")
        except Exception as e: st.error(f"Error: {e}")

# --- FACULTY LOGIN ---
def page_faculty_login():
    if st.button("← Home"): navigate_to("home")
    st.markdown("<h3>Faculty Login</h3>", unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Login"):
            if u == "admin" and p == "admin": navigate_to("faculty_dashboard")
            else: st.error("Invalid Credentials")
        st.markdown('</div>', unsafe_allow_html=True)

# --- FACULTY DASHBOARD ---
def page_faculty_dashboard():
    if st.button("← Logout"): navigate_to("home")
    st.markdown("<h2>Faculty Dashboard</h2>", unsafe_allow_html=True)
    st.markdown('<div class="custom-card">', unsafe_allow_html=True)
    st.markdown("#### Start Attendance Session")
    c1, c2 = st.columns(2)
    with c1: sub = st.selectbox("Subject", SUBJECT_LIST)
    with c2: sec = st.selectbox("Section", ["Section A", "Section B", "CSE-AIML"])
    if st.button("Go to Live Attendance"):
        st.session_state.update({'live_sub': sub, 'live_sec': sec, 'live_run': False})
        navigate_to("live_attendance")
    st.markdown('</div>', unsafe_allow_html=True)

# --- LIVE ATTENDANCE (FIXED) ---
def page_live_attendance():
    c_back, c_title = st.columns([1, 5])
    with c_back: 
        if st.button("← Back"): navigate_to("faculty_dashboard")
    
    sub = st.session_state.get('live_sub')
    sec = st.session_state.get('live_sec')
    st.markdown(f"<h3>Live Class: {sub} ({sec})</h3>", unsafe_allow_html=True)

    col_cam, col_log = st.columns([2, 1])
    
    with col_cam:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        # --- STOP BUTTON ---
        if not st.session_state['live_run']:
            if st.button("▶ START CAMERA", key="start_btn"):
                st.session_state['live_run'] = True
                st.rerun()
        else:
            if st.button("⏹ STOP ATTENDANCE", key="stop_btn"):
                st.session_state['live_run'] = False
                st.rerun()
        video_ph = st.image([])
        st.markdown('</div>', unsafe_allow_html=True)

    with col_log:
        st.markdown('<div class="custom-card">', unsafe_allow_html=True)
        st.markdown("#### Log")
        log_ph = st.empty()
        st.markdown('</div>', unsafe_allow_html=True)

    if st.session_state['live_run']:
        # 1. LOAD DATABASE ONCE (Prevents file locking issues)
        try:
            df = pd.read_csv(CSV_FILE)
            df['RollNo'] = df['RollNo'].astype(str)
        except Exception as e:
            st.error(f"Error reading database: {e}")
            st.session_state['live_run'] = False
            return

        if not os.path.exists(os.path.join(TRAIN_DIR, "trainer.yml")):
            st.error("Model not trained. Please register a student first.")
            st.session_state['live_run'] = False
        else:
            recognizer = cv2.face.LBPHFaceRecognizer_create()
            recognizer.read(os.path.join(TRAIN_DIR, "trainer.yml"))
            face_cascade = cv2.CascadeClassifier(HAAR_FILE)
            cap = cv2.VideoCapture(0)
            logs = []
            last_marked = {}
            
            while st.session_state['live_run']:
                ret, frame = cap.read()
                if not ret: break
                frame = cv2.flip(frame, 1)
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, 1.2, 5)
                
                for (x,y,w,h) in faces:
                    cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)
                    id_internal, conf = recognizer.predict(gray[y:y+h,x:x+w])
                    
                    if conf < 65:
                        real_roll = get_roll(id_internal)
                        now = datetime.now()
                        
                        # UPDATE DB LOGIC
                        # Only update if not marked in the last 60 seconds
                        if real_roll not in last_marked or (now - last_marked[real_roll]).seconds > 60:
                            
                            # 2. CHECK IF STUDENT EXISTS IN CSV FOR THIS SUBJECT
                            mask = (df['RollNo'] == str(real_roll)) & (df['Subject'] == sub)
                            
                            # .empty is safer than .any()
                            if not df.loc[mask].empty:
                                idx = df.index[mask].tolist()[0]
                                
                                # 3. UPDATE IN-MEMORY DATAFRAME
                                df.at[idx, 'Held'] = int(df.at[idx, 'Held']) + 1
                                df.at[idx, 'Attended'] = int(df.at[idx, 'Attended']) + 1
                                df.at[idx, 'LastUpdated'] = now.strftime("%H:%M:%S")
                                
                                # 4. SAVE TO CSV (Only when necessary)
                                try:
                                    df.to_csv(CSV_FILE, index=False)
                                    last_marked[real_roll] = now
                                    logs.insert(0, f"✅ {real_roll} marked at {now.strftime('%H:%M:%S')}")
                                except Exception as e:
                                    logs.insert(0, f"❌ Save Failed: {e}")
                                    st.error(f"Database Save Error: {e}")
                            else:
                                # Student face recognized, but not in CSV for this subject
                                logs.insert(0, f"⚠️ Roll {real_roll} not found in {sub}")
                        
                        cv2.putText(frame, f"Marked: {real_roll}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    else:
                        cv2.putText(frame, "Unknown", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                
                video_ph.image(frame, channels="BGR")
                log_ph.write("\n".join(logs[:15]))
                
            cap.release()
            cv2.destroyAllWindows()

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