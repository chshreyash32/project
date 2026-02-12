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
if 'live_run' not in st.session_state: st.session_state['live_run'] = False
if 'live_periods' not in st.session_state: st.session_state['live_periods'] = 1

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
    
    /* SELECT BOXES */
    div[data-baseweb="select"] > div {
        background-color: #111111;
        color: white;
        border-color: #333;
    }

    /* BUTTONS - PREMIUM BLUE WITH WHITE TEXT */
    .stButton > button {
        background-color: #007bff;
        color: #ffffff !important; /* Force White Text */
        border: none;
        border-radius: 6px;
        padding: 12px 20px;
        font-weight: 600; /* Make text bolder */
        width: 100%;
        transition: all 0.3s ease;
    }
    
    /* Ensure text inside button stays white */
    .stButton > button p {
        color: #ffffff !important;
    }

    .stButton > button:hover {
        background-color: #0056b3;
        color: #ffffff !important;
    }

    /* STOP BUTTON RED */
    div.stButton.stop-btn > button {
        background-color: #dc3545 !important;
        color: #ffffff !important;
    }

    /* TEXT HEADERS */
    h1, h2, h3 { color: white !important; font-family: 'Segoe UI', sans-serif; font-weight: 600; }
    p { color: #888; }
    
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
    st.markdown("<h1 style='text-align: center; margin-bottom: 5px;'>Intelligent Face Recognition Attendance System</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align: center;'>Gokaraju Rangaraju Institute of Engineering and Technology</h4>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Department of CSE(AIML)</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🎓 Student Portal")
        st.markdown("Login to view attendance or Register")
        if st.button("Continue as Student"): navigate_to("student_hub")
    
    with c2:
        st.markdown("### 👨‍🏫 Faculty Login")
        st.markdown("Mark attendance and view reports")
        if st.button("Continue as Faculty"): navigate_to("faculty_login")

# --- STUDENT HUB ---
def page_student_hub():
    if st.button("← Back"): navigate_to("home")
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("<h2>Student Dashboard</h2>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        st.info("Check your attendance status")
        if st.button("View Attendance"): navigate_to("student_view")
    with c2:
        st.info("New student registration")
        if st.button("Register Face"): navigate_to("student_register")

# --- STUDENT REGISTER ---
def page_student_register():
    if st.button("← Back"): navigate_to("student_hub")
    
    if st.session_state.get('reg_step') != 'capture':
        st.markdown("<h3>New Student Registration</h3>", unsafe_allow_html=True)
        with st.container():
            col_form, col_space = st.columns([2, 1])
            with col_form:
                name = st.text_input("Full Name")
                roll = st.text_input("Roll Number")
                section = st.selectbox("Section", SECTION_LIST)
                
                st.markdown("<br>", unsafe_allow_html=True)
                if st.button("Next: Capture Face"):
                    if name and roll:
                        st.session_state.update({'reg_name': name, 'reg_roll': roll, 'reg_sec': section, 'reg_step': 'capture'})
                        st.rerun()
                    else: st.error("Please fill all details.")
            
    else:
        st.markdown(f"<h3>Capturing: {st.session_state['reg_name']}</h3>", unsafe_allow_html=True)
        
        c_cam, c_txt = st.columns([2, 1])
        with c_cam: cam_ph = st.image([])
        with c_txt: 
            st.info("Look at the camera. Capturing 30 images...")
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
            df = pd.read_csv(CSV_FILE)
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
            
            st.success("Registration Successful!")
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
                    h = int(row['Held'])
                    a = int(row['Attended'])
                    p = (a/h)*100 if h > 0 else 0
                    
                    table_data.append({
                        "Subject": row['Subject'],
                        "Section": row['Section'],
                        "Held": h,
                        "Attended": a,
                        "Percentage": f"{p:.1f}%",
                        "Last Updated": row['LastUpdated']
                    })
                
                st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)
            else:
                st.error("No records found. Please register first.")
        except Exception as e:
            st.error(f"Error reading database: {e}")

# --- FACULTY LOGIN ---
def page_faculty_login():
    if st.button("← Home"): navigate_to("home")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    c_spacer, c_login, c_spacer2 = st.columns([1, 2, 1])
    
    with c_login:
        st.markdown("### Faculty Login")
        st.markdown("<p style='color:#888; font-size:14px;'>Enter your 4-digit Faculty ID.</p>", unsafe_allow_html=True)
        
        u = st.text_input("Faculty ID")
        p = st.text_input("Password", type="password")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.button("Login"):
            if len(u) == 4 and u.isdigit() and u == p:
                navigate_to("faculty_dashboard")
            else:
                st.error("Invalid Credentials.")

# --- FACULTY DASHBOARD ---
def page_faculty_dashboard():
    if st.button("← Logout"): navigate_to("home")
    st.markdown("<h2>Faculty Dashboard</h2>", unsafe_allow_html=True)
    st.markdown("#### Setup Attendance Session")
    
    c1, c2 = st.columns(2)
    with c1: sub = st.selectbox("Subject", SUBJECT_LIST)
    with c2: sec = st.selectbox("Section", SECTION_LIST)
    
    c3, c4 = st.columns(2)
    with c3:
        periods = st.selectbox("No. of Periods", [1, 2, 3, 4])
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    if st.button("🚀 Start Live Attendance"):
        st.session_state.update({
            'live_sub': sub, 
            'live_sec': sec, 
            'live_periods': periods,
            'live_run': False
        })
        navigate_to("live_attendance")

# --- LIVE ATTENDANCE (FIXED) ---
def page_live_attendance():
    c_back, c_title = st.columns([1, 5])
    with c_back: 
        if st.button("← Back"): navigate_to("faculty_dashboard")
    
    sub = st.session_state.get('live_sub')
    sec = st.session_state.get('live_sec')
    periods = st.session_state.get('live_periods')
    
    st.markdown(f"<h3>Live Class: {sub} ({sec})</h3>", unsafe_allow_html=True)
    st.markdown(f"**Adding {periods} Period(s) per student**")

    col_cam, col_log = st.columns([2, 1])
    
    with col_cam:
        if not st.session_state['live_run']:
            if st.button("▶ START CAMERA", key="start_btn"):
                st.session_state['live_run'] = True
                st.rerun()
        else:
            # STOP BUTTON (Using standard st.button but will look different due to CSS)
            st.markdown('<div class="stop-btn">', unsafe_allow_html=True)
            if st.button("⏹ STOP ATTENDANCE", key="stop_btn"):
                st.session_state['live_run'] = False
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        
        video_ph = st.image([])

    with col_log:
        st.markdown("#### Log")
        log_ph = st.empty()

    if st.session_state['live_run']:
        if not os.path.exists(os.path.join(TRAIN_DIR, "trainer.yml")):
            st.error("Model not trained.")
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
                        
                        # COOLDOWN 60s
                        if real_roll not in last_marked or (now - last_marked[real_roll]).seconds > 60:
                            try:
                                df = pd.read_csv(CSV_FILE)
                                df['RollNo'] = df['RollNo'].astype(str)
                                mask = (df['RollNo'] == str(real_roll)) & (df['Subject'] == sub)
                                
                                # Use .empty check
                                if not df.loc[mask].empty:
                                    idx = df.index[mask].tolist()[0]
                                    
                                    # Update count
                                    df.at[idx, 'Held'] = int(df.at[idx, 'Held']) + periods
                                    df.at[idx, 'Attended'] = int(df.at[idx, 'Attended']) + periods
                                    df.at[idx, 'LastUpdated'] = now.strftime("%H:%M:%S")
                                    df.to_csv(CSV_FILE, index=False)
                                    
                                    last_marked[real_roll] = now
                                    logs.insert(0, f"✅ {real_roll} (+{periods})")
                                else:
                                    logs.insert(0, f"⚠️ Student {real_roll} not registered for {sub}")
                            except Exception as e:
                                logs.insert(0, f"❌ Error: {str(e)}")
                        
                        cv2.putText(frame, f"Marked: {real_roll}", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    else:
                        cv2.putText(frame, "Unknown", (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
                
                video_ph.image(frame, channels="BGR")
                
                # HTML LOG DISPLAY FOR BETTER VISIBILITY
                log_html = ""
                for l in logs[:10]:
                    color = "#00e676" if "✅" in l else "#ff1744"
                    log_html += f"<div style='color: {color}; margin-bottom: 5px; font-family: monospace;'>{l}</div>"
                log_ph.markdown(log_html, unsafe_allow_html=True)

            cap.release()

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