import streamlit as st
import cv2
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from PIL import Image
import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import threading
import logging
from queue import Queue

# Setup logging
logging.basicConfig(level=logging.INFO)

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
if 'capture_count' not in st.session_state: st.session_state['capture_count'] = 0
if 'attendance_log' not in st.session_state: st.session_state['attendance_log'] = []
if 'last_marked' not in st.session_state: st.session_state['last_marked'] = {}
if 'reg_step' not in st.session_state: st.session_state['reg_step'] = None

# -- WebRTC Configuration (more STUN servers) --
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
    ]
})

def navigate_to(page):
    st.session_state['page'] = page
    st.rerun()

# ==========================================
# 2. CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #e0e0e0; }
    div[data-baseweb="input"] { background-color: #111111; border: 1px solid #333; border-radius: 8px; color: white; }
    div[data-baseweb="select"] > div { background-color: #111111; color: white; border-color: #333; }
    .stButton > button { background-color: #007bff; color: #ffffff !important; border: none; border-radius: 6px; padding: 12px 20px; font-weight: 600; width: 100%; }
    .stButton > button:hover { background-color: #0056b3; }
    h1, h2, h3 { color: white !important; }
    p { color: #888; }
    div[data-testid="stDataFrame"] { background-color: #111; border: 1px solid #333; border-radius: 10px; }
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
    paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR) if f.endswith(".jpg")]
    faces, ids = [], []
    for path in paths:
        try:
            img = Image.open(path).convert('L')
            faces.append(np.array(img, 'uint8'))
            ids.append(int(os.path.split(path)[-1].split(".")[1]))
        except Exception as e:
            logging.warning(f"Failed to load {path}: {e}")
    if faces:
        recognizer.train(faces, np.array(ids))
        recognizer.write(os.path.join(TRAIN_DIR, "trainer.yml"))
        return True
    return False

# ==========================================
# 4. VIDEO PROCESSOR CLASSES
# ==========================================

class RegistrationProcessor:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(HAAR_FILE)
        self.frame_count = 0
        self.captured = 0
        self.capture_queue = Queue()  # to send count to main thread
        self.student_id = None

    def set_student_id(self, sid):
        self.student_id = sid

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4)

                if self.captured < 30 and self.frame_count % 4 == 0 and self.student_id:
                    face_img = gray[y:y+h, x:x+w]
                    filename = f"{DATA_DIR}/User.{self.student_id}.{self.captured + 1}.jpg"
                    cv2.imwrite(filename, face_img)
                    self.captured += 1
                    self.capture_queue.put(self.captured)
                    logging.info(f"Captured image {self.captured}/30")

                cv2.putText(img, f"Captured: {self.captured}/30", 
                            (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

            if len(faces) == 0:
                cv2.putText(img, "No Face Detected", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

            self.frame_count += 1
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            logging.error(f"Frame processing error: {e}")
            return frame

class AttendanceProcessor:
    def __init__(self, subject, periods):
        self.subject = subject
        self.periods = periods
        self.face_cascade = cv2.CascadeClassifier(HAAR_FILE)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        model_path = os.path.join(TRAIN_DIR, "trainer.yml")
        if os.path.exists(model_path):
            self.recognizer.read(model_path)
            self.model_loaded = True
        else:
            self.model_loaded = False
        
        self.frame_count = 0

    def recv(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            
            if not self.model_loaded:
                cv2.putText(img, "Model not trained!", (50, 50), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                return av.VideoFrame.from_ndarray(img, format="bgr24")
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)
            
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 4)
                
                try:
                    id_internal, confidence = self.recognizer.predict(gray[y:y+h, x:x+w])
                    
                    if confidence < 65:
                        roll_no = get_roll(id_internal)
                        current_time = datetime.now()
                        last_marked_time = st.session_state['last_marked'].get(roll_no, None)
                        
                        if (not last_marked_time or (current_time - last_marked_time).seconds > 60) and self.frame_count % 30 == 0:
                            df = pd.read_csv(CSV_FILE)
                            df['RollNo'] = df['RollNo'].astype(str)
                            mask = (df['RollNo'] == str(roll_no)) & (df['Subject'] == self.subject)
                            
                            if not df.loc[mask].empty:
                                idx = df.index[mask].tolist()[0]
                                df.at[idx, 'Held'] = int(df.at[idx, 'Held']) + self.periods
                                df.at[idx, 'Attended'] = int(df.at[idx, 'Attended']) + self.periods
                                df.at[idx, 'LastUpdated'] = current_time.strftime("%H:%M:%S %d-%b-%Y")
                                df.to_csv(CSV_FILE, index=False)
                                
                                st.session_state['last_marked'][roll_no] = current_time
                                
                                log_msg = f"✅ {roll_no} (+{self.periods})"
                                if log_msg not in st.session_state['attendance_log']:
                                    st.session_state['attendance_log'].insert(0, log_msg)
                        
                        cv2.putText(img, str(roll_no), (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    else:
                        cv2.putText(img, "Unknown", (x, y - 10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                except:
                    pass
            
            self.frame_count += 1
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            logging.error(f"Attendance frame error: {e}")
            return frame

# ==========================================
# 5. PAGES
# ==========================================

def page_home():
    st.markdown("<h1 style='text-align: center;'>CSE (AIML) Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Gokaraju Rangaraju Institute of Engineering and Technology</p>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Continue as Student", use_container_width=True): navigate_to("student_hub")
    with c2:
        if st.button("Continue as Faculty", use_container_width=True): navigate_to("faculty_login")

def page_student_hub():
    if st.button("← Back"): navigate_to("home")
    st.markdown("<h2>Student Dashboard</h2>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1:
        if st.button("View Attendance"): navigate_to("student_view")
    with c2:
        if st.button("Register Face"): navigate_to("student_register")

def page_student_register():
    if st.button("← Back"): 
        st.session_state['capture_count'] = 0
        st.session_state['student_internal_id'] = None
        st.session_state['reg_step'] = None
        navigate_to("student_hub")
    
    if st.session_state.get('reg_step') != 'capture':
        st.markdown("<h3>New Student Registration</h3>", unsafe_allow_html=True)
        name = st.text_input("Full Name")
        roll = st.text_input("Roll Number").strip().upper()
        section = st.selectbox("Section", SECTION_LIST)
        
        if st.button("Next: Auto-Capture Face"):
            if name and roll:
                final_id = save_mapping(get_next_id(), roll)
                st.session_state.update({
                    'reg_name': name, 
                    'reg_roll': roll, 
                    'reg_sec': section, 
                    'reg_step': 'capture',
                    'capture_count': 0,
                    'student_internal_id': final_id
                })
                st.rerun()
            else:
                st.error("Please fill all details.")
    else:
        st.markdown(f"<h3>Auto-Capturing: {st.session_state.get('reg_name', 'Student')}</h3>", unsafe_allow_html=True)
        
        c_cam, c_txt = st.columns([3, 2])
        
        with c_txt:
            count = st.session_state.get('capture_count', 0)
            st.markdown(f"### Progress: {count}/30")
            st.progress(count / 30)
            
            st.success("🟢 GREEN box = face detected")
            st.info("📸 Auto-capturing — move head slightly")
            
            if count >= 30:
                st.success("✅ Capture Complete!")
        
        with c_cam:
            processor = RegistrationProcessor()
            if 'student_internal_id' in st.session_state:
                processor.set_student_id(st.session_state['student_internal_id'])
            
            video_constraints = {
                "video": {
                    "width": {"ideal": 640},
                    "height": {"ideal": 480},
                    "frameRate": {"ideal": 24}
                },
                "audio": False
            }
            
            webrtc_ctx = webrtc_streamer(
                key="registration",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=lambda: processor,
                media_stream_constraints=video_constraints,
                async_processing=True,
            )
            
            # Poll queue for capture count updates
            if webrtc_ctx and webrtc_ctx.video_processor:
                q = webrtc_ctx.video_processor.capture_queue
                updated = False
                while not q.empty():
                    new_count = q.get()
                    st.session_state['capture_count'] = new_count
                    updated = True
                if updated:
                    st.rerun()
            
            if st.session_state.get('capture_count', 0) >= 30 and webrtc_ctx:
                st.success("Processing registration...")
                if train_model():
                    df = pd.read_csv(CSV_FILE)
                    df = df[df['RollNo'].astype(str) != st.session_state['reg_roll']]
                    
                    new_rows = []
                    for sub in SUBJECT_LIST:
                        new_rows.append({
                            "RollNo": st.session_state['reg_roll'], 
                            "Name": st.session_state['reg_name'],
                            "Subject": sub, 
                            "Section": st.session_state['reg_sec'],
                            "Held": 0, 
                            "Attended": 0, 
                            "LastUpdated": "-"
                        })
                    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                    df.to_csv(CSV_FILE, index=False)
                    
                    st.success("✅ Registration Successful!")
                    st.balloons()
                    if st.button("Finish & Go Back"):
                        st.session_state['reg_step'] = None
                        st.session_state['capture_count'] = 0
                        st.session_state['student_internal_id'] = None
                        navigate_to("student_hub")

def page_student_view():
    if st.button("← Back"): navigate_to("student_hub")
    st.markdown("<h3>Check Attendance</h3>", unsafe_allow_html=True)
    
    roll = st.text_input("Enter Roll Number").strip().upper()
    
    if st.button("View Records"):
        try:
            df = pd.read_csv(CSV_FILE)
            df['RollNo'] = df['RollNo'].astype(str)
            data = df[df['RollNo'] == roll]
            
            if not data.empty:
                st.markdown(f"**Student:** {data.iloc[0]['Name']} ({roll})")
                table_data = []
                for _, row in data.iterrows():
                    h = int(row['Held'])
                    a = int(row['Attended'])
                    p = (a / h * 100) if h > 0 else 0
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
                st.error("No records found. Register first.")
        except Exception as e:
            st.error(f"Error: {e}")

def page_faculty_login():
    if st.button("← Home"): navigate_to("home")
    
    st.markdown("### Faculty Login")
    u = st.text_input("Faculty ID (4 digits)")
    p = st.text_input("Password", type="password")
    
    if st.button("Login"):
        if len(u) == 4 and u.isdigit() and u == p:
            navigate_to("faculty_dashboard")
        else:
            st.error("Invalid credentials.")

def page_faculty_dashboard():
    if st.button("← Logout"): navigate_to("home")
    st.markdown("<h2>Faculty Dashboard</h2>", unsafe_allow_html=True)
    
    c1, c2 = st.columns(2)
    with c1: sub = st.selectbox("Subject", SUBJECT_LIST)
    with c2: sec = st.selectbox("Section", SECTION_LIST)
    
    periods = st.selectbox("Periods to mark", [1, 2, 3, 4])
    
    if st.button("🚀 Start Live Attendance"):
        st.session_state.update({
            'live_sub': sub, 
            'live_sec': sec, 
            'live_periods': periods,
            'attendance_log': [],
            'last_marked': {}
        })
        navigate_to("live_attendance")

def page_live_attendance():
    if st.button("← Back"): 
        st.session_state['last_marked'] = {}
        navigate_to("faculty_dashboard")
    
    sub = st.session_state.get('live_sub', 'Unknown')
    sec = st.session_state.get('live_sec', 'Unknown')
    periods = st.session_state.get('live_periods', 1)
    
    st.markdown(f"<h3>Live Attendance: {sub} ({sec}) — +{periods} per detection</h3>", unsafe_allow_html=True)

    col_cam, col_log = st.columns([3, 2])
    
    with col_log:
        st.markdown("#### Attendance Log")
        for log in st.session_state['attendance_log'][:15]:
            st.success(log)
        if not st.session_state['attendance_log']:
            st.info("Waiting for recognized faces...")

    with col_cam:
        if not os.path.exists(os.path.join(TRAIN_DIR, "trainer.yml")):
            st.error("Model not trained — register students first.")
        else:
            webrtc_streamer(
                key="attendance",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=lambda: AttendanceProcessor(sub, periods),
                media_stream_constraints={
                    "video": {"width": {"ideal": 640}, "height": {"ideal": 480}, "frameRate": {"ideal": 20}},
                    "audio": False
                },
                async_processing=True,
            )
            st.success("Camera active — green boxes = marked")

# ==========================================
# 6. ROUTER
# ==========================================
page_map = {
    "home": page_home,
    "student_hub": page_student_hub,
    "student_register": page_student_register,
    "student_view": page_student_view,
    "faculty_login": page_faculty_login,
    "faculty_dashboard": page_faculty_dashboard,
    "live_attendance": page_live_attendance
}

page_map[st.session_state['page']]()
