import streamlit as st
import cv2
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from PIL import Image
import time

# --- WEBRTC IMPORTS ---
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
    import av
except ImportError:
    st.error("⚠️ Install requirements: pip install streamlit-webrtc av opencv-python-headless")
    st.stop()

# ==========================================
# 1. CONFIGURATION
# ==========================================
st.set_page_config(layout="wide", page_title="CSE Attendance", page_icon="📷")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(BASE_DIR, "trainer")
CSV_FILE = os.path.join(BASE_DIR, "attendance.csv")
MAPPING_FILE = os.path.join(BASE_DIR, "student_map.json")
HAAR_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

SUBJECT_LIST = ["Machine Learning", "Big Data Analytics", "Software Engineering", "DSRP", "JCP", "Mini Project"]
SECTION_LIST = ["Section A", "Section B", "Section C"]

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)

# Initialize Files
if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["RollNo", "Name", "Subject", "Section", "Held", "Attended", "LastUpdated"]).to_csv(CSV_FILE, index=False)
if not os.path.exists(MAPPING_FILE):
    with open(MAPPING_FILE, 'w') as f: json.dump({}, f)

# Session State
if 'page' not in st.session_state: st.session_state['page'] = 'home'

# WebRTC Config (Fixes camera connection issues)
RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# ==========================================
# 2. HELPER FUNCTIONS
# ==========================================
def navigate_to(page):
    st.session_state['page'] = page
    st.rerun()

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
# 3. VIDEO PROCESSORS
# ==========================================

# --- REGISTRATION PROCESSOR ---
class RegistrationProcessor(VideoTransformerBase):
    def __init__(self):
        self.count = 0
        self.limit = 30
        self.face_cascade = cv2.CascadeClassifier(HAAR_FILE)
        # We need these from session state, but safe to access once at init
        self.roll = st.session_state.get('reg_roll', 'Unknown')
        self.save_id = st.session_state.get('reg_id', 1)
        self.last_save_time = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            
            # Logic: Capture automatically if limit not reached
            current_time = time.time()
            if self.count < self.limit and (current_time - self.last_save_time) > 0.2: # 0.2s delay
                self.count += 1
                fname = f"{DATA_DIR}/User.{self.save_id}.{self.count}.jpg"
                cv2.imwrite(fname, gray[y:y+h,x:x+w])
                self.last_save_time = current_time

            # Draw status on video
            status_text = f"Captured: {self.count}/{self.limit}"
            color = (0, 255, 255) if self.count < self.limit else (0, 255, 0)
            cv2.putText(img, status_text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
            if self.count >= self.limit:
                cv2.putText(img, "DONE! Click Stop", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- ATTENDANCE PROCESSOR ---
class AttendanceProcessor(VideoTransformerBase):
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        try:
            self.recognizer.read(os.path.join(TRAIN_DIR, "trainer.yml"))
            self.model_loaded = True
        except:
            self.model_loaded = False
        
        self.face_cascade = cv2.CascadeClassifier(HAAR_FILE)
        self.last_marked = {}
        # Get context from simple file read or safe default
        self.sub = st.session_state.get('live_sub', 'Subject')
        self.per = st.session_state.get('live_periods', 1)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        if not self.model_loaded:
            cv2.putText(img, "Model Not Found!", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            id_internal, conf = self.recognizer.predict(gray[y:y+h,x:x+w])

            if conf < 65:
                roll = get_roll(id_internal)
                now = datetime.now()
                
                # Check Cooldown
                if roll not in self.last_marked or (now - self.last_marked[roll]).seconds > 60:
                    # Update CSV
                    try:
                        df = pd.read_csv(CSV_FILE)
                        df['RollNo'] = df['RollNo'].astype(str)
                        mask = (df['RollNo'] == str(roll)) & (df['Subject'] == self.sub)
                        
                        if not df.loc[mask].empty:
                            idx = df.index[mask].tolist()[0]
                            df.at[idx, 'Held'] = int(df.at[idx, 'Held']) + self.per
                            df.at[idx, 'Attended'] = int(df.at[idx, 'Attended']) + self.per
                            df.at[idx, 'LastUpdated'] = now.strftime("%H:%M:%S")
                            df.to_csv(CSV_FILE, index=False)
                            self.last_marked[roll] = now
                    except: pass # Ignore file write errors for stability
                
                # Feedback
                is_recent = roll in self.last_marked and (now - self.last_marked[roll]).seconds < 60
                color = (0,255,0) if is_recent else (0,255,255)
                text = f"{roll} {'(Marked)' if is_recent else ''}"
                cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            else:
                cv2.putText(img, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 4. PAGES
# ==========================================
st.markdown("<style>.stApp{background-color:black; color:white;}</style>", unsafe_allow_html=True)

if st.session_state['page'] == "home":
    st.title("Face Attendance System")
    c1, c2 = st.columns(2)
    if c1.button("Student Portal"): navigate_to("student_hub")
    if c2.button("Faculty Login"): navigate_to("faculty_login")

elif st.session_state['page'] == "student_hub":
    if st.button("Back"): navigate_to("home")
    st.title("Student Hub")
    if st.button("Register New Student"): navigate_to("student_register")
    if st.button("Check Attendance"): navigate_to("student_view")

elif st.session_state['page'] == "student_register":
    if st.button("Back"): navigate_to("student_hub")
    st.subheader("Register Face")
    
    if 'reg_step' not in st.session_state:
        name = st.text_input("Name")
        roll = st.text_input("Roll No")
        sec = st.selectbox("Section", SECTION_LIST)
        if st.button("Next"):
            if name and roll:
                # Prepare ID
                internal_id = save_mapping(get_next_id(), roll)
                st.session_state.update({'reg_name': name, 'reg_roll': roll, 'reg_sec': sec, 'reg_id': internal_id, 'reg_step': 'capture'})
                st.rerun()
    else:
        st.info(f"Registering: {st.session_state['reg_name']}. Look at camera!")
        
        # WEBRTC STREAMER
        ctx = webrtc_streamer(
            key="registration", 
            video_processor_factory=RegistrationProcessor, 
            rtc_configuration=RTC_CONFIG
        )
        
        st.warning("Wait until video says 'DONE! Click Stop', then click Stop button on camera.")
        
        if st.button("Save & Train Model"):
            with st.spinner("Training..."):
                if train_model():
                    # Add Entry
                    df = pd.read_csv(CSV_FILE)
                    df = df[df['RollNo'].astype(str) != st.session_state['reg_roll']]
                    new_rows = []
                    for sub in SUBJECT_LIST:
                        new_rows.append({"RollNo": st.session_state['reg_roll'], "Name": st.session_state['reg_name'], "Subject": sub, "Section": st.session_state['reg_sec'], "Held": 0, "Attended": 0, "LastUpdated": "-"})
                    df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
                    df.to_csv(CSV_FILE, index=False)
                    st.success("Registered Successfully!")
                    del st.session_state['reg_step']
                    navigate_to("student_hub")
                else:
                    st.error("No faces captured. Try again.")

elif st.session_state['page'] == "student_view":
    if st.button("Back"): navigate_to("student_hub")
    roll = st.text_input("Enter Roll No")
    if st.button("Check"):
        df = pd.read_csv(CSV_FILE)
        st.dataframe(df[df['RollNo'].astype(str) == roll])

elif st.session_state['page'] == "faculty_login":
    if st.button("Back"): navigate_to("home")
    if st.text_input("Password", type="password") == "1234":
        navigate_to("faculty_dashboard")

elif st.session_state['page'] == "faculty_dashboard":
    if st.button("Logout"): navigate_to("home")
    sub = st.selectbox("Subject", SUBJECT_LIST)
    per = st.number_input("Periods", 1, 4, 1)
    if st.button("Go Live"):
        st.session_state.update({'live_sub': sub, 'live_periods': per})
        navigate_to("live_attendance")

elif st.session_state['page'] == "live_attendance":
    if st.button("Back"): navigate_to("faculty_dashboard")
    st.subheader(f"Live: {st.session_state['live_sub']}")
    
    webrtc_streamer(
        key="attendance",
        video_processor_factory=AttendanceProcessor,
        rtc_configuration=RTC_CONFIG
    )
    
    st.markdown("If you see **Green Text (Marked)** on the video, attendance is saved.")
