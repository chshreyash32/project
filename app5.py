import streamlit as st
import cv2
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from PIL import Image
import threading

# --- NEW IMPORTS FOR AUTOMATIC VIDEO ---
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration
    import av
except ImportError:
    st.error("⚠️ Libraries missing! Please run: pip install streamlit-webrtc av")
    st.stop()

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
SUBJECT_LIST = ["Machine Learning", "Big Data Analytics", "Software Engineering", "DSRP", "JCP", "ML Lab", "BDA Lab", "Mini Project", "Constitution of India"]
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
if 'live_periods' not in st.session_state: st.session_state['live_periods'] = 1

# -- GLOBAL LOCK FOR THREAD SAFETY --
lock = threading.Lock()

def navigate_to(page):
    st.session_state['page'] = page
    st.rerun()

# ==========================================
# 2. CUSTOM CSS
# ==========================================
st.markdown("""
<style>
    .stApp { background-color: #000000; color: #e0e0e0; }
    div[data-baseweb="input"] { background-color: #111111; border: 1px solid #333; color: white; }
    .stButton > button { background-color: #007bff; color: white !important; border: none; padding: 10px 20px; width: 100%; }
    .stButton > button:hover { background-color: #0056b3; }
    div[data-testid="stDataFrame"] { background-color: #111; }
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
    if not os.listdir(DATA_DIR): return False
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    paths = [os.path.join(DATA_DIR, f) for f in os.listdir(DATA_DIR)]
    faces, ids = [], []
    for path in paths:
        try:
            img = np.array(Image.open(path).convert('L'), 'uint8')
            id_val = int(os.path.split(path)[-1].split(".")[1])
            faces.append(img)
            ids.append(id_val)
        except: pass
    if faces:
        recognizer.train(faces, np.array(ids))
        recognizer.write(os.path.join(TRAIN_DIR, "trainer.yml"))
        return True
    return False

# ==========================================
# 4. WEBRTC PROCESSORS (THE MAGIC PART)
# ==========================================

# -- REGISTRATION PROCESSOR --
class RegistrationProcessor(VideoTransformerBase):
    def __init__(self):
        self.count = 0
        self.limit = 30
        self.roll = st.session_state.get('reg_roll', 'Unknown')
        # We need to determine ID only once per session
        self.internal_id = None 

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        detector = cv2.CascadeClassifier(HAAR_FILE)
        faces = detector.detectMultiScale(gray, 1.3, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            
            if self.count < self.limit:
                # Lazy load ID to avoid reading file on every frame
                if self.internal_id is None:
                    # NOTE: Writing to JSON inside thread is risky, assume handled before or use lock
                    with lock:
                        self.internal_id = save_mapping(get_next_id(), self.roll)
                
                self.count += 1
                fname = f"{DATA_DIR}/User.{self.internal_id}.{self.count}.jpg"
                cv2.imwrite(fname, gray[y:y+h,x:x+w])
                cv2.putText(img, f"Capturing: {self.count}/{self.limit}", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
            else:
                cv2.putText(img, "Done! Stop Camera", (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# -- ATTENDANCE PROCESSOR --
class AttendanceProcessor(VideoTransformerBase):
    def __init__(self):
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        if os.path.exists(os.path.join(TRAIN_DIR, "trainer.yml")):
            self.recognizer.read(os.path.join(TRAIN_DIR, "trainer.yml"))
        self.detector = cv2.CascadeClassifier(HAAR_FILE)
        self.last_marked = {}
        # Get context from session state (passed via external check mostly, but tricky in thread)
        # We will assume global/session state is roughly valid or set defaults
        self.subject = "Subject" 
        self.periods = 1
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector.detectMultiScale(gray, 1.2, 5)

        for (x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
            try:
                id_internal, conf = self.recognizer.predict(gray[y:y+h,x:x+w])
                if conf < 65:
                    roll = get_roll(id_internal)
                    cv2.putText(img, f"{roll}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                    
                    # MARK ATTENDANCE LOGIC
                    now = datetime.now()
                    if roll not in self.last_marked or (now - self.last_marked[roll]).seconds > 60:
                        with lock: # Thread safe CSV writing
                            df = pd.read_csv(CSV_FILE)
                            df['RollNo'] = df['RollNo'].astype(str)
                            # NOTE: In real deployment, pass subject/periods safely. 
                            # Here we rely on file state or defaults for simplicity in this snippet.
                            # For robust app, these should be passed to __init__
                            pass 
                        
                        # Just visual feedback here because passing 'sub' into this thread is complex 
                        # without factory. We will mark 'Pending' visually.
                        self.last_marked[roll] = now
                else:
                    cv2.putText(img, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            except:
                pass
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ==========================================
# 5. PAGES
# ==========================================

# --- HOME ---
def page_home():
    st.markdown("<h1 style='text-align: center;'>Face Recognition Attendance</h1>", unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Student Portal"): navigate_to("student_hub")
    with c2:
        if st.button("Faculty Portal"): navigate_to("faculty_login")

# --- STUDENT HUB ---
def page_student_hub():
    if st.button("← Back"): navigate_to("home")
    st.markdown("## Student Dashboard")
    c1, c2 = st.columns(2)
    with c1: 
        if st.button("View Attendance"): navigate_to("student_view")
    with c2: 
        if st.button("Register Face"): navigate_to("student_register")

# --- STUDENT REGISTER ---
def page_student_register():
    if st.button("← Back"): navigate_to("student_hub")
    
    if st.session_state.get('reg_step') != 'capture':
        st.markdown("### New Registration")
        name = st.text_input("Name")
        roll = st.text_input("Roll No")
        sec = st.selectbox("Section", SECTION_LIST)
        if st.button("Start Capture"):
            if name and roll:
                st.session_state.update({'reg_name': name, 'reg_roll': roll, 'reg_sec': sec, 'reg_step': 'capture'})
                st.rerun()
            else: st.error("Fill details")
    else:
        st.markdown(f"### Capturing: {st.session_state['reg_name']}")
        st.info("Allow camera access. The system will auto-capture 30 frames.")
        
        # WEBRTC STREAMER FOR REGISTRATION
        ctx = webrtc_streamer(
            key="registration",
            video_processor_factory=RegistrationProcessor,
            rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        )
        
        if st.button("Training Complete? Finish"):
            if train_model():
                # Add Metadata to CSV
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
                st.success("Registered Successfully!")
                st.session_state['reg_step'] = None
                navigate_to("student_hub")
            else:
                st.error("No face data found. Did you capture enough?")

# --- STUDENT VIEW ---
def page_student_view():
    if st.button("← Back"): navigate_to("student_hub")
    roll = st.text_input("Roll Number")
    if st.button("Check"):
        df = pd.read_csv(CSV_FILE)
        data = df[df['RollNo'].astype(str) == roll]
        st.dataframe(data)

# --- FACULTY LOGIN ---
def page_faculty_login():
    if st.button("← Home"): navigate_to("home")
    u = st.text_input("ID")
    p = st.text_input("Password", type="password")
    if st.button("Login"):
        if u == "1234" and p == "1234": navigate_to("faculty_dashboard")
        else: st.error("Invalid")

# --- FACULTY DASHBOARD ---
def page_faculty_dashboard():
    if st.button("Logout"): navigate_to("home")
    st.markdown("## Faculty Dashboard")
    sub = st.selectbox("Subject", SUBJECT_LIST)
    periods = st.selectbox("Periods", [1,2,3])
    if st.button("Start Live Attendance"):
        st.session_state.update({'live_sub': sub, 'live_periods': periods})
        navigate_to("live_attendance")

# --- LIVE ATTENDANCE (AUTOMATIC) ---
def page_live_attendance():
    if st.button("← Back"): navigate_to("faculty_dashboard")
    st.markdown(f"### Live: {st.session_state['live_sub']}")
    
    # Passing arguments to the thread using a closure/factory
    class LiveProcessor(VideoTransformerBase):
        def __init__(self):
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            if os.path.exists(os.path.join(TRAIN_DIR, "trainer.yml")):
                self.recognizer.read(os.path.join(TRAIN_DIR, "trainer.yml"))
            self.detector = cv2.CascadeClassifier(HAAR_FILE)
            self.last_marked = {}
            self.sub = st.session_state['live_sub']
            self.per = st.session_state['live_periods']

        def recv(self, frame):
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = self.detector.detectMultiScale(gray, 1.2, 5)

            for (x,y,w,h) in faces:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                try:
                    id_internal, conf = self.recognizer.predict(gray[y:y+h,x:x+w])
                    if conf < 65:
                        roll = get_roll(id_internal)
                        now = datetime.now()
                        
                        # COOLDOWN & DB WRITE
                        if roll not in self.last_marked or (now - self.last_marked[roll]).seconds > 60:
                            with lock:
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
                        
                        # STATUS ON VIDEO
                        status = "Marked" if (roll in self.last_marked and (now - self.last_marked[roll]).seconds < 60) else "Processing"
                        color = (0,255,0) if status == "Marked" else (0,255,255)
                        cv2.putText(img, f"{roll}: {status}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    else:
                        cv2.putText(img, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
                except: pass
            
            return av.VideoFrame.from_ndarray(img, format="bgr24")

    # START STREAM
    webrtc_streamer(
        key="attendance",
        video_processor_factory=LiveProcessor,
        rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
    )

# ==========================================
# 6. ROUTER
# ==========================================
if st.session_state['page'] == "home": page_home()
elif st.session_state['page'] == "student_hub": page_student_hub()
elif st.session_state['page'] == "student_register": page_student_register()
elif st.session_state['page'] == "student_view": page_student_view()
elif st.session_state['page'] == "faculty_login": page_faculty_login()
elif st.session_state['page'] == "faculty_dashboard": page_faculty_dashboard()
elif st.session_state['page'] == "live_attendance": page_live_attendance()
