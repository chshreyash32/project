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

# -- WebRTC Configuration --
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

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
        color: #ffffff !important;
        border: none;
        border-radius: 6px;
        padding: 12px 20px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
    }
    
    .stButton > button p {
        color: #ffffff !important;
    }

    .stButton > button:hover {
        background-color: #0056b3;
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
    
    /* VIDEO FRAME */
    video {
        border: 3px solid #00ff00;
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
# 4. VIDEO PROCESSOR CLASSES
# ==========================================

class RegistrationProcessor:
    """Processes video frames for student registration with auto-capture"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(HAAR_FILE)
        self.frame_count = 0
        self.lock = threading.Lock()
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        with self.lock:
            student_id = st.session_state.get('student_internal_id')
            current_count = st.session_state.get('capture_count', 0)
            
            for (x, y, w, h) in faces:
                # Draw GREEN box around face
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                # Auto-capture every 15 frames (about 0.5 seconds)
                if current_count < 30 and self.frame_count % 15 == 0 and student_id:
                    try:
                        # Save face image
                        face_img = gray[y:y+h, x:x+w]
                        cv2.imwrite(f"{DATA_DIR}/User.{student_id}.{current_count + 1}.jpg", face_img)
                        st.session_state['capture_count'] = current_count + 1
                    except Exception as e:
                        print(f"Error saving image: {e}")
                
                # Display count on video
                cv2.putText(img, f"Captured: {st.session_state.get('capture_count', 0)}/30", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Display message if no face detected
            if len(faces) == 0:
                cv2.putText(img, "No Face Detected", (50, 50), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            self.frame_count += 1
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


class AttendanceProcessor:
    """Processes video frames for live attendance with face recognition"""
    
    def __init__(self, subject, periods):
        self.subject = subject
        self.periods = periods
        self.face_cascade = cv2.CascadeClassifier(HAAR_FILE)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        
        # Load trained model
        model_path = os.path.join(TRAIN_DIR, "trainer.yml")
        if os.path.exists(model_path):
            self.recognizer.read(model_path)
            self.model_loaded = True
        else:
            self.model_loaded = False
        
        self.lock = threading.Lock()
        self.frame_count = 0
        
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        if not self.model_loaded:
            cv2.putText(img, "Model not trained!", (50, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            return av.VideoFrame.from_ndarray(img, format="bgr24")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.2, 5)
        
        with self.lock:
            for (x, y, w, h) in faces:
                # Draw GREEN box
                cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 3)
                
                try:
                    # Recognize face
                    id_internal, confidence = self.recognizer.predict(gray[y:y+h, x:x+w])
                    
                    if confidence < 65:  # Good match
                        roll_no = get_roll(id_internal)
                        
                        # Mark attendance (with cooldown)
                        current_time = datetime.now()
                        last_marked_time = st.session_state['last_marked'].get(roll_no)
                        
                        # Mark if not marked in last 60 seconds
                        if not last_marked_time or (current_time - last_marked_time).seconds > 60:
                            # Only mark every 30 frames (about 1 second)
                            if self.frame_count % 30 == 0:
                                try:
                                    df = pd.read_csv(CSV_FILE)
                                    df['RollNo'] = df['RollNo'].astype(str)
                                    mask = (df['RollNo'] == str(roll_no)) & (df['Subject'] == self.subject)
                                    
                                    if not df.loc[mask].empty:
                                        idx = df.index[mask].tolist()[0]
                                        df.at[idx, 'Held'] = int(df.at[idx, 'Held']) + self.periods
                                        df.at[idx, 'Attended'] = int(df.at[idx, 'Attended']) + self.periods
                                        df.at[idx, 'LastUpdated'] = current_time.strftime("%H:%M:%S")
                                        df.to_csv(CSV_FILE, index=False)
                                        
                                        # Update session state
                                        st.session_state['last_marked'][roll_no] = current_time
                                        
                                        # Add to log
                                        log_msg = f"✅ {roll_no} (+{self.periods})"
                                        if log_msg not in st.session_state['attendance_log']:
                                            st.session_state['attendance_log'].insert(0, log_msg)
                                except Exception as e:
                                    print(f"Error marking attendance: {e}")
                        
                        # Display name on video
                        cv2.putText(img, str(roll_no), (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.putText(img, f"Conf: {int(confidence)}", (x, y + h + 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    else:
                        # Unknown face
                        cv2.putText(img, "Unknown", (x, y - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                        
                except Exception as e:
                    print(f"Recognition error: {e}")
            
            # Show frame count
            cv2.putText(img, f"Frame: {self.frame_count}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            self.frame_count += 1
        
        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ==========================================
# 5. PAGES
# ==========================================

# --- HOME ---
def page_home():
    st.markdown("<h1 style='text-align: center; margin-bottom: 5px;'>CSE (AIML) Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Gokaraju Rangaraju Institute of Engineering and Technology</p>", unsafe_allow_html=True)
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

# --- STUDENT REGISTER (WITH WEBRTC AUTO-CAPTURE) ---
def page_student_register():
    if st.button("← Back"): 
        st.session_state['capture_count'] = 0
        st.session_state['student_internal_id'] = None
        navigate_to("student_hub")
    
    if st.session_state.get('reg_step') != 'capture':
        st.markdown("<h3>New Student Registration</h3>", unsafe_allow_html=True)
        with st.container():
            col_form, col_space = st.columns([2, 1])
            with col_form:
                name = st.text_input("Full Name")
                roll = st.text_input("Roll Number")
                section = st.selectbox("Section", SECTION_LIST)
                
                st.markdown("<br>", unsafe_allow_html=True)
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
                    else: st.error("Please fill all details.")
            
    else:
        st.markdown(f"<h3>🎥 Auto-Capturing: {st.session_state['reg_name']}</h3>", unsafe_allow_html=True)
        
        c_cam, c_txt = st.columns([2, 1])
        
        with c_txt:
            st.markdown(f"### Progress: {st.session_state.get('capture_count', 0)}/30")
            st.progress(st.session_state.get('capture_count', 0) / 30)
            
            st.markdown("**Instructions:**")
            st.success("🟢 GREEN box shows your face")
            st.info("📸 Auto-capturing in progress...")
            st.markdown("- Look at the camera")
            st.markdown("- Move your face slightly")
            st.markdown("- Different angles help")
            
            if st.session_state.get('capture_count', 0) >= 30:
                st.success("✅ Capture Complete!")
        
        with c_cam:
            if st.session_state.get('capture_count', 0) < 30:
                # Start WebRTC stream with auto-capture
                webrtc_ctx = webrtc_streamer(
                    key="registration",
                    mode=WebRtcMode.SENDRECV,
                    rtc_configuration=RTC_CONFIGURATION,
                    video_processor_factory=RegistrationProcessor,
                    async_processing=True,
                    media_stream_constraints={"video": True, "audio": False},
                )
                
            else:
                st.success("🎉 All 30 images captured! Processing...")
                
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
                    if st.button("Finish"):
                        st.session_state['reg_step'] = None
                        st.session_state['capture_count'] = 0
                        st.session_state['student_internal_id'] = None
                        navigate_to("student_hub")
                else:
                    st.error("Model training failed. Please try again.")

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
            'attendance_log': [],
            'last_marked': {}
        })
        navigate_to("live_attendance")

# --- LIVE ATTENDANCE (WITH WEBRTC) ---
def page_live_attendance():
    if st.button("← Back"): 
        st.session_state['last_marked'] = {}
        navigate_to("faculty_dashboard")
    
    sub = st.session_state.get('live_sub')
    sec = st.session_state.get('live_sec')
    periods = st.session_state.get('live_periods')
    
    st.markdown(f"<h3>🎥 Live Class: {sub} ({sec})</h3>", unsafe_allow_html=True)
    st.markdown(f"**Adding {periods} Period(s) per student**")

    col_cam, col_log = st.columns([2, 1])
    
    with col_log:
        st.markdown("#### 📋 Attendance Log")
        log_container = st.container()
        with log_container:
            if st.session_state['attendance_log']:
                for log in st.session_state['attendance_log'][:20]:
                    if "✅" in log:
                        st.success(log)
                    else:
                        st.error(log)
            else:
                st.info("Waiting for faces...")

    with col_cam:
        if not os.path.exists(os.path.join(TRAIN_DIR, "trainer.yml")):
            st.error("❌ Model not trained. Register students first.")
        else:
            st.success("🟢 GREEN boxes will appear on recognized faces!")
            st.info("📹 Attendance marks automatically when faces detected")
            
            # Create processor with current session data
            def processor_factory():
                return AttendanceProcessor(sub, periods)
            
            # Start WebRTC stream
            webrtc_ctx = webrtc_streamer(
                key="attendance",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                video_processor_factory=processor_factory,
                async_processing=True,
                media_stream_constraints={"video": True, "audio": False},
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
