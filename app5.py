import streamlit as st
import cv2
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from PIL import Image

# ==========================================
# CONFIG
# ==========================================
st.set_page_config(layout="wide", page_title="CSE (AIML) Attendance", page_icon="📷")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(BASE_DIR, "trainer")
CSV_FILE = os.path.join(BASE_DIR, "attendance.csv")
MAPPING_FILE = os.path.join(BASE_DIR, "student_map.json")
HAAR_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

SUBJECT_LIST = [
    "Machine Learning","Big Data Analytics","Software Engineering",
    "DSRP","JCP","ML Lab","BDA Lab","Mini Project","Constitution of India"
]

SECTION_LIST = ["Section A", "Section B", "Section C"]

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)

if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["RollNo","Name","Subject","Section","Held","Attended","LastUpdated"]).to_csv(CSV_FILE,index=False)

if not os.path.exists(MAPPING_FILE):
    with open(MAPPING_FILE,'w') as f: json.dump({},f)

if 'page' not in st.session_state: st.session_state.page="home"
if 'capture_count' not in st.session_state: st.session_state.capture_count=0

def navigate_to(p):
    st.session_state.page=p
    st.rerun()

# ==========================================
# BACKEND
# ==========================================
def get_next_id():
    with open(MAPPING_FILE,'r') as f: data=json.load(f)
    if not data: return 1
    return max([int(k) for k in data.keys()])+1

def save_mapping(i,r):
    with open(MAPPING_FILE,'r') as f: data=json.load(f)
    data[str(i)]=r
    with open(MAPPING_FILE,'w') as f: json.dump(data,f)
    return i

def get_roll(i):
    with open(MAPPING_FILE,'r') as f: data=json.load(f)
    return data.get(str(i),"Unknown")

def train_model():
    recognizer=cv2.face.LBPHFaceRecognizer_create()
    paths=[os.path.join(DATA_DIR,f) for f in os.listdir(DATA_DIR)]
    faces,ids=[],[]
    for p in paths:
        try:
            img=Image.open(p).convert('L')
            faces.append(np.array(img,'uint8'))
            ids.append(int(os.path.split(p)[-1].split(".")[1]))
        except: pass
    if faces:
        recognizer.train(faces,np.array(ids))
        recognizer.write(os.path.join(TRAIN_DIR,"trainer.yml"))
        return True
    return False

# ==========================================
# HOME
# ==========================================
def page_home():
    st.title("Intelligent Face Recognition Attendance System")

    c1,c2=st.columns(2)
    with c1:
        if st.button("Student Portal"):
            navigate_to("student_hub")
    with c2:
        if st.button("Faculty Login"):
            navigate_to("faculty_login")

# ==========================================
# STUDENT HUB
# ==========================================
def page_student_hub():
    if st.button("Back"): navigate_to("home")

    c1,c2=st.columns(2)
    with c1:
        if st.button("View Attendance"): navigate_to("student_view")
    with c2:
        if st.button("Register Face"): navigate_to("student_register")

# ==========================================
# ✅ STUDENT REGISTER (FIXED CAMERA)
# ==========================================
def page_student_register():
    if st.button("Back"): navigate_to("student_hub")

    name=st.text_input("Name")
    roll=st.text_input("Roll")
    section=st.selectbox("Section",SECTION_LIST)

    if name and roll:
        detector=cv2.CascadeClassifier(HAAR_FILE)
        final_id=save_mapping(get_next_id(),roll)

        img=st.camera_input("Capture Face (30 images)")

        if img is not None:
            file_bytes=np.asarray(bytearray(img.read()),dtype=np.uint8)
            frame=cv2.imdecode(file_bytes,1)

            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces=detector.detectMultiScale(gray,1.3,5)

            for (x,y,w,h) in faces:
                st.session_state.capture_count+=1
                cv2.imwrite(
                    f"{DATA_DIR}/User.{final_id}.{st.session_state.capture_count}.jpg",
                    gray[y:y+h,x:x+w]
                )

            st.write("Captured:",st.session_state.capture_count,"/30")

        if st.session_state.capture_count>=30:
            st.success("Training model...")

            if train_model():
                df=pd.read_csv(CSV_FILE)
                new=[]
                for sub in SUBJECT_LIST:
                    new.append({
                        "RollNo":roll,"Name":name,"Subject":sub,
                        "Section":section,"Held":0,"Attended":0,"LastUpdated":"-"
                    })
                df=pd.concat([df,pd.DataFrame(new)],ignore_index=True)
                df.to_csv(CSV_FILE,index=False)

                st.success("Registration Complete")
                st.session_state.capture_count=0

# ==========================================
# STUDENT VIEW
# ==========================================
def page_student_view():
    if st.button("Back"): navigate_to("student_hub")
    roll=st.text_input("Enter Roll")

    if st.button("Check"):
        df=pd.read_csv(CSV_FILE)
        df['RollNo']=df['RollNo'].astype(str)
        data=df[df['RollNo']==roll]
        if not data.empty:
            st.dataframe(data)
        else:
            st.error("No record")

# ==========================================
# FACULTY LOGIN
# ==========================================
def page_faculty_login():
    if st.button("Back"): navigate_to("home")
    u=st.text_input("ID")
    p=st.text_input("Password",type="password")

    if st.button("Login"):
        if u==p and len(u)==4:
            navigate_to("faculty_dashboard")

# ==========================================
# FACULTY DASHBOARD
# ==========================================
def page_faculty_dashboard():
    if st.button("Logout"): navigate_to("home")

    sub=st.selectbox("Subject",SUBJECT_LIST)
    sec=st.selectbox("Section",SECTION_LIST)

    if st.button("Start Attendance"):
        st.session_state.live_sub=sub
        navigate_to("live_attendance")

# ==========================================
# ✅ LIVE ATTENDANCE (FIXED CAMERA)
# ==========================================
def page_live_attendance():
    if st.button("Back"): navigate_to("faculty_dashboard")

    if not os.path.exists(os.path.join(TRAIN_DIR,"trainer.yml")):
        st.error("Model not trained")
        return

    recognizer=cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(TRAIN_DIR,"trainer.yml"))
    face_cascade=cv2.CascadeClassifier(HAAR_FILE)

    img=st.camera_input("Live Attendance")

    if img is not None:
        file_bytes=np.asarray(bytearray(img.read()),dtype=np.uint8)
        frame=cv2.imdecode(file_bytes,1)

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray,1.2,5)

        for (x,y,w,h) in faces:
            id_internal,conf=recognizer.predict(gray[y:y+h,x:x+w])
            if conf<65:
                roll=get_roll(id_internal)
                st.success(f"Marked: {roll}")
            else:
                st.warning("Unknown")

        st.image(frame,channels="BGR")

# ==========================================
# ROUTER
# ==========================================
if st.session_state.page=="home": page_home()
elif st.session_state.page=="student_hub": page_student_hub()
elif st.session_state.page=="student_register": page_student_register()
elif st.session_state.page=="student_view": page_student_view()
elif st.session_state.page=="faculty_login": page_faculty_login()
elif st.session_state.page=="faculty_dashboard": page_faculty_dashboard()
elif st.session_state.page=="live_attendance": page_live_attendance()

