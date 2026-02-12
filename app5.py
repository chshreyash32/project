import streamlit as st
import cv2
import pandas as pd
import numpy as np
import os
import json
from datetime import datetime
from PIL import Image

# ==============================
# CONFIG
# ==============================
st.set_page_config(layout="wide")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "dataset")
TRAIN_DIR = os.path.join(BASE_DIR, "trainer")
CSV_FILE = os.path.join(BASE_DIR, "attendance.csv")
MAPPING_FILE = os.path.join(BASE_DIR, "student_map.json")
HAAR_FILE = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(TRAIN_DIR, exist_ok=True)

if not os.path.exists(CSV_FILE):
    pd.DataFrame(columns=["RollNo","Name","Subject","Section","Held","Attended","LastUpdated"]).to_csv(CSV_FILE,index=False)

if not os.path.exists(MAPPING_FILE):
    with open(MAPPING_FILE,'w') as f:
        json.dump({},f)

SUBJECT_LIST=["Machine Learning","Big Data Analytics","Software Engineering"]
SECTION_LIST=["Section A","Section B","Section C"]

if "page" not in st.session_state:
    st.session_state.page="home"

def go(p):
    st.session_state.page=p
    st.rerun()

# ==============================
# UTIL FUNCTIONS
# ==============================

def get_next_id():
    with open(MAPPING_FILE) as f:
        data=json.load(f)
    return max([int(k) for k in data.keys()], default=0)+1

def save_mapping(id,roll):
    with open(MAPPING_FILE) as f:
        data=json.load(f)
    data[str(id)]=roll
    with open(MAPPING_FILE,"w") as f:
        json.dump(data,f)
    return id

def get_roll(id):
    with open(MAPPING_FILE) as f:
        data=json.load(f)
    return data.get(str(id),"Unknown")

# ==============================
# TRAIN MODEL (FIXED)
# ==============================

def train_model():
    if not hasattr(cv2,"face"):
        st.error("Install opencv-contrib-python-headless")
        return False

    recognizer=cv2.face.LBPHFaceRecognizer_create()

    faces=[]
    ids=[]

    for file in os.listdir(DATA_DIR):
        path=os.path.join(DATA_DIR,file)
        img=Image.open(path).convert("L")
        faces.append(np.array(img,"uint8"))
        ids.append(int(file.split(".")[1]))

    if len(faces)==0:
        return False

    recognizer.train(faces,np.array(ids))
    recognizer.write(os.path.join(TRAIN_DIR,"trainer.yml"))
    return True

# ==============================
# HOME
# ==============================

def page_home():
    st.title("Face Recognition Attendance")

    c1,c2=st.columns(2)

    with c1:
        if st.button("Student"):
            go("student")

    with c2:
        if st.button("Faculty"):
            go("faculty")

# ==============================
# STUDENT REGISTER
# ==============================

def page_student():

    if st.button("Back"):
        go("home")

    name=st.text_input("Name")
    roll=st.text_input("Roll")

    detector=cv2.CascadeClassifier(HAAR_FILE)

    if "count" not in st.session_state:
        st.session_state.count=0

    img=st.camera_input("Capture Face")

    if img is not None and name and roll:

        id=save_mapping(get_next_id(),roll)

        bytes_data=np.asarray(bytearray(img.read()),dtype=np.uint8)
        frame=cv2.imdecode(bytes_data,1)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces=detector.detectMultiScale(gray,1.3,5)

        for (x,y,w,h) in faces:
            st.session_state.count+=1
            cv2.imwrite(
                f"{DATA_DIR}/User.{id}.{st.session_state.count}.jpg",
                gray[y:y+h,x:x+w]
            )

        st.write("Captured:",st.session_state.count,"/30")

    if st.session_state.count>=30:
        st.success("Training Model...")
        train_model()
        st.session_state.count=0
        st.success("Registration Complete")

# ==============================
# FACULTY LIVE ATTENDANCE
# ==============================

def page_faculty():

    if st.button("Back"):
        go("home")

    if not os.path.exists(os.path.join(TRAIN_DIR,"trainer.yml")):
        st.error("Train model first")
        return

    recognizer=cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(os.path.join(TRAIN_DIR,"trainer.yml"))

    face_cascade=cv2.CascadeClassifier(HAAR_FILE)

    img=st.camera_input("Take Attendance")

    if img is not None:

        bytes_data=np.asarray(bytearray(img.read()),dtype=np.uint8)
        frame=cv2.imdecode(bytes_data,1)
        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

        faces=face_cascade.detectMultiScale(gray,1.2,5)

        for (x,y,w,h) in faces:
            id,conf=recognizer.predict(gray[y:y+h,x:x+w])

            if conf<65:
                roll=get_roll(id)
                st.success(f"Attendance Marked: {roll}")

                df=pd.read_csv(CSV_FILE)
                now=datetime.now().strftime("%H:%M:%S")

                df.loc[len(df)]=[
                    roll,
                    roll,
                    "ML",
                    "A",
                    1,
                    1,
                    now
                ]
                df.to_csv(CSV_FILE,index=False)

# ==============================
# ROUTER
# ==============================

if st.session_state.page=="home":
    page_home()

elif st.session_state.page=="student":
    page_student()

elif st.session_state.page=="faculty":
    page_faculty()
