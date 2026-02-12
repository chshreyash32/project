import streamlit as st
import os
import cv2
import numpy as np
from mtcnn import MTCNN
from keras_facenet import FaceNet
import pandas as pd
from datetime import datetime
import time

# --- 1. PAGE SETUP & GRIET BRANDING ---
st.set_page_config(page_title="GRIET AIML Portal", layout="wide")

st.markdown("""
    <style>
    .main-header {text-align: center; color: #1E3A8A; font-weight: bold; margin-bottom: 0px;}
    .sub-header {text-align: center; color: #4B5563; margin-top: 0px;}
    </style>
    """, unsafe_allow_html=True)

st.markdown("<h1 class='main-header'>GOKARAJU RANGARAJU INSTITUTE OF ENGINEERING AND TECHNOLOGY</h1>", unsafe_allow_html=True)
st.markdown("<h3 class='sub-header'>CSE (AIML) - 3rd Year Attendance Portal</h3>", unsafe_allow_html=True)
st.divider()

# --- 2. INITIALIZE MODELS & SESSION STATE ---
@st.cache_resource
def load_models():
    return MTCNN(), FaceNet()

detector, embedder = load_models()

if 'logged_in' not in st.session_state:
    st.session_state.update({'logged_in': False, 'user_type': None, 'user_id': None})

def align_face(img, landmarks, size=160):
    """Aligns face based on eye positions to ensure better feature extraction"""
    try:
        L_eye = landmarks['left_eye']
        R_eye = landmarks['right_eye']
        
        # Calculate angle for rotation
        dY = R_eye[1] - L_eye[1]
        dX = R_eye[0] - L_eye[0]
        angle = np.degrees(np.arctan2(dY, dX))
        
        # Center of eyes
        eye_center = ((L_eye[0] + R_eye[0]) // 2, (L_eye[1] + R_eye[1]) // 2)
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eye_center, angle, 1.0)
        
        # Positioning: eyes at 45% height, centered horizontally
        tX = size * 0.5
        tY = size * 0.45
        M[0, 2] += (tX - eye_center[0])
        M[1, 2] += (tY - eye_center[1])
        
        aligned = cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_CUBIC)
        return aligned
    except:
        return cv2.resize(img, (size, size))

def get_embedding(face_img):
    """Extract and normalize face embedding using standard FaceNet whitening"""
    try:
        # Ensure face is RGB
        if len(face_img.shape) == 2:
            face_img = cv2.cvtColor(face_img, cv2.COLOR_GRAY2RGB)
        
        # Resize to FaceNet input size
        face_img = cv2.resize(face_img, (160, 160))
        face_img = face_img.astype('float32')
        
        # BRIGHTNESS/CONTRAST NORMALIZATION (Standard FaceNet Whitening)
        mean, std = face_img.mean(), face_img.std()
        face_img = (face_img - mean) / max(std, 1e-10) # Centering and scaling
        
        # Get embedding
        samples = np.expand_dims(face_img, axis=0)
        emb = embedder.embeddings(samples)[0]
        
        # L2 normalization (crucial for cosine similarity via dot product)
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        
        return emb
    except Exception as e:
        print(f"Embedding error: {e}")
        return None

def is_good_face(detection, img_shape):
    """Check if detected face has good quality"""
    x, y, w, h = detection['box']
    if w < 60 or h < 60:
        return False
    margin = 10
    if x < margin or y < margin or x+w > img_shape[1]-margin or y+h > img_shape[0]-margin:
        return False
    return True

def recognize_face(test_embedding, known_embeddings, known_names, threshold=0.70):
    """Recognize face using improved consensus voting"""
    if test_embedding is None or len(known_embeddings) == 0:
        return "UNKNOWN", 0.0, "none"
    
    # Calculate similarities with ALL known embeddings
    similarities = np.dot(known_embeddings, test_embedding)
    
    # k-NN Voting (using k=7 for more stability)
    k = min(7, len(known_names))
    top_indices = np.argsort(similarities)[-k:][::-1]
    
    top_scores = similarities[top_indices]
    top_names = known_names[top_indices]
    
    # Consensus check: find the most frequent name in top k
    unique_names, counts = np.unique(top_names, return_counts=True)
    winner_idx = np.argmax(counts)
    winner_name = unique_names[winner_idx]
    winner_votes = counts[winner_idx]
    
    # Average similarity of the winner's samples in the top k
    winner_avg_score = np.mean(top_scores[top_names == winner_name])
    
    # Decision criteria:
    # 1. Broadly require at least 50% of top k to agree
    # 2. And the average score must exceed threshold
    if winner_votes >= (k // 2 + 1) and winner_avg_score >= threshold:
        return str(winner_name).upper(), winner_avg_score, "verified"
    
    # Second chance: if it's a very strong match, allow it even with fewer votes
    if winner_avg_score >= (threshold + 0.1):
        return str(winner_name).upper(), winner_avg_score, "strong_match"
        
    return "UNKNOWN", winner_avg_score, "ambiguous"

# --- 3. LOGIN INTERFACE ---
if not st.session_state['logged_in']:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("👨‍🎓 Student Login")
        s_id = st.text_input("Admission Number", key="s_id")
        s_pw = st.text_input("Password (Admission No)", type="password", key="s_pw")
        if st.button("Login as Student"):
            if s_id == s_pw and s_id != "":
                st.session_state.update({'logged_in': True, 'user_type': 'Student', 'user_id': s_id.strip().upper()})
                st.rerun()
            else: st.error("Invalid Credentials")
    with col2:
        st.subheader("👩‍🏫 Faculty Login")
        f_id = st.text_input("Staff Code", key="f_id")
        f_pw = st.text_input("Password (Staff Code)", type="password", key="f_pw")
        if st.button("Login as Faculty"):
            if f_id == f_pw and f_id != "":
                st.session_state.update({'logged_in': True, 'user_type': 'Faculty', 'user_id': f_id})
                st.rerun()
            else: st.error("Invalid Credentials")

# --- 4. LOGGED IN PORTALS ---
else:
    st.sidebar.title(f"Logged in as: {st.session_state['user_id']}")
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

    subjects = ["ML", "BDA", "DSRP", "JCP", "ML LAB", "BDA LAB", "MPS", "COI"]

    # --- STUDENT PORTAL ---
    if st.session_state['user_type'] == 'Student':
        tab1, tab2 = st.tabs(["📊 Attendance Records", "📝 Register Face"])
        
        with tab1:
            u_section = st.selectbox("Select Your Section", ["A", "B", "C"])
            summary = []
            section_folder = f"attendance/Section_{u_section}"
            
            for sub in subjects:
                attended, total = 0, 0
                if os.path.exists(section_folder):
                    sub_files = [f for f in os.listdir(section_folder) if f.startswith(sub)]
                    total = len(sub_files)
                    for file in sub_files:
                        df = pd.read_csv(f"{section_folder}/{file}")
                        if st.session_state['user_id'] in df['Admission_No'].astype(str).values:
                            attended += 1
                summary.append({"Subject": sub, "Attended": attended, "Total Classes": total, 
                                "Percentage": f"{(attended/total)*100:.1f}%" if total > 0 else "0%"})
            st.table(pd.DataFrame(summary))

        with tab2:
            st.subheader("📸 Register Face for AI Attendance")
            
            st.markdown("""
            ### 📋 Instructions:
            1. **Click** "Start Camera & Capture" button below
            2. **Wait** for camera to open and show your live preview
            3. **Position** your face in center of the screen
            4. **Stay still** - system will automatically capture 50 high-quality images
            5. **Wait** until all 50 images are captured
            6. **Click** "Complete Registration" to save your face data
            """)
            
            st.warning("⚠️ **Important:** Ensure good lighting and look directly at camera!")
            
            # Initialize session state for capture control
            if 'capturing' not in st.session_state:
                st.session_state.capturing = False
            if 'capture_count' not in st.session_state:
                st.session_state.capture_count = 0
            
            status_container = st.empty()
            image_placeholder = st.empty()
            
            col_a, col_b = st.columns(2)
            
            with col_a:
                if st.button("🎥 Start Camera & Capture", type="primary", disabled=st.session_state.capturing):
                    st.session_state.capturing = True
                    st.session_state.capture_count = 0
                    st.rerun()
            
            with col_b:
                if st.button("⏹️ Stop Capture", type="secondary", disabled=not st.session_state.capturing):
                    st.session_state.capturing = False
                    st.rerun()
            
            # CAMERA CAPTURE LOGIC
            if st.session_state.capturing:
                user_id = st.session_state['user_id']
                path = f"dataset/{user_id}"
                os.makedirs(path, exist_ok=True)
                
                # Clear old images
                for old_file in os.listdir(path):
                    try:
                        os.remove(os.path.join(path, old_file))
                    except:
                        pass
                
                status_container.info("📷 Opening camera... Please wait...")
                
                # Open camera
                cap = cv2.VideoCapture(0)
                
                # Give camera time to initialize
                time.sleep(1)
                
                if not cap.isOpened():
                    status_container.error("❌ Failed to open camera! Please check camera permissions.")
                    st.session_state.capturing = False
                    st.stop()
                
                status_container.success("✅ Camera opened! Position your face in the frame...")
                
                captured = 0
                total_to_capture = 50
                frame_skip = 0
                warmup_frames = 30  # Skip first 30 frames to let camera adjust
                
                while st.session_state.capturing and captured < total_to_capture:
                    ret, frame = cap.read()
                    
                    if not ret:
                        status_container.error("❌ Failed to read from camera!")
                        break
                    
                    # Skip warmup frames
                    if warmup_frames > 0:
                        warmup_frames -= 1
                        # Show live preview during warmup
                        preview = frame.copy()
                        cv2.putText(preview, "Initializing... Please wait", (50, 50), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                        cv2.putText(preview, f"Warmup: {30-warmup_frames}/30", (50, 100), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
                        image_placeholder.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                        time.sleep(0.05)
                        continue
                    
                    frame_skip += 1
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces
                    detections = detector.detect_faces(rgb)
                    
                    # Draw on frame for preview
                    display_frame = frame.copy()
                    
                    # Add capture status
                    cv2.putText(display_frame, f"Captured: {captured}/{total_to_capture}", (10, 40), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)
                    
                    # Only capture every 4th frame for variety
                    should_capture = (frame_skip % 4 == 0)
                    
                    if detections and should_capture:
                        for detection in detections:
                            confidence = detection['confidence']
                            
                            if confidence > 0.95 and is_good_face(detection, frame.shape):
                                x, y, w, h = detection['box']
                                
                                # Add padding
                                padding = 30
                                x_pad = max(0, x - padding)
                                y_pad = max(0, y - padding)
                                w_pad = min(frame.shape[1] - x_pad, w + 2*padding)
                                h_pad = min(frame.shape[0] - y_pad, h + 2*padding)
                                
                                # Extract and save face
                                face = frame[y_pad:y_pad+h_pad, x_pad:x_pad+w_pad]
                                
                                if face.size > 0:
                                    # Save the captured face
                                    cv2.imwrite(f"{path}/{captured}.jpg", face)
                                    captured += 1
                                    
                                    # Draw GREEN rectangle for successful capture
                                    cv2.rectangle(display_frame, (x_pad, y_pad), 
                                                (x_pad+w_pad, y_pad+h_pad), (0, 255, 0), 4)
                                    cv2.putText(display_frame, f"CAPTURED! #{captured}", (x, y-40), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                                    cv2.putText(display_frame, f"Conf: {confidence:.2f}", (x, y-10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    
                                    # Update status
                                    status_container.success(f"✅ Progress: {captured}/{total_to_capture} images captured!")
                                    
                                    break  # Only capture one face per frame
                            else:
                                # Draw YELLOW rectangle for detected but not captured
                                x, y, w, h = detection['box']
                                cv2.rectangle(display_frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                                if confidence <= 0.95:
                                    cv2.putText(display_frame, "Low confidence", (x, y-10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                                else:
                                    cv2.putText(display_frame, "Too close to edge", (x, y-10), 
                                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    else:
                        # No face detected
                        if not detections:
                            cv2.putText(display_frame, "No face detected - Please look at camera", (50, 100), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                    
                    # Show live preview
                    image_placeholder.image(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB), 
                                          channels="RGB", use_container_width=True)
                    
                    # Small delay
                    time.sleep(0.05)
                
                # Release camera
                cap.release()
                st.session_state.capturing = False
                st.session_state.capture_count = captured
                
                if captured >= total_to_capture:
                    image_placeholder.empty()
                    status_container.success(f"🎉 Successfully captured {captured} images! Click 'Complete Registration' below.")
                elif captured > 0:
                    image_placeholder.empty()
                    status_container.warning(f"⚠️ Only captured {captured} images. Need at least 30. Try again!")
                else:
                    image_placeholder.empty()
                    status_container.error("❌ No images captured. Please try again with better lighting.")
            
            st.divider()
            
            # Complete Registration Button
            if st.button("✅ Complete Registration", type="primary"):
                user_id = st.session_state['user_id']
                path = f"dataset/{user_id}"
                
                if not os.path.exists(path):
                    st.error("❌ No images found! Please capture images first.")
                else:
                    num_images = len(os.listdir(path))
                    
                    if num_images < 30:
                        st.error(f"❌ Only {num_images} images found. Please capture at least 30 images!")
                    else:
                        with st.spinner(f"🧠 Processing {num_images} images and learning your face..."):
                            X, Y = [], []
                            successful = 0
                            failed = 0
                            
                            # Process all captured images
                            for img_file in sorted(os.listdir(path)):
                                img_path = os.path.join(path, img_file)
                                img = cv2.imread(img_path)
                                
                                if img is not None:
                                    try:
                                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                                        # Detect to get landmarks for better alignment during registration
                                        res = detector.detect_faces(rgb)
                                        if res:
                                            aligned = align_face(rgb, res[0]['keypoints'])
                                            embedding = get_embedding(aligned)
                                        else:
                                            embedding = get_embedding(rgb)
                                        
                                        if embedding is not None:
                                            X.append(embedding)
                                            Y.append(user_id)
                                            successful += 1
                                        else:
                                            failed += 1
                                    except Exception as e:
                                        failed += 1
                                        continue
                            
                            if len(X) < 20:
                                st.error(f"❌ Not enough valid embeddings ({len(X)}). Please recapture with better lighting!")
                            else:
                                # Load existing embeddings and remove old entries for this student
                                if os.path.exists("embeddings.npz"):
                                    d = np.load("embeddings.npz", allow_pickle=True)
                                    mask = np.array([name.upper() for name in d['names']]) != user_id.upper()
                                    old_embeddings = d['embeddings'][mask]
                                    old_names = np.array([name.upper() for name in d['names']])[mask]
                                    X = np.concatenate((old_embeddings, X))
                                    Y = np.concatenate((old_names, Y))
                                    st.info(f"Updated existing registration for {user_id}")
                                
                                # Save embeddings
                                np.savez_compressed("embeddings.npz", 
                                                  embeddings=np.array(X), 
                                                  names=np.array(Y))
                                
                                st.balloons()
                                st.success(f"""
                                ✅ **Registration Successful!**
                                
                                - Valid face samples: **{successful}**
                                - Failed samples: **{failed}**
                                - Total samples for {user_id}: **{len([y for y in Y if y == user_id])}**
                                
                                You can now attend classes using face recognition!
                                """)

    # --- FACULTY PORTAL ---
    else:
        st.subheader("👩‍🏫 Faculty Control Panel")
        
        # Show registered students
        if os.path.exists("embeddings.npz"):
            data = np.load("embeddings.npz", allow_pickle=True)
            unique_students = len(np.unique(data['names']))
            total_samples = len(data['names'])
            st.info(f"📊 **Database:** {unique_students} students registered with {total_samples} face samples")
        else:
            st.warning("⚠️ No students registered yet!")
        
        c1, c2, c3 = st.columns(3)
        with c1: 
            sec = st.selectbox("Section", ["A", "B", "C"])
        with c2: 
            sub = st.selectbox("Subject", subjects)
        with c3:
            threshold = st.slider("Recognition Threshold", 0.50, 0.85, 0.65, 0.05)
            st.caption(f"Lower = Lenient | Higher = Strict")
        
        st.info("""
        📌 **Best Practices:**
        - Ensure good, consistent lighting
        - Students should be 2-4 feet from camera
        - Students look directly at camera
        - Process one student at a time for accuracy
        """)
        
        # Scanner controls
        if "scanning" not in st.session_state:
            st.session_state.scanning = False
        
        col_btn1, col_btn2 = st.columns(2)
        with col_btn1:
            if st.button("🚀 Start Live Attendance", type="primary"):
                st.session_state.scanning = True
                st.rerun()
        with col_btn2:
            if st.button("🛑 Stop Scanning", type="secondary"):
                st.session_state.scanning = False
                st.rerun()
        
        if st.session_state.scanning:
            if not os.path.exists("embeddings.npz"):
                st.error("❌ No registered students! Students must register first.")
                st.session_state.scanning = False
            else:
                # Load embeddings
                data = np.load("embeddings.npz", allow_pickle=True)
                k_emb = data['embeddings']
                k_names = data['names']
                
                # Normalize embeddings
                norms = np.linalg.norm(k_emb, axis=1, keepdims=True)
                k_emb = k_emb / np.maximum(norms, 1e-10)
                
                # Setup paths
                path = f"attendance/Section_{sec}"
                os.makedirs(path, exist_ok=True)
                csv_p = f"{path}/{sub}_{datetime.now().strftime('%Y-%m-%d')}.csv"
                
                # Initialize CSV
                if not os.path.exists(csv_p): 
                    pd.DataFrame(columns=["Admission_No", "Time", "Confidence"]).to_csv(csv_p, index=False)
                
                # UI placeholders
                FRAME = st.empty()
                INFO = st.empty()
                DEBUG = st.empty()
                
                # Open camera
                cap = cv2.VideoCapture(0)
                time.sleep(1)  # Camera warmup
                
                if not cap.isOpened():
                    st.error("❌ Cannot open camera!")
                    st.session_state.scanning = False
                    st.stop()
                
                recognized_students = set()
                frame_count = 0
                
                INFO.success("✅ Camera opened! Scanning for faces...")
                
                while st.session_state.scanning:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Camera read error!")
                        break
                    
                    frame_count += 1
                    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Detect faces
                    detections = detector.detect_faces(rgb)
                    
                    debug_info = []
                    
                    for detection in detections:
                        if detection['confidence'] > 0.95:
                            x, y, w, h = detection['box']
                            
                            if not is_good_face(detection, frame.shape):
                                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 0), 2)
                                cv2.putText(frame, "Low Quality", (x, y-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                                continue
                            
                            # Extract face and align
                            try:
                                # Align using detection landmarks
                                aligned_face = align_face(rgb, detection['keypoints'])
                                name, confidence, method = recognize_face(
                                    get_embedding(aligned_face), k_emb, k_names, threshold=threshold
                                )
                                
                                debug_info.append(f"{name}: {confidence:.2f}")
                                
                                # Attendance logic
                                if name != "UNKNOWN":
                                    color = (0, 255, 0)
                                    if name not in recognized_students:
                                        df = pd.read_csv(csv_p)
                                        if name not in df["Admission_No"].astype(str).values:
                                            now = datetime.now().strftime("%H:%M:%S")
                                            new_data = pd.DataFrame({
                                                "Admission_No": [name],
                                                "Time": [now],
                                                "Confidence": [f"{confidence:.3f}"]
                                            })
                                            pd.concat([df, new_data]).to_csv(csv_p, index=False)
                                            st.toast(f"✅ {name} - {confidence:.2f}", icon="✅")
                                            recognized_students.add(name)
                                else:
                                    color = (255, 0, 0)
                                
                                # Visual feedback
                                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                                cv2.putText(frame, f"{name} ({confidence:.2f})", (x, y-10), 
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                            except Exception as e:
                                debug_info.append(f"Err: {str(e)[:15]}")
                    
                    # Add frame counter
                    cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    
                    # Display
                    FRAME.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
                    
                    # Update info
                    df = pd.read_csv(csv_p)
                    INFO.success(f"📊 **Present:** {len(df)} students | **Detected:** {len(detections)} faces")
                    
                    if debug_info:
                        DEBUG.info("🔍 " + " | ".join(debug_info[:3]))
                    
                    time.sleep(0.03)
                
                cap.release()
                
                # Show results
                final_df = pd.read_csv(csv_p)
                st.success(f"✅ **Attendance Complete:** {len(final_df)} students marked present")
                
                if len(final_df) > 0:
                    st.dataframe(final_df, use_container_width=True)