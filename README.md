🎓 Intelligent Face Recognition Attendance System
A real-time AI-powered attendance management system that automatically records student attendance using face recognition technology.
The system provides separate portals for students and faculty, enabling seamless attendance tracking, reporting, and management.
This project was developed as part of B.Tech CSE (AIML) academic learning to demonstrate practical implementation of Computer Vision, Machine Learning, and Web Application Development.

🚀 Project Overview
Traditional attendance systems are time-consuming, error-prone, and easy to manipulate. This system provides an automated, secure, and efficient solution using:
Face detection and recognition
Live attendance marking
Student registration via face capture
Faculty dashboard for session control
Real-time attendance updates
Attendance analytics and reporting
The system captures facial data, trains a recognition model, and marks attendance automatically during live sessions.
✨ Features

👨‍🎓 Student Portal
Student registration with face capture (30 images)
Secure student identity mapping
View personal attendance records
Subject-wise attendance tracking
Attendance percentage calculation

👨‍🏫 Faculty Portal
Faculty login authentication
Create live attendance sessions
Select subject and section
Real-time attendance marking
Multiple period support
Live activity logs

🤖 AI & Computer Vision
Face detection using Haar Cascade
Face recognition using LBPH algorithm
Model training from captured dataset
Unknown face detection
Cooldown system to avoid duplicate marking

📊 Attendance Management
CSV-based attendance storage
Student database management
Subject-wise attendance tracking
Automatic attendance updates
Timestamp logging

🎨 User Interface
Streamlit web interface
Premium dark UI design
Real-time camera feed
Interactive dashboards

🛠️ Technology Stack
Programming Language
Python
Framework
Streamlit (Web Interface)
Libraries
OpenCV — Face detection and recognition
NumPy — Numerical computation
Pandas — Data storage and processing
PIL — Image handling

Concepts Used:
Computer Vision
Machine Learning
Face Recognition
Image Processing
Real-Time Systems

🏗️ System Architecture
Student Registration → Image Capture → Dataset Creation
                          ↓
                    Model Training
                          ↓
Live Camera Feed → Face Recognition → Attendance Update
                          ↓
                    Attendance Database

📁 Project Structure
project/
│
├── dataset/                # Stored face images
├── trainer/                # Trained model files
├── attendance.csv          # Attendance database
├── student_map.json        # Student ID mapping
│
├── main.py                 # Main application
├── collect_data.py         # Face data collection
├── train_model.py          # Model training
├── attendance_system.py    # Attendance logic
├── dashboard.py            # UI dashboard
├── attendance_summary.py   # Attendance reports
│
├── requirements.txt        # Dependencies
└── README.md


🔄 Application Workflow (System Flow) [app5.py] STREAMLIT APP
This application follows a structured multi-step workflow implemented using a state-based navigation system in Streamlit.
🏠 1. Home Page
Entry point of the system.
Users choose between:
Student Portal
Faculty Portal
Navigation handled using Streamlit session state routing.

👨‍🎓 2. Student Portal Flow
➤ Student Dashboard
Access attendance records.
Register new face for first-time users.
➤ Student Registration Process
Enter:
Full Name
Roll Number
Section
System activates webcam.
Captures 30 facial images using OpenCV.
Images stored in dataset folder.
System assigns internal student ID.
Face recognition model is trained automatically.
Student record created in attendance database for all subjects.

Output:
Trained face recognition model
Student mapping stored
Attendance records initialized
➤ Attendance Viewing
Student enters roll number.
System fetches records from CSV database.

Displays:
Subject-wise attendance
Total classes held
Classes attended
Attendance percentage
Last updated time

👨‍🏫 3. Faculty Portal Flow
➤ Faculty Login
Authentication using 4-digit faculty ID.
Redirects to faculty dashboard on successful login.
➤ Attendance Session Setup
Faculty configures:
Subject selection
Section selection
Number of periods
Start live attendance session

🎥 4. Live Attendance System (Core AI Module)
When faculty starts attendance:
System loads trained face recognition model.
Webcam starts real-time video capture.
Face detection performed using Haar Cascade.
Face recognition using LBPH algorithm.
If match confidence is acceptable:
Student identity retrieved
Attendance marked automatically
Unknown faces are ignored.
Cooldown period prevents duplicate marking.
Real-time features:
Live camera feed
Attendance logs
Student recognition status
Auto update of attendance records

📊 5. Attendance Database Update
When a student is recognized:
Classes held updated
Classes attended updated
Timestamp recorded
Data stored in CSV database
Logs shown on dashboard

🗂️ 6. Data Storage & Model Management
The system maintains:
Dataset Folder → Captured face images
Trainer Folder → Trained model file
Student Mapping → Internal ID ↔ Roll number
Attendance Database → CSV records

🧠 Internal System Logic
The application is built using:
Session-based page routing
Real-time camera processing loop
Model training pipeline
Face-to-ID mapping system
Cooldown-based attendance marking
CSV-based persistent storage
This architecture ensures:

✅ Real-time performance
✅ Scalable student management
✅ Accurate attendance tracking
✅ Secure identity mapping

⚙️ Installation & Setup
1️⃣ Clone Repository
git clone https://github.com/yourusername/project.git
cd project

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run Application
streamlit run main.py

📌 How It Works
Student Registration
Enter name and roll number
Capture 30 face images
Train recognition model automatically
Faculty Attendance
Login as faculty
Select subject and section
Start live attendance session
System detects and marks students automatically
Attendance Storage
Stored in CSV database
Updated in real-time
Subject-wise tracking

🔐 Authentication
Faculty login requires:
4-digit faculty ID
Password same as ID (demo implementation)

📊 Future Improvements
Deep learning based face recognition (FaceNet / MTCNN)
Cloud database integration
Mobile deployment
Multi-user authentication
Anti-spoofing detection
Performance optimization
Attendance analytics dashboard

🎯 Learning Outcomes
Through this project, I gained practical experience in:
Building real-time computer vision systems
Machine learning model training
Web app development with Streamlit
Image processing pipelines
Data management using Pandas
End-to-end system design

👨‍💻 Author
Shreyash Chintalwar
B.Tech CSE (Artificial Intelligence & Machine Learning)
Computer Vision Enthusiast
AI & ML Learner
Software Developer
