# Smart Driver Monitoring System (DMS)

An AI-powered Driver Monitoring System designed to improve road safety by detecting unsafe driver behaviors in real time using Computer Vision and Deep Learning.

The system combines **YOLOv8**, **MediaPipe Face Mesh**, and **Faster R-CNN** to monitor driver behavior, detect safety violations, calculate risk levels, and generate real-time alerts through a Streamlit web application.

---

# Project Overview

The Smart Driver Monitoring System analyzes live camera streams, images, and videos to detect dangerous driving behaviors, including:

- Mobile phone usage
- Smoking detection
- Seatbelt violation detection
- Driver drowsiness detection
- Driver risk assessment

The system also stores detection history in MongoDB, sends Telegram notifications for high-risk events, and provides an interactive dashboard for monitoring and analytics.

---
# Project Diagrams

## System Architecture


<img width="2720" height="2400" alt="system_architecture_pipeline" src="https://github.com/user-attachments/assets/38c0ef22-9f8f-4bd0-988e-2526bdf5db7b" />

---

## System Design
<img width="1440" height="600" alt="image" src="https://github.com/user-attachments/assets/b426209e-a480-4076-b81f-df6404d06534" />



---

## Application Workflow

<img width="1440" height="340" alt="image" src="https://github.com/user-attachments/assets/bcb01855-a883-4e45-b9b4-d6eb5237f286" />


---
# Project Objectives

- Detect mobile phone usage while driving.
- Detect seatbelt violations.
- Detect smoking behavior.
- Detect driver drowsiness using EAR, MAR, and head pose estimation.
- Calculate the driver's risk level.
- Generate real-time alerts for dangerous situations.
- Compare the performance of YOLOv8 and Faster R-CNN.
- Store detection history using MongoDB.
- Provide an interactive Streamlit dashboard.

---

# Features

- Real-time driver monitoring
- Image detection
- Video detection
- Phone detection
- Smoking detection
- Seatbelt detection
- Drowsiness detection
- Head pose estimation
- Risk assessment engine
- MongoDB integration
- Telegram alert notifications
- Dashboard and analytics
- Cloud deployment using Streamlit Community Cloud

---

# AI Models

## YOLOv8

- Phone Detection
- Seatbelt Detection
- Smoking Detection

## MediaPipe Face Mesh

- Eye Aspect Ratio (EAR)
- Mouth Aspect Ratio (MAR)
- Head Pose Estimation
- Drowsiness Detection

## Faster R-CNN

- Seatbelt Detection
- Performance comparison with YOLOv8

---

# Technologies Used

- Python
- YOLOv8
- Faster R-CNN
- MediaPipe Face Mesh
- OpenCV
- Streamlit
- MongoDB Atlas
- Telegram Bot API
- PyTorch
- NumPy
- Git & GitHub

---

# Live Demo

Streamlit Application:

https://depi-project-6j4pfappwjcdzjlkbjvwqqg.streamlit.app/



---

# Project Screenshots

Include screenshots of:

- Login Page
- Home Page
- Real-Time Detection
- Image Detection
- Video Detection
- Dashboard
- MongoDB Alerts
- Telegram Notifications

---

# Project Team

## Team Leader

Randa Ashraf Mansour

## Team Members

- Youssef Esam Mohamed
- Ahmed Mohamed Shaker
- Habiba Yahia Lotfy
- Menna Allah Walid Ali
- Hend Mohamed Hassan

---

