# Smart Highway AI Surveillance System

## Overview

Smart Highway AI Surveillance System is an AI-powered traffic monitoring application built using Django, YOLOv8, OpenCV, and ByteTrack.

The system allows users to upload CCTV/highway videos from multiple cameras and automatically:

- Detect vehicles
- Track vehicles
- Count vehicles crossing a virtual exit line
- Generate analytics dashboards
- Monitor vehicle movement
- Identify potentially missing vehicles

---

## Features

### Vehicle Detection
- YOLOv8 object detection
- Detects:
  - Cars
  - Trucks
  - Buses
  - Motorbikes

### Vehicle Tracking
- ByteTrack integration
- Unique vehicle IDs
- Duplicate count prevention

### Exit Line Counting
- Counts vehicles only after crossing a predefined line
- More accurate traffic analytics

### Dashboard Analytics
- Total Vehicles
- Safe Vehicles
- Missing Vehicles
- Cars
- Trucks
- Bikes
- Buses
- Detection Accuracy

### Web Application
- Django Backend
- Bootstrap Frontend
- File Upload Support
- Analytics Dashboard

---

## Technology Stack

### Backend
- Python
- Django

### Computer Vision
- YOLOv8
- OpenCV
- Supervision
- ByteTrack

### Frontend
- HTML
- Bootstrap 5

### Database
- SQLite

### Version Control
- Git
- GitHub

---

## Project Structure

smart_highway_ai/

├── manage.py

├── surveillance/

│ ├── detector.py

│ ├── views.py

│ ├── models.py

│ ├── urls.py

│ └── templates/

│ └── home.html

├── smart_highway_ai/

│ ├── settings.py

│ ├── urls.py

│ └── wsgi.py

└── requirements.txt

---

## Installation

### Clone Repository

```bash
git clone https://github.com/Tejasai-99/smart-highway-ai.git
cd smart-highway-ai
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows:

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Migrations

```bash
python manage.py migrate
```

### Start Server

```bash
python manage.py runserver
```

Open:

```text
http://127.0.0.1:8000
```

---

## Future Enhancements

- Real-time CCTV Streaming
- Number Plate Recognition
- Email Alerts
- SMS Alerts
- Vehicle Re-identification
- Multi-Camera Vehicle Matching
- Cloud Deployment
- AI-Based Anomaly Detection

---

## Author

Tejasai

AI & Machine Learning Developer
