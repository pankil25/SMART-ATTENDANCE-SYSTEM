# SMART ATTENDANCE SYSTEM :

## DEMO VIDEO:

https://github.com/user-attachments/assets/31e23a2d-7fb3-4c17-8ebb-0f6eae48f5cf

## Overview
The Smart Attendance System is a face recognition-based attendance system that uses deep learning and computer vision techniques to recognize faces from images and video streams. This project uses `face_recognition` for face detection, Keras for building the recognition model, and Flask for providing a web interface to monitor and manage attendance.

## Features
- **Real-time Face Recognition**: Detects and recognizes faces from live video feeds.
- **Deep Learning Model**: Uses Convolutional Neural Networks (CNN) for face recognition.
- **Web Interface**: Built using Flask, allowing users to view and manage attendance records.
- **SQLite Database**: Attendance data is stored in an SQLite database.


## Prerequisites

- Python 3.12 or higher
- `pip` (Python package installer)
- Flask==3.0.3
- face_recognition==1.3.0
- Keras==3.5.0
- tensorflow==2.16.0rc0
- opencv-python==4.10.0.84
- dlib==19.24.2
- tqdm==4.66.1
- numpy==1.23.5
- scipy==1.12.0
- scikit-learn==1.4.2
- sqlite3==3.36.0


## Project Structure

├── app.py                  # Flask app for the web interface <br>
├── train.py                # Script to train the model using a dataset <br>
├── detection.py            # Script for detecting faces and marking attendance<br>
├── models/                 # Contains saved models after training<br>
├── dataset/                # Directory for storing face images for training<br>
├── templates/              # HTML templates for the Flask app<br>
├── static/                 # Static files for the web app (CSS, JS)<br>
├── README.md               # Project documentation<br>

## Usage

### 1. Run the Flask App
To start the web interface, execute:

```bash
python app.py
```
Navigate to http://localhost:5000 in your browser to view the web interface.

### 2. Train the Face Recognition Model
To train the model using images in the dataset/ directory, run:
```bash
python train.py
```
Ensure the dataset is structured as follows:

dataset/<br>
├── Person_1/<br>
│   ├── img1.jpg<br>
│   ├── img2.jpg<br>
├── Person_2/<br>
│   ├── img1.jpg<br>
│   ├── img2.jpg<br>



### 3. Start Face Detection
To begin recognizing faces and recording attendance, run:

``` bash
python detection.py
```
This will activate your webcam, detect faces, and display recognized names.

### 4. View Attendance
Attendance is recorded in an SQLite database (attendance.db). You can view the attendance records via the web interface or query the database directly.


