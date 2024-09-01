# crowd-monitoring-system
# Hand and Crowd Detection using OpenCV, MediaPipe, and YOLO

This project implements hand detection using MediaPipe, crowd detection using YOLO, and background change detection using ORB (Oriented FAST and Rotated BRIEF) in real-time from multiple video sources. The system also includes functionality to monitor specific hand gestures and count people in the video.

## Table of Contents
- [Introduction](#introduction)
- [Libraries Used](#libraries-used)
- [Working](#working)
- [Screenshots](#screenshots)
- [Conclusion](#conclusion)
- [How to Use](#how-to-use)

## Introduction
This project is a multi-tasking video processing application that detects hand gestures, counts people in video frames, and monitors background changes. It's developed using OpenCV, MediaPipe, YOLO, and other essential libraries in Python. The system handles multiple video inputs simultaneously and provides real-time analysis and feedback.

## Libraries Used
The following libraries are used in this project:
- OpenCV
- MediaPipe
- Tkinter
- PIL (Pillow)
- NumPy
- Ultralytics (YOLO)

To install these dependencies, you can use the following pip command:
```bash
pip install opencv-python mediapipe tkinter pillow numpy ultralytics
```

## Working
Hand Detection: The project uses MediaPipe Hands to detect hands and identify specific gestures. If a gesture resembling "OK" is detected, it triggers a caution warning.
Crowd Counting: The YOLO model is employed to detect people in video frames. The system counts people and updates the count dynamically.
Background Change Detection: Using ORB, the project detects significant changes in the background and raises a warning if the changes exceed a certain threshold.
Multiple Video Inputs: The system processes up to four video inputs simultaneously, including pre-recorded videos and live webcam feed.


## Screenshots
Sample output showing hand detection, crowd counting, and background change detection.

## Conclusion
This project demonstrates the power of real-time video analysis using OpenCV, MediaPipe, and YOLO. It can be extended for various surveillance and monitoring applications where hand gestures, crowd estimation, and background changes are critical.

## How to Use
Clone the repository:

```bash
git clone https://github.com/AAC-Open-Source-Pool/Safer-Tourism
```
Navigate to the project directory:
```bash
cd repository-name
```
Install the required libraries:
```bash
pip install -r requirements.txt
```

Run the project:
```bash
python your_script.py
```
