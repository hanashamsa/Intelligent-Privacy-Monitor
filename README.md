# PrivacyGuard AI
### Intelligent Privacy & Shoulder Surfing Detection System using Computer Vision, Face Recognition, Gaze Estimation and Behavioral Analysis

![Python](https://img.shields.io/badge/Python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![YOLO](https://img.shields.io/badge/YOLO-v11-green)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-blue)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

# Overview

PrivacyGuard AI is a real-time intelligent privacy protection system designed to detect potential **shoulder surfing attacks** around a computer workstation.

Unlike traditional surveillance systems that only detect people, PrivacyGuard AI understands:

- Who the person is
- Where they are located
- What they are looking at
- How long they have been there
- Whether they pose a privacy threat

The system fuses multiple computer vision models and behavioral analysis modules to estimate the probability of a shoulder surfing attack before triggering an incident.

This project was designed as a production-quality AI surveillance framework suitable for research, portfolio demonstrations, and future deployment.

---

# Problem Statement

Sensitive information displayed on laptops and workstations can be unintentionally exposed to nearby people.

Traditional CCTV systems merely record events without understanding user behavior.

PrivacyGuard AI actively analyzes the environment to determine whether someone is attempting to observe the user's screen.

The system automatically:

- Detects people
- Identifies the owner
- Estimates head pose
- Estimates eye gaze
- Understands spatial relationships
- Calculates threat levels
- Captures forensic evidence
- Generates explainable alerts

---

# System Architecture

```
Camera
   │
   ▼
YOLO Scene Detection
   │
   ▼
ByteTrack Multi Object Tracking
   │
   ▼
Face Detection
   │
   ▼
ArcFace Recognition
   │
   ▼
Identity Manager
   │
   ▼
Head Pose Estimation
   │
   ▼
L2CS-Net Gaze Estimation
   │
   ▼
Screen Awareness
   │
   ▼
Spatial Analysis
   │
   ▼
Behavior Analysis
   │
   ▼
Threat Engine
   │
   ▼
Incident Recorder
   │
   ▼
Evidence Package
   │
   ▼
Dashboard
```
---
```
# Using Guide 

1- Add Images of the owner to a newly made folder named face_db\owner ( This must hold aorund 250-300 pictures of the owner as this is used for training. possibly the pictures are capured from a video and converted to .jpg at 30fps )

2- Download and add  User.\insight\models\biffalo_l as this is a must step for buiding the database of owner 

3- Make a folder embeddings\owner_embedding.npy as the trained dataset is stored here 

4- Run face\build_dataset.py , this make the training and loads the data to owner_embedding.npy ( choose the right path for certain os ).
#results will be visible as
==================================================
Images Processed : 300
Valid Faces      : 300
Database Created Successfully
==================================================

5- Download and load face_landmarker.task at pose\head_pose.py as we use 'mediapipe'

6- Main parameters can be found and altered under surveilance\threat_engine.py
```
----
---

# Features

## Real-Time Person Detection

Uses Ultralytics YOLO to detect

- Person
- Laptop
- Monitor
- Mobile Phone

Real-time GPU accelerated inference.

---

## Multi Object Tracking

Uses ByteTrack to assign persistent IDs.

Example

```
Person #4

↓

Tracker ID remains 4

↓

Identity persists
```

---

## Face Recognition

Powered by InsightFace ArcFace.

The system creates an embedding database of the owner.

During execution every tracked face is compared against the owner embedding.

Output

```
Owner

Unknown
```

---

## Identity Cache

Face recognition is expensive.

Instead of recognizing every frame

```
30 FPS

↓

900 recognitions/minute
```

PrivacyGuard caches identities.

Recognition occurs only every few frames.

Benefits

- Higher FPS
- Lower GPU usage
- Stable identities

---

## Head Pose Estimation

MediaPipe FaceMesh estimates

- Looking Left
- Looking Right
- Looking Center

Future versions will provide continuous

- Yaw
- Pitch
- Roll

---

## Gaze Estimation

Powered by L2CS-Net.

Outputs

```
Yaw

Pitch

Eye Direction

Looking At Screen
```

instead of relying only on head orientation.

---

## Scene Understanding

The detector understands

- Person
- Laptop
- Monitor
- Mobile Phone

instead of only people.

This allows future scene reasoning.

---

## Screen Awareness

The system identifies the active screen in the scene.

Current implementation estimates whether gaze is directed toward the screen.

Future versions will perform true gaze-ray intersection.

---

## Spatial Analysis

Computes

- Owner location
- Stranger location
- Distance to owner
- Relative position
- Danger zone

Example

```
Unknown

↓

Behind Owner

↓

Inside Danger Zone
```

---

## Behavioral Analysis

The system does not classify individual frames.

Instead it analyses behavior over time.

Tracks

- Time spent near owner
- Presence inside danger zone
- Continuous suspicious behavior

This dramatically reduces false alarms.

---

## Threat Engine

Multiple AI modules are fused together.

Current inputs

- Identity
- Head Pose
- Eye Gaze
- Distance
- Danger Zone
- Duration
- Screen Awareness

Outputs

```
SAFE

LOW

MEDIUM

HIGH
```

with a confidence score.

---

## Incident Recorder

When a threat is detected

PrivacyGuard automatically

- Saves screenshot
- Saves buffered video
- Creates incident package

---

## Circular Video Buffer

Maintains previous camera frames in memory.

Incident videos include

```
10 seconds before

↓

Threat

↓

Recording
```

Future versions will also include post-event recording.

---

## Explainable AI

Every threat includes reasoning.

Example

```
Threat Score : 0.91

Reason

✓ Unknown Person

✓ Looking At Screen

✓ Inside Danger Zone

✓ Stayed 4.2 seconds

✓ Very Close
```

---

## Live Dashboard

Real-time dashboard displaying

- FPS
- Owner presence
- Unknown people
- Threat score
- Threat level
- Recorder state
- Incident count

---

## Local Alert System

Displays a real-time privacy warning when

```
HIGH THREAT
```

is detected.

No cloud services required.

Entirely local.

---

# Technologies Used

## Deep Learning

- PyTorch
- TorchVision

## Object Detection

- Ultralytics YOLO

## Tracking

- ByteTrack
- Supervision

## Face Recognition

- InsightFace
- ArcFace

## Head Pose

- MediaPipe FaceMesh

## Eye Gaze

- L2CS-Net

## Computer Vision

- OpenCV

## Numerical Processing

- NumPy

---

# Project Structure

```
PrivacyGuardAI/

│
├── app.py
├── config.py
├── requirements.txt
│
├── detector/
│
├── tracker/
│
├── face/
│
├── identity/
│
├── gaze/
│
├── pose/
│
├── surveillance/
│
├── scene/
│
├── recording/
│
├── alerts/
│
├── explainability/
│
├── ui/
│
├── outputs/
│
│     ├── incidents/
│     ├── screenshots/
│     └── videos/
│
├── models/
│
└── embeddings/
```

---

# Current Pipeline

```
Camera

↓

YOLO

↓

ByteTrack

↓

Face Recognition

↓

Identity Cache

↓

Head Pose

↓

Gaze Estimation

↓

Scene Detection

↓

Screen Awareness

↓

Spatial Analysis

↓

Behavior Analysis

↓

Threat Engine

↓

Incident Recorder

↓

Alert System

↓

Dashboard
```

---

# Current Threat Logic

The threat score is currently computed using weighted feature fusion.

Current factors include

| Feature | Description |
|----------|-------------|
| Identity | Owner / Unknown |
| Danger Zone | Person near protected user |
| Head Pose | Orientation of face |
| Eye Gaze | Looking at screen |
| Duration | Time spent in danger zone |
| Distance | Distance to owner |

Future versions will replace this rule-based engine with a machine learning classifier.

---

# Performance Optimizations

Current optimizations

- Identity caching
- GPU inference
- Face crop recognition
- Rolling video buffer
- Cached tracker identities

Future optimizations

- TensorRT
- ONNX Runtime
- Mixed precision inference
- Multi-threaded pipeline

---

# Future Work

## Machine Learning Threat Engine

Replace manually selected weights with

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

using real incident data.

---

## Screen Geometry

Instead of treating the laptop bounding box as the screen

Estimate the actual display region.

---

## Phone Detection

Increase threat level when

```
Unknown

+

Phone

+

Looking At Screen
```

---

## Multi Camera Support

Synchronize

- Front camera
- CCTV
- External USB camera

---

## Incident Dashboard

Interactive dashboard

- Incident history
- Video playback
- Search
- Statistics
- Heatmaps

---

## Docker Deployment

Containerized deployment

```
docker compose up
```

---

# Example Output

```
Owner

Threat : SAFE

Score : 0.02
```

```
Unknown

Threat : HIGH

Score : 0.93

Reason

✓ Looking at Screen

✓ Inside Danger Zone

✓ Unknown Identity

✓ Stayed 5.4 seconds
```

---

# Current Limitations

Current implementation is rule-based.

Threat scores are manually weighted.

The system currently assumes

- Laptop bounding box approximates the screen
- Single owner
- Single workstation

Future work addresses these limitations through

- Learned threat classification
- True gaze-ray intersection
- Multi-user support
- Adaptive calibration

---

# Research Contribution

PrivacyGuard AI combines multiple AI subsystems into a unified privacy monitoring framework.

Integrated modules include

- Object Detection
- Multi Object Tracking
- Face Recognition
- Head Pose Estimation
- Eye Gaze Estimation
- Spatial Reasoning
- Behavioral Analysis
- Threat Assessment
- Explainable AI
- Incident Recording

Rather than relying on a single neural network, PrivacyGuard AI performs **multi-modal evidence fusion** to estimate shoulder surfing risk in real time.

---

# License

MIT License

---

# Author

**Muhammed Hanas V.H.**

M.Sc. Artificial Intelligence

Johannes Kepler University Linz

Austria
