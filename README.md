# HAWKEYE — Deep Learning Threat Detection

> A real-time computer vision system for detecting suspicious human activities in video streams, extracting evidence, and delivering responsive web-based monitoring.

## Overview

HAWKEYE is a real-time deep learning and computer vision platform designed to analyze video streams and identify potentially suspicious human activities.

The system combines object detection, video understanding, computer vision processing, and a multithreaded processing pipeline to transform raw video into actionable detection results.

The primary goal is to build a responsive surveillance intelligence pipeline capable of:

- Processing video streams in real time
- Detecting humans and relevant objects
- Identifying suspicious activity patterns
- Running deep-learning inference efficiently
- Extracting evidence from detected events
- Streaming processed results through a web interface
- Maintaining responsive performance through concurrent processing

---

## Key Features

### Real-Time Video Processing

Processes video streams continuously using OpenCV and a concurrent processing pipeline designed to reduce processing bottlenecks.

### Deep Learning-Based Detection

Uses YOLOv8-based object detection to identify relevant entities within video frames.

### Suspicious Activity Detection

Analyzes detected activity to identify potentially suspicious human behaviors and events.

### Evidence Extraction

Captures relevant evidence associated with detected events to support later inspection and analysis.

### Web-Based Monitoring

Provides processed video output through a Flask-based web application for responsive monitoring.

### Multithreaded Processing

Separates computationally intensive video-processing operations from streaming and application-level operations to improve responsiveness.

### Computer Vision Pipeline

Combines multiple computer vision and deep learning components into a continuous processing pipeline:

```
Video Input
     │
     ▼
Frame Acquisition
     │
     ▼
Preprocessing
     │
     ▼
Object Detection
     │
     ▼
Activity Analysis
     │
     ▼
Suspicious Activity Detection
     │
     ├──────────────► Evidence Extraction
     │
     ▼
Processed Video Stream
     │
     ▼
Web Interface
```

### Technology Stack
Deep Learning
*YOLOv8
*PyTorch
*PyTorchVideo

Computer Vision
*OpenCV
*Video frame processing
*Object detection
*Activity analysis

Backend
*Python
*Flask

Systems & Performance
*Multithreading
*Concurrent video processing
*Real-time frame handling
*Performance-oriented inference pipeline


### System Architecture

HAWKEYE is organized around a real-time video processing pipeline.
```
                    ┌─────────────────────┐
                    │     Video Source    │
                    │ Camera / Video File │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │  Frame Acquisition  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    Preprocessing    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   YOLOv8 Detection  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Activity Analysis   │
                    └──────────┬──────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
          ┌─────────────────┐   ┌─────────────────┐
          │ Suspicious      │   │ Normal Activity │
          │ Activity        │   │                 │
          └────────┬────────┘   └─────────────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Evidence        │
          │ Extraction      │
          └────────┬────────┘
                   │
                   ▼
          ┌─────────────────┐
          │ Flask Streaming │
          │ Interface       │
          └─────────────────┘
```


