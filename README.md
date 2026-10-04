# Intelligent Face Tracker with Auto-Registration & Visitor Counting

An AI-driven unique visitor counter that processes video streams to detect, track, and recognize faces in real-time. The system automatically registers new faces upon first detection, recognizing them in subsequent frames, and tracks them continuously until they exit the frame.

## Setup Instructions
1. Install Python 3.9+
2. Install dependencies: `pip install -r requirements.txt`
3. Configure settings in `config.json`
4. Download the ONNX model files to `models/` directory (Detailed instructions later)
5. Run the application: `python main.py`

## Architecture & Assumptions
- **Architecture**: Core Pipeline utilizes YOLO for detection, ByteTrack logic for object tracking, and ArcFace ONNX model for embeddings.
- **Assumptions**: 
  - RTSP stream or static MP4 videos have sufficient lighting and faces are mostly frontal.
  - SQLite WAL mode is used for simple fault-tolerant persistence.

## Compute Load Estimate (Approx)
- **CPU-Only Setup**: ~35-55% load on 4-Core CPU at 30 FPS (`frame_skip=3`).
- **GPU Setup**: ~12-18% CPU, ~25-40% GPU on modern RTX with ~2GB VRAM.

This project is a part of a hackathon run by https://katomaran.com
