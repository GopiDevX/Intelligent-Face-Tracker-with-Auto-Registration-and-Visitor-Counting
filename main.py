import json
import cv2
import os
from datetime import datetime
from database.db_manager import DatabaseManager
from logging_system.logger import SystemLogger
from pipeline.detector import FaceDetector
from pipeline.recognizer import FaceRecognizer
from pipeline.tracker import FaceTracker
from pipeline.stream_manager import StreamManager

def load_config(config_path="config.json"):
    with open(config_path, "r") as f:
        return json.load(f)

def main():
    # 1. Load Configuration
    config = load_config()
    db_path = config["storage"]["db_path"]
    log_file = config["storage"]["log_file"]
    entries_dir = config["storage"]["entries_dir"]
    exits_dir = config["storage"]["exits_dir"]
    frame_skip = config["pipeline"]["frame_skip"]
    sim_threshold = config["pipeline"]["recognition_similarity_threshold"]
    max_lost_frames = config["pipeline"]["max_lost_frames"]

    # 2. Initialize Subsystems
    db = DatabaseManager(db_path)
    logger = SystemLogger(log_file, entries_dir, exits_dir)
    
    detector = FaceDetector(model_path="models/yolov8n-face.pt", conf_threshold=config["pipeline"]["detection_confidence_threshold"])
    recognizer = FaceRecognizer(model_path="models/w600k_r50.onnx")
    tracker = FaceTracker(db, logger, sim_threshold=sim_threshold, max_lost_frames=max_lost_frames)

    # 3. Setup Video Source (0 for webcam, or path for video file / RTSP)
    # Ideally passed via argument, defaulting to a sample path
    video_source = r"sample videos\sample1.mp4" 
    try:
        stream = StreamManager(video_source)
    except Exception as e:
        logger.log_error(str(e))
        return

    logger.log_info(f"System Initialized. Starting video processing from {video_source}")

    frame_count = 0
    while True:
        ret, frame = stream.read_frame()
        if not ret:
            logger.log_info("End of video stream.")
            break

        timestamp = datetime.now()

        # Frame skip logic for performance
        if frame_count % frame_skip == 0:
            # Detect
            boxes = detector.detect(frame)
            
            frame_data = []
            for box in boxes:
                x1, y1, x2, y2, conf = box
                # Margin for cropping
                margin_x = int((x2 - x1) * 0.1)
                margin_y = int((y2 - y1) * 0.1)
                
                cx1 = max(0, x1 - margin_x)
                cy1 = max(0, y1 - margin_y)
                cx2 = min(frame.shape[1], x2 + margin_x)
                cy2 = min(frame.shape[0], y2 + margin_y)
                
                face_crop = frame[cy1:cy2, cx1:cx2].copy()
                
                # Recognize
                embedding = recognizer.get_embedding(face_crop)
                
                frame_data.append({
                    'box': [x1, y1, x2, y2, conf],
                    'embedding': embedding,
                    'crop': face_crop
                })

            # Track & Log
            tracker.process_frame_detections(frame_data, timestamp, frame)

        # Drawing logic for visual feedback (if UI enabled)
        # ... (can be extended for a WebUI/cv2.imshow)

        frame_count += 1

    stream.release()
    logger.log_info(f"Total Unique Visitors Count: {db.get_unique_visitor_count()}")

if __name__ == "__main__":
    main()
