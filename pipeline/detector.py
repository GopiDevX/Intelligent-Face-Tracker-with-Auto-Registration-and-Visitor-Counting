import cv2
from ultralytics import YOLO
import os

class FaceDetector:
    def __init__(self, model_path="yolov8n-face.pt", conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        
        # If the face model doesn't exist locally, fallback to standard YOLOv8n to detect people (for testing purposes)
        if not os.path.exists(model_path):
            print(f"Warning: {model_path} not found. Falling back to standard yolov8n.pt (detecting class 0: person).")
            model_path = "yolov8n.pt"
            
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect(self, frame):
        """
        Detect faces (or persons in fallback) in the frame.
        Returns a list of bounding boxes: [x1, y1, x2, y2, confidence]
        """
        if self.model is None:
            return []

        # Run inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        boxes = []
        for result in results:
            for box in result.boxes:
                # If using standard YOLO, optionally filter for class 0 (person)
                cls_id = int(box.cls[0])
                if "yolov8n.pt" in self.model.ckpt_path and cls_id != 0:
                    continue
                    
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                boxes.append([int(x1), int(y1), int(x2), int(y2), conf])
                
        return boxes
