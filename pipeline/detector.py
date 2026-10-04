import cv2
from ultralytics import YOLO

class FaceDetector:
    def __init__(self, model_path="yolov8n-face.pt", conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        try:
            # We assume a YOLOv8 face model is provided or downloaded automatically
            self.model = YOLO(model_path)
        except Exception as e:
            print(f"Failed to load YOLO model: {e}")
            self.model = None

    def detect(self, frame):
        """
        Detect faces in the frame.
        Returns a list of bounding boxes: [x1, y1, x2, y2, confidence]
        """
        if self.model is None:
            return []

        # Run inference
        results = self.model(frame, conf=self.conf_threshold, verbose=False)
        
        boxes = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                boxes.append([int(x1), int(y1), int(x2), int(y2), conf])
                
        return boxes
