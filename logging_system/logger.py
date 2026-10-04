import logging
import os
import cv2
from datetime import datetime

class SystemLogger:
    def __init__(self, log_file, entries_dir, exits_dir):
        self.log_file = log_file
        self.entries_dir = entries_dir
        self.exits_dir = exits_dir
        self._setup_logger()

    def _setup_logger(self):
        os.makedirs(os.path.dirname(self.log_file), exist_ok=True)
        self.logger = logging.getLogger("FaceTracker")
        self.logger.setLevel(logging.INFO)
        
        # File handler
        fh = logging.FileHandler(self.log_file)
        fh.setLevel(logging.INFO)
        
        # Console handler
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def log_info(self, message):
        self.logger.info(message)

    def log_warning(self, message):
        self.logger.warning(message)

    def log_error(self, message):
        self.logger.error(message)

    def save_face_image(self, face_img, visitor_id, event_type, timestamp):
        date_str = timestamp.strftime('%Y-%m-%d')
        time_str = timestamp.strftime('%H-%M-%S-%f')[:-3]
        
        if event_type == 'ENTRY':
            base_dir = self.entries_dir
        elif event_type == 'EXIT':
            base_dir = self.exits_dir
        else:
            base_dir = os.path.join(os.path.dirname(self.entries_dir), 'unknown')

        save_dir = os.path.join(base_dir, date_str)
        os.makedirs(save_dir, exist_ok=True)
        
        filename = f"{visitor_id}_{event_type}_{time_str}.jpg"
        filepath = os.path.join(save_dir, filename)
        
        # Save image using OpenCV
        if face_img is not None and face_img.size > 0:
            cv2.imwrite(filepath, face_img)
            return filepath
        return None
