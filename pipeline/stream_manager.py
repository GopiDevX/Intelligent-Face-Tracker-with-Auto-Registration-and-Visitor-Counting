import cv2

class StreamManager:
    def __init__(self, source):
        self.source = source
        self.cap = cv2.VideoCapture(self.source)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video source: {self.source}")

    def read_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return False, None
        return True, frame

    def release(self):
        if self.cap:
            self.cap.release()
