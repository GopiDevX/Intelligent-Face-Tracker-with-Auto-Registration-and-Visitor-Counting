import cv2
import numpy as np
import onnxruntime as ort

class FaceRecognizer:
    def __init__(self, model_path="models/w600k_r50.onnx"):
        self.model_path = model_path
        try:
            self.session = ort.InferenceSession(self.model_path, providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
            self.input_name = self.session.get_inputs()[0].name
            self.input_shape = self.session.get_inputs()[0].shape
        except Exception as e:
            print(f"Warning: Failed to load ONNX face recognition model: {e}")
            self.session = None

    def preprocess(self, face_img):
        face_img = cv2.resize(face_img, (112, 112))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = np.transpose(face_img, (2, 0, 1))
        face_img = (face_img / 127.5) - 1.0
        face_img = np.expand_dims(face_img, axis=0).astype(np.float32)
        return face_img

    def get_embedding(self, face_img):
        if face_img is None or face_img.size == 0:
            return np.zeros(512, dtype=np.float32)
            
        if self.session is None:
            # Fallback for testing when ONNX model is missing:
            # Generate a pseudo-embedding based on the crop's center color/histogram
            # This ensures different objects get different embeddings.
            small_img = cv2.resize(face_img, (16, 32)).flatten().astype(np.float32)
            # Pad or truncate to 512
            if small_img.shape[0] < 512:
                small_img = np.pad(small_img, (0, 512 - small_img.shape[0]))
            else:
                small_img = small_img[:512]
            # Normalize
            norm = np.linalg.norm(small_img)
            if norm > 0:
                small_img = small_img / norm
            return small_img

        input_tensor = self.preprocess(face_img)
        embedding = self.session.run(None, {self.input_name: input_tensor})[0][0]
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding

    @staticmethod
    def compute_similarity(emb1, emb2):
        return np.dot(emb1, emb2)
