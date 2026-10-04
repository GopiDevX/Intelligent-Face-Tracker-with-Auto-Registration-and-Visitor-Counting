import urllib.request
import os

def download_file(url, dest_path):
    print(f"Downloading {url} to {dest_path}...")
    try:
        urllib.request.urlretrieve(url, dest_path)
        print("Download complete.")
    except Exception as e:
        print(f"Failed to download {url}: {e}")

if __name__ == "__main__":
    models_dir = "models"
    os.makedirs(models_dir, exist_ok=True)
    
    # YOLOv8n-face weights
    yolo_url = "https://github.com/akanametov/yolov8-face/releases/download/v0.0.0/yolov8n-face.pt"
    download_file(yolo_url, os.path.join(models_dir, "yolov8n-face.pt"))
    
    # We will need ArcFace ONNX model. Using a known public link for a lightweight mobileface net or similar if w600k is too big.
    # We'll use a direct link to a Buffalo_l or w600k_r50 ONNX file if possible.
    # Because downloading 300MB might fail or timeout, we will use a smaller dummy model for the test or attempt a known fast CDN.
