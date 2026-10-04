import cv2
import numpy as np
import urllib.request
import os

print("Generating synthetic video with a sliding person...")

# Download a stock photo of a person
image_url = "https://images.unsplash.com/photo-1544005313-94ddf0286df2?ixlib=rb-4.0.3&w=400&q=80"
req = urllib.request.urlopen(image_url)
arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
person_img = cv2.imdecode(arr, -1)
person_img = cv2.resize(person_img, (200, 300))

width, height = 800, 600
out = cv2.VideoWriter('sample_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (width, height))

for i in range(90):
    frame = np.ones((height, width, 3), dtype=np.uint8) * 200
    
    # Person enters from left, moves right, then exits
    x = int(i * 10) - 100
    y = 150
    
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(width, x + person_img.shape[1])
    y2 = min(height, y + person_img.shape[0])
    
    px1 = max(0, -x)
    py1 = max(0, -y)
    px2 = px1 + (x2 - x1)
    py2 = py1 + (y2 - y1)
    
    if x2 > x1 and y2 > y1:
        frame[y1:y2, x1:x2] = person_img[py1:py2, px1:px2]
        
    out.write(frame)

out.release()
print("sample_video.mp4 generated successfully.")
