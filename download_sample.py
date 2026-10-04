import urllib.request
import os

video_url = "https://www.w3schools.com/html/mov_bbb.mp4" # Big Buck Bunny snippet
dest_path = "sample_video.mp4"

print(f"Downloading sample video to {dest_path}...")
try:
    urllib.request.urlretrieve(video_url, dest_path)
    print("Download complete.")
except Exception as e:
    print(f"Failed to download video: {e}")
