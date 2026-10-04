from flask import Flask, render_template, Response, jsonify
import cv2
import threading
import json
from database.db_manager import DatabaseManager
import os

app = Flask(__name__)

# Assuming DB is in data/tracker.db as per config.json
DB_PATH = "data/tracker.db"
db = DatabaseManager(DB_PATH)

# Global variables for stream (in a real app, use better IPC)
current_frame = None
lock = threading.Lock()

def update_frame(frame):
    global current_frame
    with lock:
        current_frame = frame

def generate_video_stream():
    global current_frame
    while True:
        with lock:
            if current_frame is None:
                continue
            
            # Encode frame to JPEG
            ret, buffer = cv2.imencode('.jpg', current_frame)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
        
        # Yield in multipart format
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_video_stream(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/stats')
def stats():
    try:
        unique_visitors = db.get_unique_visitor_count()
        return jsonify({
            'status': 'success',
            'unique_visitors': unique_visitors
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    # Ensure data directory exists
    os.makedirs('data', exist_ok=True)
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
