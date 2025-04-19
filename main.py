# main.py (Flask App)
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from datetime import datetime
import cv2
import numpy as np
import os
import pyttsx3
import sqlite3
from utils.face_module import detect_and_recognize, send_telegram_alert, init_model

app = Flask(__name__)
CORS(app)

# Global model and paths
model, detector = init_model()
DATABASE = 'face_logs.db'
TELEGRAM_CHAT_ID = 'YOUR_CHAT_ID'
TELEGRAM_BOT_TOKEN = 'YOUR_BOT_TOKEN'

# Create DB if not exists
def init_db():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS logs
                 (id INTEGER PRIMARY KEY, name TEXT, timestamp TEXT, similarity REAL, image_path TEXT)''')
    conn.commit()
    conn.close()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/logs')
def get_logs():
    conn = sqlite3.connect(DATABASE)
    c = conn.cursor()
    c.execute("SELECT * FROM logs ORDER BY timestamp DESC")
    logs = c.fetchall()
    conn.close()
    return jsonify(logs)

@app.route('/upload_frame', methods=['POST'])
def upload_frame():
    file = request.files['frame']
    npimg = np.frombuffer(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    name, sim, face_img = detect_and_recognize(frame, detector, model)
    timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

    if name != "Unknown":
        img_path = f'logs/{timestamp}_{name}.jpg'
        cv2.imwrite(img_path, face_img)

        # Save to DB
        conn = sqlite3.connect(DATABASE)
        c = conn.cursor()
        c.execute("INSERT INTO logs (name, timestamp, similarity, image_path) VALUES (?, ?, ?, ?)",
                  (name, timestamp, sim, img_path))
        conn.commit()
        conn.close()

        # TTS
        tts = pyttsx3.init()
        tts.say(f"Xin chao {name}!")
        tts.runAndWait()
    else:
        img_path = f'logs/{timestamp}_unknown.jpg'
        cv2.imwrite(img_path, face_img)
        send_telegram_alert(img_path, TELEGRAM_CHAT_ID, TELEGRAM_BOT_TOKEN)

    return jsonify({"name": name, "similarity": sim})

if __name__ == '__main__':
    os.makedirs("logs", exist_ok=True)
    init_db()
    app.run(host='0.0.0.0', port=5000)
