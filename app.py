import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request,jsonify
from insightface.app import FaceAnalysis
from datetime import datetime
import mysql.connector
import pyttsx3
import time
import threading
import requests
import json
import google.generativeai as genai
from googleapiclient.discovery import build
from youtubesearchpython import VideosSearch
import re

YOUTUBE_API_KEY = 'AIzaSyBrKT29tJXuBtB4SN-dHS1nh0n9GzaIyeY'
YOUTUBE_API_SERVICE_NAME = 'youtube'
YOUTUBE_API_VERSION = 'v3'
genai.configure(api_key="AIzaSyAnVlLuUKPq69nlMIoZO4TNeRUSqzB_ww0")
model = genai.GenerativeModel("gemini-1.5-flash")
TELEGRAM_BOT_TOKEN = '7626448762:AAGfIpL2CmTvm0MGX4ScQtfLID9vv_k8h80'
TELEGRAM_CHAT_ID = '6818084029'
DISCORD_WEBHOOK_URL = "https://discord.com/api/webhooks/1375503695575388283/7QJvOr_guPoZxSQKxgn9BG7PX7KvsIveyD1bDI7pYeYMX0rFSFAmtN7pRGeOHYUn48gr"


def process_gpt_message(message: str) -> str:
    try:
        # gọi Gemini với prompt nhận được
        response = model.generate_content(message)
        return response.text
    except Exception as e:
        print(f"❌ Lỗi khi gọi Gemini: {e}")
        return "❌ Lỗi kết nối chatbot."

def send_discord_message(message, image_path=None):
    try:
        data = {"content": message}
        files = None

        if image_path and os.path.exists(image_path):
            with open(image_path, "rb") as f:
                files = {"file": (os.path.basename(image_path), f)}
                requests.post(DISCORD_WEBHOOK_URL, data=data, files=files)
        else:
            requests.post(DISCORD_WEBHOOK_URL, json=data)
    except Exception as e:
        print(f"❌ Discord error: {e}")

def notify_discord_async(message, image_path=None):
    threading.Thread(target=send_discord_message, args=(message, image_path), daemon=True).start()

def send_telegram_message(message, image_path=None):
    try:
        if image_path and os.path.exists(image_path):
            with open(image_path, 'rb') as photo:
                files = {'photo': photo}
                data = {"chat_id": TELEGRAM_CHAT_ID, "caption": message}
                url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendPhoto"
                requests.post(url, data=data, files=files)
        else:
            data = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, data=data)
    except Exception as e:
        print(f"❌ Telegram error: {e}")

def notify_telegram_async(message, image_path=None):
    threading.Thread(target=send_telegram_message, args=(message, image_path), daemon=True).start()

def notify_all_async(message, image_path=None):
    notify_telegram_async(message, image_path)
    notify_discord_async(message, image_path)
# --- Phần nhận diện mặt (giữ nguyên) ---

def load_embeddings(file="datasets/face_features/embeddings.npz"):
    if not os.path.exists(file):
        raise FileNotFoundError(f"❌ Không tìm thấy file embedding tại {file}")
    data = np.load(file, allow_pickle=True)
    return data['embeddings'], data['names']

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def find_best_match(embedding, known_embeddings, known_names, threshold=0.5):
    similarities = [cosine_similarity(embedding, k) for k in known_embeddings]
    if similarities:
        best_idx = np.argmax(similarities)
        if similarities[best_idx] > threshold:
            return known_names[best_idx], similarities[best_idx]
    return "Unknown", 0

app = Flask(__name__)
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))
known_embeddings, known_names = load_embeddings()
print(f"✅ Đã load {len(known_names)} embeddings.")
last_recognition_time = {}

def play_welcome_sound(name):
    engine = pyttsx3.init()
    clean_name = ' '.join(part.capitalize() for part in name.split('_'))
    engine.say(f"Xin chào {clean_name}, Wellcome to IOTLab Dai Nam!")
    engine.runAndWait()

def log_recognition(name):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("recognition_log.txt", "a") as log_file:
        log_file.write(f"{timestamp} - {name}\n")

def log_to_mysql(name, image_path, similarity=1.0, student_id="unknown", camera_id="lab_aiot"):
    try:
        with open(image_path, 'rb') as img_file:
            image_blob = img_file.read()

        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='minhquy1',
            database='face_recognition'
        )
        cursor = conn.cursor()

        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        sql = """
            INSERT INTO logs (person_name, student_id, similarity, timestamp, camera_id, image_blob)
            VALUES (%s, %s, %s, %s, %s, %s)
        """
        cursor.execute(sql, (name, student_id, similarity, now, camera_id, image_blob))
        conn.commit()
        cursor.close()
        conn.close()
        print(f"✅ Đã lưu log vào MySQL kèm ảnh cho {name}")
    except mysql.connector.Error as err:
        print(f"❌ Lỗi khi ghi log vào MySQL: {err}")
    except Exception as e:
        print(f"❌ Lỗi xử lý ảnh: {e}")

def save_embedding_to_txt(embedding, name, file="datasets/face_features/embeddings.txt"):
    with open(file, "a") as f:
        embedding_str = ' '.join(map(str, embedding))
        f.write(f"{name}: {embedding_str}\n")




RECOGNITION_INTERVAL = 60

def gen_frames():
    cap = cv2.VideoCapture(0)
    embeddings = []

    while True:
        success, frame = cap.read()
        if not success:
            break

        faces = face_app.get(frame)
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            name, score = find_best_match(face.embedding, known_embeddings, known_names)
            score = float(score)

            # Vẽ khung và nhãn
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} ({score:.2f})" if name != "Unknown" else name
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)

            # Tạo thư mục & tên file ảnh
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("recognized_faces", exist_ok=True)

            if name != "Unknown":
                current_time = time.time()
                last_time = last_recognition_time.get(name, 0)

                if current_time - last_time > RECOGNITION_INTERVAL:
                    # Phát âm thanh chào mừng
                    threading.Thread(target=play_welcome_sound, args=(name,), daemon=True).start()

                    image_path = f"recognized_faces/{name}_{timestamp}.jpg"
                    cv2.imwrite(image_path, frame)

                    student_id = name[-4:] if name[-4:].isdigit() else "unknown"

                    # Ghi log vào MySQL
                    threading.Thread(target=log_to_mysql, args=(name, image_path, score, student_id), daemon=True).start()

                    # Gửi thông báo qua Telegram và Discord
                    notify_all_async(
                        f"📸 Đã nhận diện: {name} (ID: {student_id}) với độ tương đồng {score:.2f}",
                        image_path
                    )

                    last_recognition_time[name] = current_time
                else:
                    print(f"🔁 {name} đã được nhận diện gần đây, không ghi log.")
            else:
                image_path = f"recognized_faces/unknown_{timestamp}.jpg"
                cv2.imwrite(image_path, frame)

                # Gửi thông báo phát hiện người lạ
                notify_all_async("⚠️ Phát hiện người lạ không xác định!", image_path)

            save_embedding_to_txt(face.embedding, name)
            embeddings.append(face.embedding)

        # Lưu embedding trung bình
        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            save_embedding_to_txt(avg_embedding, "average_embedding")

        # Truyền frame qua stream
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

SONG_MAP = {
    "nơi này có anh": "https://www.youtube.com/watch?v=FN7ALfpGxiI",
    "chúng ta của hiện tại": "https://www.youtube.com/watch?v=psZ1g9fMfeo",
    "hãy trao cho anh": "https://www.youtube.com/watch?v=knW7-x7Y7RE",
    "em của ngày hôm qua": "https://www.youtube.com/watch?v=Vt4kAu-ziRY"
    # Bạn có thể thêm nhiều bài hơn
}

def search_youtube(query):
    try:
        youtube = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)
        request = youtube.search().list(
            q=query,
            part="snippet",
            maxResults=1,
            type="video",
            videoCategoryId="10",  # Music
            videoDuration="any",  # Hoặc "any" nếu bạn muốn tất cả
            safeSearch="moderate"
        )
        response = request.execute()

        items = response.get("items", [])
        if not items:
            return "❌ Không tìm thấy video phù hợp."

        video_id = items[0]["id"]["videoId"]
        video_title = items[0]["snippet"]["title"]

        return f"🎵 Đang mở bài: {video_title}\nhttps://youtu.be/{video_id}"

    except Exception as e:
        print(f"❌ Lỗi khi gọi YouTube API: {e}")
        return "❌ Đã xảy ra lỗi khi tìm kiếm video từ YouTube."

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/logs')
def view_logs():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='minhquy1',
            database='face_recognition'
        )
        cursor = conn.cursor()
        cursor.execute("SELECT person_name, student_id, similarity, timestamp, camera_id FROM logs ORDER BY timestamp DESC")
        logs = cursor.fetchall()
        cursor.close()
        conn.close()
        return render_template('logs.html', logs=logs)
    except mysql.connector.Error as err:
        return f"Lỗi kết nối CSDL: {err}"

# --- Phần bot Telegram chạy song song ---
# Đây là ví dụ đơn giản chạy bot Telegram độc lập, có thể lắng nghe và trả lời tin nhắn
# Bạn có thể mở rộng để xử lý tin nhắn hoặc gọi send_telegram_message từ app face recognition

def telegram_bot_listener():
    last_update_id = None
    print("🤖 Telegram bot listener started...")

    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
            if last_update_id:
                url += f"?offset={last_update_id + 1}"

            response = requests.get(url).json()
            for update in response.get("result", []):
                last_update_id = update["update_id"]
                message = update.get("message", {}).get("text", "")
                chat_id = update.get("message", {}).get("chat", {}).get("id")

                if message:
                    reply = process_gpt_message(message)
                    send_telegram_message(reply)
            time.sleep(2)
        except Exception as e:
            print(f"❌ Lỗi Telegram listener: {e}")
            time.sleep(5)
@app.route('/gpt', methods=['POST'])
def gpt():
    data = request.get_json()
    msg = data.get("message", "").strip()
    msg_lower = msg.lower()

    if not msg:
        return jsonify({"reply": "❌ Vui lòng nhập câu hỏi."})

    # Nếu người dùng yêu cầu mở video
    if any(kw in msg_lower for kw in ['mở', 'play', 'bật', 'phát bài', 'mở nhạc']):
        query = msg_lower
        for prefix in ['mở', 'play', 'bật', 'phát bài', 'mở nhạc']:
            query = query.replace(prefix, '')
        query = query.strip()

        # Ưu tiên kiểm tra trong SONG_MAP trước
        for title, url in SONG_MAP.items():
            if title in query:
                return jsonify({'reply': f'🎵 Đang mở bài: {title.title()}\n{url}'})

        # Nếu không tìm thấy, fallback sang tìm trên YouTube
        video_url = search_youtube(query)
        return jsonify({'reply': f'🎵 Đang mở bài: {query}\n{video_url}'})

    # Nếu không phải yêu cầu mở video, xử lý như GPT bình thường
    reply = process_gpt_message(msg)
    return jsonify({"reply": reply})

@app.route('/dashboard')
def dashboard():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='minhquy1',
            database='face_recognition'
        )
        cursor = conn.cursor(dictionary=True)

        # Dữ liệu thống kê
        cursor.execute("""
            SELECT COUNT(*) AS total_logs FROM logs 
            WHERE DATE(timestamp) = CURDATE()
        """)
        total_logs = cursor.fetchone()['total_logs']

        cursor.execute("""
            SELECT COUNT(*) AS unknowns FROM logs 
            WHERE person_name = 'Unknown' AND DATE(timestamp) = CURDATE()
        """)
        unknowns = cursor.fetchone()['unknowns']

        cursor.execute("""
            SELECT person_name, COUNT(*) as appearances 
            FROM logs WHERE DATE(timestamp) = CURDATE() AND person_name != 'Unknown' 
            GROUP BY person_name ORDER BY appearances DESC LIMIT 1
        """)
        top_user_row = cursor.fetchone()
        top_user = top_user_row['person_name'] if top_user_row else "N/A"
        top_count = top_user_row['appearances'] if top_user_row else 0

        # Lấy 10 log gần nhất
        cursor.execute("""
            SELECT person_name, student_id, similarity, timestamp, camera_id 
            FROM logs ORDER BY timestamp DESC LIMIT 10
        """)
        logs = cursor.fetchall()

        cursor.close()
        conn.close()

        return render_template("dashboard.html", total_logs=total_logs, unknowns=unknowns,
                               top_user=top_user, top_count=top_count, logs=logs)
    except mysql.connector.Error as err:
        return f"Lỗi truy vấn dashboard: {err}"

if __name__ == '__main__':
    # Chạy bot Telegram listener song song thread riêng
    threading.Thread(target=telegram_bot_listener, daemon=True).start()

    # Chạy app Flask chính
    app.run(debug=True)
