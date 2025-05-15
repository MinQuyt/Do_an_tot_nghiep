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


genai.configure(api_key="AIzaSyAnVlLuUKPq69nlMIoZO4TNeRUSqzB_ww0")
model = genai.GenerativeModel("gemini-1.5-flash")
TELEGRAM_BOT_TOKEN = '7626448762:AAGfIpL2CmTvm0MGX4ScQtfLID9vv_k8h80'
TELEGRAM_CHAT_ID = '6818084029'

def process_gpt_message(message: str) -> str:
    try:
        # gọi Gemini với prompt nhận được
        response = model.generate_content(message)
        return response.text
    except Exception as e:
        print(f"❌ Lỗi khi gọi Gemini: {e}")
        return "❌ Lỗi kết nối chatbot."
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

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} ({score:.2f})" if name != "Unknown" else name
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            os.makedirs("recognized_faces", exist_ok=True)

            if name != "Unknown":
                current_time = time.time()
                last_time = last_recognition_time.get(name, 0)

                if current_time - last_time > RECOGNITION_INTERVAL:
                    threading.Thread(target=play_welcome_sound, args=(name,), daemon=True).start()

                    image_path = f"recognized_faces/{name}_{timestamp}.jpg"
                    cv2.imwrite(image_path, frame)
                    student_id = name[-4:] if name[-4:].isdigit() else "unknown"

                    threading.Thread(target=log_to_mysql, args=(name, image_path, score, student_id), daemon=True).start()

                    # Gửi Telegram message async
                    notify_telegram_async(
                        f"📸 Đã nhận diện: {name} (ID: {student_id}) với độ tương đồng {score:.2f}",
                        image_path,
                    )

                    last_recognition_time[name] = current_time
                else:
                    print(f"🔁 {name} đã được nhận diện gần đây, không ghi log.")
            else:
                image_path = f"recognized_faces/unknown_{timestamp}.jpg"
                cv2.imwrite(image_path, frame)

                notify_telegram_async("⚠️ Phát hiện người lạ không xác định!", image_path)

            save_embedding_to_txt(face.embedding, name)
            embeddings.append(face.embedding)

        if len(embeddings) > 0:
            avg_embedding = np.mean(embeddings, axis=0)
            save_embedding_to_txt(avg_embedding, "average_embedding")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        img = cv2.imdecode(np.frombuffer(file.read(), np.uint8), cv2.IMREAD_COLOR)
        faces = face_app.get(img)
        for face in faces:
            name = "new_person"
            embedding = face.embedding
            known_embeddings.append(embedding)
            known_names.append(name)

        save_path = f"datasets/new_persons/{name}/{file.filename}"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        cv2.imwrite(save_path, img)

        return "File uploaded successfully!"

    return "Something went wrong!", 500

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
    import time
    last_update_id = None
    print("🤖 Telegram bot listener started...")
    while True:
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/getUpdates"
            if last_update_id:
                url += f"?offset={last_update_id + 1}"
            response = requests.get(url)
            data = response.json()

            if data["ok"]:
                for update in data["result"]:
                    last_update_id = update["update_id"]
                    message = update.get("message")
                    if message:
                        chat_id = message["chat"]["id"]
                        text = message.get("text", "")
                        # Ví dụ phản hồi đơn giản
                        if text.lower() == "hello":
                            send_telegram_message("Hello! Tôi là bot nhận diện khuôn mặt.", None)
                        elif text.lower().startswith("echo "):
                            send_telegram_message(text[5:], None)
                        else:
                            send_telegram_message("Bạn có thể gửi 'hello' hoặc 'echo <text>'.", None)
            time.sleep(1)
        except Exception as e:
            print(f"❌ Lỗi bot Telegram listener: {e}")
            time.sleep(5)
@app.route('/gpt', methods=['POST'])
def gpt():
    data = request.get_json()
    msg = data.get("message", "").strip()
    if not msg:
        return jsonify({"reply": "❌ Vui lòng nhập câu hỏi."})
    # gửi nội dung người dùng tới Gemini
    reply = process_gpt_message(msg)
    return jsonify({"reply": reply})

if __name__ == '__main__':
    # Chạy bot Telegram listener song song thread riêng
    threading.Thread(target=telegram_bot_listener, daemon=True).start()

    # Chạy app Flask chính
    app.run(debug=True)
