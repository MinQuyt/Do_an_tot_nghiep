import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request
from insightface.app import FaceAnalysis
from datetime import datetime
import mysql.connector
from flask import send_file
import io
import base64

# Hàm load embeddings từ file .npz
def load_embeddings(file="datasets/face_features/embeddings.npz"):
    if not os.path.exists(file):
        raise FileNotFoundError(f"❌ Không tìm thấy file embedding tại {file}")
    data = np.load(file, allow_pickle=True)
    return data['embeddings'], data['names']

# Hàm tính cosine similarity giữa 2 vector
def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

# Hàm tìm người có khuôn mặt tương tự nhất
def find_best_match(embedding, known_embeddings, known_names, threshold=0.5):
    similarities = [cosine_similarity(embedding, k) for k in known_embeddings]
    if similarities:
        best_idx = np.argmax(similarities)
        if similarities[best_idx] > threshold:
            return known_names[best_idx], similarities[best_idx]
    return "Unknown", 0

# Hàm tính trung bình các embedding từ nhiều khung hình
def average_embeddings(embeddings):
    return np.mean(embeddings, axis=0)

# Khởi tạo Flask
app = Flask(__name__)

# Khởi tạo InsightFace
face_app = FaceAnalysis(name="buffalo_l")
face_app.prepare(ctx_id=0, det_size=(640, 640))

# Load embeddings đã được lưu
known_embeddings, known_names = load_embeddings()
print(f"✅ Đã load {len(known_names)} embeddings.")

# Lưu log nhận diện vào file txt
def log_recognition(name):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("recognition_log.txt", "a") as log_file:
        log_file.write(f"{timestamp} - {name}\n")

# ✅ Lưu log nhận diện vào MySQL
def log_to_mysql(name, image_path, similarity=1.0, student_id="unknown", camera_id="cam_01"):
    try:
        # Đọc ảnh thành bytes
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


def create_logs_table_if_not_exists():
    try:
        conn = mysql.connector.connect(
            host='localhost',
            user='root',
            password='minhquy1',
            database='face_recognition'
        )
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS logs (
                id INT AUTO_INCREMENT PRIMARY KEY,
                person_name VARCHAR(100),
                student_id VARCHAR(20),
                similarity FLOAT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                camera_id VARCHAR(50),
                image_blob LONGBLOB
            )
        """)
        conn.commit()
        cursor.close()
        conn.close()
        print("✅ Table 'logs' ensured.")
    except mysql.connector.Error as err:
        print(f"❌ Lỗi khi tạo bảng logs: {err}")

# Lưu embedding vector vào file txt
def save_embedding_to_txt(embedding, name, file="datasets/face_features/embeddings.txt"):
    with open(file, "a") as f:
        embedding_str = ' '.join(map(str, embedding))
        f.write(f"{name}: {embedding_str}\n")

# Hàm sinh frame từ webcam
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

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} ({score:.2f})" if name != "Unknown" else name
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)

            if name != "Unknown":
                log_recognition(name)

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f"recognized_faces/{name}_{timestamp}.jpg"
                if not os.path.exists("recognized_faces"):
                    os.makedirs("recognized_faces")
                cv2.imwrite(image_path, frame)

                # ✅ Ghi vào MySQL
                log_to_mysql(name, image_path)

            save_embedding_to_txt(face.embedding, name)
            embeddings.append(face.embedding)

        if len(embeddings) > 0:
            avg_embedding = average_embeddings(embeddings)
            save_embedding_to_txt(avg_embedding, "average_embedding")

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

# Route chính
@app.route('/')
def index():
    return render_template('index.html')

# Route stream video
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route upload ảnh để thêm vào hệ thống nhận diện
@app.route('/upload_image', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    if file:
        img = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
        faces = face_app.get(img)
        for face in faces:
            name = "new_person"  # Placeholder
            embedding = face.embedding
            known_embeddings.append(embedding)
            known_names.append(name)

        save_path = f"datasets/new_persons/{name}/{file.filename}"
        if not os.path.exists(f"datasets/new_persons/{name}"):
            os.makedirs(f"datasets/new_persons/{name}")
        cv2.imwrite(save_path, img)
        
        # Send the image as base64 for display in the frontend
        img_base64 = base64.b64encode(file.read()).decode('utf-8')
        return render_template('upload_image.html', img_base64=img_base64)

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
        cursor.execute("SELECT id, person_name, student_id, similarity, timestamp, camera_id, image_blob FROM logs ORDER BY timestamp DESC")
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        # Chuyển dữ liệu sang danh sách dictionary
        logs = []
        for row in rows:
            img_base64 = base64.b64encode(row[6]).decode('utf-8') if row[6] else None
            logs.append({
                'id': row[0],
                'name': row[1],
                'student_id': row[2],
                'similarity': f"{row[3]:.2f}",
                'timestamp': row[4].strftime("%Y-%m-%d %H:%M:%S"),
                'camera_id': row[5],
                'image_base64': img_base64
            })

        return render_template("logs.html", logs=logs)

    except mysql.connector.Error as err:
        return f"❌ Lỗi khi truy vấn logs: {err}", 500

if __name__ == '__main__':
    # Tạo bảng logs nếu chưa có
    create_logs_table_if_not_exists()
    
    app.run(debug=True)
