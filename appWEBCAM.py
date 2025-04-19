import os
import cv2
import numpy as np
from flask import Flask, render_template, Response, request
from insightface.app import FaceAnalysis
from datetime import datetime

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

# Lưu log nhận diện
def log_recognition(name):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open("recognition_log.txt", "a") as log_file:
        log_file.write(f"{timestamp} - {name}\n")

# Lưu embedding vector vào file txt
def save_embedding_to_txt(embedding, name, file="datasets/face_features/embeddings.txt"):
    with open(file, "a") as f:
        embedding_str = ' '.join(map(str, embedding))
        f.write(f"{name}: {embedding_str}\n")

# Hàm sinh frame từ webcam
def gen_frames():
    cap = cv2.VideoCapture(0)  # Sử dụng webcam mặc định (0 là chỉ định webcam)
    embeddings = []  # Danh sách các embedding từ các khung hình
    while True:
        success, frame = cap.read()
        if not success:
            break

        faces = face_app.get(frame)
        for face in faces:
            x1, y1, x2, y2 = map(int, face.bbox)
            name, score = find_best_match(face.embedding, known_embeddings, known_names)

            # Vẽ bounding box và tên lên khuôn mặt
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{name} ({score:.2f})" if name != "Unknown" else name
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0) if name != "Unknown" else (0, 0, 255), 2)

            # Lưu log khi nhận diện thành công
            if name != "Unknown":
                log_recognition(name)

            # Lưu ảnh khi phát hiện khuôn mặt
            if name != "Unknown":
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_path = f"recognized_faces/{name}_{timestamp}.jpg"
                if not os.path.exists("recognized_faces"):
                    os.makedirs("recognized_faces")
                cv2.imwrite(image_path, frame)

            # Lưu embedding vào file txt
            save_embedding_to_txt(face.embedding, name)

            # Lưu embedding vào danh sách để tính trung bình
            embeddings.append(face.embedding)

        # Tính trung bình embedding của nhiều khung hình
        if len(embeddings) > 0:
            avg_embedding = average_embeddings(embeddings)
            # Lưu trung bình embedding vào file
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
            name = "new_person"  # Placeholder for new person, can be modified
            embedding = face.embedding
            known_embeddings.append(embedding)
            known_names.append(name)

        # Save the new image for this person
        save_path = f"datasets/new_persons/{name}/{file.filename}"
        if not os.path.exists(f"datasets/new_persons/{name}"):
            os.makedirs(f"datasets/new_persons/{name}")
        cv2.imwrite(save_path, img)
        return "Image uploaded and added!", 200
    return "Something went wrong!", 500

if __name__ == '__main__':
    app.run(debug=True)
