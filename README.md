
## ⚙️ Create Environment and Install Packages

```bash
conda create -n quy_face -c conda-forge onnxruntime python=3.9
conda activate face-dev
```

```bash
pip install -r requirements.txt
```

> ⚠️ Yêu cầu:
> - Python 3.9
> - ONNX Runtime (GPU)
> - MySQL Connector
> - pip install Flask pyttsx3 mysql-connector SpeechRecognition pyaudio  python-telegram-bot pymysql flask opencv-python numpy insightface
> - pip install "C:\Users\Admin\Downloads\insightface_windows-main\whls\insightface-0.7.3-cp39-cp39-win_amd64.whl" 
---

## 🧑‍💼 Add New Persons to Dataset

1. **Tạo thư mục với tên tương ứng người cần thêm**

   ```
   datasets/
   ├── backup/
   ├── data/
   ├── face_features/
   └── new_persons/
       ├── tran-minh-quy/
       └── nguyen-van-a/
   ```

2. **Thêm ảnh chân dung rõ mặt vào từng thư mục**

   ```
   └── new_persons/
       ├── tran-minh-quy/
       │   ├── image1.jpg
       │   └── image2.jpg
       └── nguyen-van-a/
           ├── img1.jpg
           └── img2.jpg
   ```

3. **Chạy script lưu embeddings của dữ liệu trong dataset vào datasets/face_features/embeddings.npz**

   ```bash
   python save_embeddings.py
   ```

4. **Chạy nhận diện khuôn mặt (thực thi chính)**

   ```bash
   python app.py
   ```

---

## 📦 Technology Stack

### 🧠 Face Detection

- **RetinaFace (`det_10g.onnx`)**  
  Dùng để phát hiện khuôn mặt trong ảnh/video, độ chính xác cao trong nhiều điều kiện ánh sáng và góc độ khác nhau.

- **SCRFD (optional)**  
  Một lựa chọn thay thế nhẹ hơn nếu cần tối ưu tốc độ.

### 🧠 Face Recognition

- **ArcFace (`w600k_r50.onnx`)**  
  Trích xuất vector đặc trưng 512 chiều từ khuôn mặt để nhận diện. So sánh vector bằng cosine similarity.

---

## 🛢️ Database - MySQL

### 🧱 Cấu trúc bảng `logs`

| Trường         | Kiểu dữ liệu     | Mô tả                          |
|----------------|------------------|--------------------------------|
| `id`           | INT (PK)         | ID tự tăng                    |
| `person_name`  | VARCHAR(100)     | Tên người nhận diện           |
| `student_id`   | VARCHAR(20)      | Mã sinh viên (nếu có)         |
| `student_name` | VARCHAR(255)     | Họ tên sinh viên              |
| `similarity`   | FLOAT            | Điểm tương đồng với dữ liệu   |
| `timestamp`    | DATETIME         | Thời điểm ghi nhận            |
| `camera_id`    | VARCHAR(50)      | ID camera (nếu nhiều camera)  |
| `image_blob`   | LONGBLOB         | Ảnh cắt khuôn mặt đã nhận diện|

> ❗ Vector nhúng không được lưu trong cơ sở dữ liệu, mà lưu riêng trong thư mục `datasets/face_features/` với tên file tương ứng.

---

## 📲 Telegram Integration

- Bot Token: `7626448762:AAGfIpL2CmTvm0MGX4ScQtfLID9vv_k8h80`
- Admin Chat ID: `6818084029`

> Khi phát hiện khuôn mặt lạ (chưa có trong hệ thống), bot sẽ gửi ảnh và thông tin cảnh báo đến điện thoại qua Telegram.

---

## 🖼️ Demo Architecture

```
 [Camera IP]
         ↓
[Face Detection (RetinaFace)]
         ↓
[Face Embedding (ArcFace)]
         ↓
[So sánh Cosine Similarity]
         ↓            ↘
 [Đã nhận dạng]     [Người lạ]
         ↓              ↓
 [Ghi log + tên]    [Gửi Telegram]
         ↓              ↓
 [Lưu DB MySQL]   [Lưu DB MySQL]
         ↓              ↓
 [Hiển thị tên]   [Ghi "Unknown"]
```

---

## 📚 Reference

- 🔗 [InsightFace - ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)
- 🔗 [VectorNguyen - Face Recognition](https://github.com/vectornguyen76/face-recognition.git)
- 🔗 [InsightFace-REST](https://github.com/SthPhoenix/InsightFace-REST)
