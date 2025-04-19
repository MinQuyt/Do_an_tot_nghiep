# Do_an_tot_nghiep
I: Cài đặt môi trường ảo Conda và cài đặt các thư viện cần thiết
-  Download Miniconda: https://www.anaconda.com/docs/getting-started/miniconda/install
-  Mở màn hình CommandLine với quyền admin để tạo các môi trường ảo
  ```shell
   conda create -n quy_face4 -c conda-forge onnxruntime python=3.9 //Tạo môi trường ảo, python phiên bản 3.9, và cài onnxruntime để có thể sử dụng GPU để huấn luyện models
   ```
Download thư viện insightface từ trên github về máy https://github.com/cobanov/insightface_windows/tree/main/whls

  ```shell
   pip install C:\Users\Admin\Downloads\insightface_windows-main\whls\insightface-0.7.3-cp39-cp39-win_amd64.whl // chạy lệnh để cài thư viện vào môi trường ảo
   ```
Download các thư viện cần thiết:
  ```shell
  pip install flask opencv-python numpy insightface mysql-connector-python
  ```
    
II,### Thêm người mới vào dataset

1. **Create a folder with the folder name being the name of the person**

   ```
   datasets/
   ├── backup
   ├── data
   ├── face_features
   └── new_persons
       ├── name-person1
       └── name-person2
   ```

2. **Add the person's photo in the folder**

   ```
   datasets/
   ├── backup
   ├── data
   ├── face_features
   └── new_persons
       ├── name-person1
       │   └── image1.jpg
       │   └── image2.jpg
       └── name-person2
           └── image1.jpg
           └── image2.jpg
   ```

3. **Run to add new persons**

   ```shell
   python save_embeddings.py
   ```

4. **Run to recognize**

   ```shell
   python app.py
   ```
## Technology

### Face Detection

1. **Retinaface**

   - Retinaface is a powerful face detection algorithm known for its accuracy and speed. It utilizes a single deep convolutional network to detect faces in an image with high precision.

2. **Yolov5-face**

   - Yolov5-face is based on the YOLO (You Only Look Once) architecture, specializing in face detection. It provides real-time face detection with a focus on efficiency and accuracy.

3. **SCRFD**
   - SCRFD (Single-Shot Scale-Aware Face Detector) is designed for real-time face detection across various scales. It is particularly effective in detecting faces at different resolutions within the same image.

### Face Recognition

1. **ArcFace**

   - ArcFace is a state-of-the-art face recognition algorithm that focuses on learning highly discriminative features for face verification and identification. It is known for its robustness to variations in lighting, pose, and facial expressions.

    <p align="center">
    <img src="./Screenshot 2025-03-15 164219.png" alt="ArcFace" />
    <br>
    <em>ArcFace</em>
    </p>
## Reference
- [InsightFace - ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/arcface_torch)
- [face-recognition](https://github.com/vectornguyen76/face-recognition.git)
- [InsightFace-REST](https://github.com/SthPhoenix/InsightFace-REST)
