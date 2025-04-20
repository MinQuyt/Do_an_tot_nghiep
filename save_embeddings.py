import os
import cv2
import numpy as np
from insightface.app import FaceAnalysis

# Hàm load khuôn mặt từ thư mục và tạo embedding
def load_and_encode_faces(base_dir="datasets/new_persons"):
    known_embeddings = []
    known_names = []

    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=0)

    # Lấy tất cả ảnh từ thư mục new_persons
    for person_name in os.listdir(base_dir):
        person_folder = os.path.join(base_dir, person_name)
        if not os.path.isdir(person_folder):
            continue

        for filename in os.listdir(person_folder):
            img_path = os.path.join(person_folder, filename)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Không thể đọc ảnh: {img_path}")
                continue

            faces = app.get(img)
            if len(faces) > 0:
                embedding = faces[0].embedding
                known_embeddings.append(embedding)
                known_names.append(person_name)
            else:
                print(f"⚠️ Không tìm thấy khuôn mặt trong: {img_path}")

    return known_embeddings, known_names

# Hàm lưu embeddings vào file .npz
def save_embeddings_to_file(known_embeddings, known_names, output_file="datasets/face_features/embeddings.npz"):
    if not known_embeddings:
        print("❌ Không có khuôn mặt nào để lưu. Kiểm tra lại ảnh đầu vào.")
        return

    embeddings_array = np.array(known_embeddings)
    names_array = np.array(known_names)
    
    # Lưu định dạng .npz
    np.savez(output_file, embeddings=embeddings_array, names=names_array)
    print(f"✅ Đã lưu {len(known_names)} embeddings vào {output_file}")

if __name__ == "__main__":
    print("🔄 Đang tạo embedding từ ảnh trong thư mục...")

    known_embeddings, known_names = load_and_encode_faces()
    save_embeddings_to_file(known_embeddings, known_names)
