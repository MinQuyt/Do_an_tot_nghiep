# Do_an_tot_nghiep
1: Cài đặt môi trường ảo Conda và cài đặt các thư viện cần thiết
-  Download Miniconda: https://www.anaconda.com/docs/getting-started/miniconda/install
-  Mở màn hình CommandLine với quyền admin để tạo các môi trường ảo
  ```shell
   conda create -n quy_face4 -c conda-forge onnxruntime python=3.9 //Tạo môi trường ảo, python phiên bản 3.9, và cài onnxruntime để có thể sử dụng GPU để huấn luyện models
   ```
Download thư viện insightface từ trên github về máy https://github.com/cobanov/insightface_windows/tree/main/whls

  ```shell
   pip install C:\Users\Admin\Downloads\insightface_windows-main\whls\insightface-0.7.3-cp39-cp39-win_amd64.whl // chạy lệnh để cài thư viện vào môi trường ảo
   ```
      
