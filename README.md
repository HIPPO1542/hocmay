# Phân loại rau củ quả từ hình ảnh bằng các thuật toán học máy

Đồ án phân loại một ảnh vào một trong các lớp của bộ dữ liệu Fruits and Vegetables Image Recognition. HOG + SVM là mốc so sánh dùng đặc trưng thiết kế thủ công. CNN học đặc trưng RGB từ đầu, còn MobileNetV2 khởi tạo backbone bằng trọng số ImageNet. Cả ba dùng cùng danh sách nhãn và được đánh giá trên cùng tập test sau khi xử lý ảnh lỗi và ảnh trùng.

Nguồn ảnh là [Fruits and Vegetables Image Recognition Dataset trên Kaggle](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition). Số lớp, số ảnh và kích thước được đo từ dữ liệu đã tải. [REPORT_GUIDE.md](REPORT_GUIDE.md) hướng dẫn trình bày phương pháp, kết quả và giới hạn của đồ án.

## Cấu trúc

| Đường dẫn | Nội dung |
| --- | --- |
| `data/raw/` | Ảnh gốc đã giải nén, giữ cấu trúc train, validation và test |
| `data/processed/manifest.json` | Danh sách ảnh sạch, ánh xạ nhãn gián tiếp qua `class_names` và dấu vân tay dữ liệu |
| `src/dataset.py`, `src/eda.py` | Kiểm tra ảnh, phát hiện ảnh trùng và xuất biểu đồ PNG |
| `src/preprocessing.py`, `src/hog_features.py` | Đọc ảnh và tiền xử lý dùng chung khi train và suy luận |
| `src/train_*.py`, `src/model_artifacts.py` | Train ba mô hình, dùng chung vòng lặp Keras và định dạng metadata |
| `src/evaluate.py`, `src/compare_models.py`, `src/predict.py` | Đánh giá, xuất bảng so sánh và nhận diện ảnh |
| `notebooks/01_EDA.ipynb` đến `06_Model_Comparison.ipynb` | Sáu notebook chạy trực tiếp cùng các module CLI |
| `models/`, `results/` | Checkpoint, metadata, bảng số liệu, ảnh minh họa và dự đoán theo ảnh |
| `app.py`, `.streamlit/` | Demo hai trang, có theme và giới hạn tệp tải lên |
| `tests/`, `src/smoke.py` | Kiểm thử và chạy end-to-end với ảnh tổng hợp |

Mã đọc ảnh dùng Pillow. HOG dùng scikit-image, SVM dùng scikit-learn, hai mạng học sâu dùng TensorFlow/Keras. pandas và Matplotlib xuất bảng và hình, Streamlit dựng giao diện.

## Cài đặt trên Windows

Mở PowerShell ở thư mục chứa `requirements.txt`. Dùng Python 3.11 x64. Các lệnh gọi trực tiếp Python trong môi trường ảo để tránh yêu cầu thay đổi PowerShell Execution Policy.

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

Trong các phần bên dưới, `python` là Python của môi trường đã cài. Nếu chưa kích hoạt venv, thay `python` bằng `.\.venv\Scripts\python.exe`. Có thể mở notebook bằng VS Code với extension Jupyter và chọn interpreter trong venv.

## Tải dữ liệu

**Tải thủ công.** Trên trang Kaggle của bộ dữ liệu, chọn Download, rồi giải nén vào `data/raw/`. Có thể giữ một thư mục bao ngoài miễn là ba thư mục split nằm cạnh nhau. Dung lượng ZIP có thể vượt 2 GB, cần thêm chỗ cho ảnh đã giải nén.

**Kaggle API.** Chạy CLI trong môi trường đã xác thực. Dùng phiên đăng nhập của CLI, biến môi trường `KAGGLE_API_TOKEN` hoặc tệp thông tin xác thực Kaggle do bạn quản lý. Trên Windows khi dùng venv, có thể gọi `.\.venv\Scripts\kaggle.exe` nếu lệnh `kaggle` chưa có trong PATH.

```bash
python -m pip install kaggle
kaggle auth login
kaggle datasets download -d kritikseth/fruit-and-vegetable-image-recognition -p data/raw --unzip
```

Xem cách cài và xác thực ở [Kaggle CLI](https://github.com/Kaggle/kaggle-cli) và cú pháp ở [Datasets CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/datasets.md).

```text
data/raw/
  train/<tên lớp>/<ảnh>
  validation/<tên lớp>/<ảnh>
  test/<tên lớp>/<ảnh>
```

## Chuẩn bị dữ liệu, huấn luyện và đánh giá

```bash
python -m src.dataset
python -m src.train_svm
python -m src.train_cnn --epochs 30 --batch-size 32
python -m src.train_mobilenet --epochs 15 --batch-size 32 --fine-tune-epochs 5
python -m src.evaluate
python -m src.compare_models
```

- **EDA.** `dataset_summary.json` ghi số ảnh trước và sau lọc, lớp thiếu ảnh, nhóm ảnh trùng, kích thước ảnh và tỷ số lớp train lớn nhất trên lớp nhỏ nhất. `dataset_images.csv` ghi trạng thái từng tệp. `dataset_class_counts.csv` có số ảnh mỗi lớp trước và sau lọc. Xem bốn ảnh `eda_*.png` trong `results/figures/`.
- **Chia tập.** Bước kiểm tra nhận diện ảnh giống hệt bằng SHA-256 trên kích thước và pixel RGB sau hiệu chỉnh EXIF. Khi cùng nhãn, một bản được giữ theo ưu tiên test, validation, rồi train. Khi cùng ảnh nhưng khác nhãn, cả nhóm bị loại để kiểm tra lại. Nếu validation vẫn có đủ lớp, giữ cách chia tập đã lọc. Nếu validation thiếu lớp, giữ các ảnh validation sạch, rồi lấy thêm từ train sạch theo từng lớp với seed 42. Với mỗi lớp, mục tiêu validation bằng khoảng 10% số ảnh train sạch của lớp trước khi bổ sung, đồng thời giữ ít nhất một ảnh cho train. Tập test sạch giữ nguyên. Việc bổ sung chỉ thay split trong manifest và được lưu trong `dataset_validation_repair.csv`. Ảnh gần giống sau crop, resize hoặc nén lại có thể lọt qua kiểm tra chính xác này.
- **HOG + SVM.** Resize grayscale 64×64 và trích xuất HOG. Pipeline fit StandardScaler và SVC trên train. Chọn Linear hoặc RBF và siêu tham số bằng Macro F1 trên validation, dùng Accuracy để phá hòa. Mặc định hiệu chỉnh xác suất bằng CV nội bộ trên train sau khi chọn siêu tham số. Thêm `--no-probability` để bỏ bước hiệu chỉnh. Nhãn chính lấy từ `predict()`, thứ hạng xác suất từ `predict_proba()` khi có.
- **CNN.** Resize RGB 128×128. Kiến trúc ba khối Conv2D và MaxPooling, GlobalAveragePooling, Dense và Dropout. Checkpoint chứa lớp chuẩn hóa pixel về [0, 1].
- **MobileNetV2.** Resize RGB 224×224. Checkpoint ánh xạ pixel về [−1, 1], tương đương `preprocess_input`. Head train với backbone đóng băng, tùy chọn tinh chỉnh các lớp cuối ở learning rate `1e-5`. BatchNormalization giữ thống kê ImageNet trong fine-tuning. Thêm `--fine-tune-epochs 0` để chỉ train head.
- **Cả hai mạng học sâu.** Flip ngang, xoay, zoom và chỉnh tương phản chỉ bật khi train. EarlyStopping, ModelCheckpoint và ReduceLROnPlateau theo dõi loss validation. Số epoch mặc định của CNN là 30, có thể giảm khi cần chạy nhanh. Mỗi mô hình đi kèm JSON ghi danh sách lớp, preprocess, thông số huấn luyện và checksum checkpoint. Giữ hai tệp đi cùng nhau nếu chuyển máy.
- **Test.** Chạy sau khi chốt mô hình. Kết quả gồm Accuracy, Macro Precision, Macro Recall, Macro F1, classification report, confusion matrix, danh sách ảnh sai và các cặp dễ nhầm có hướng. Macro dùng toàn bộ danh sách lớp với `zero_division=0`. Ba mô hình phải được đánh giá trong cùng phiên để so sánh. Có thể thử riêng bằng `python -m src.evaluate --models svm`. Khi đủ ba checkpoint, chạy đánh giá lại cả ba trong một lệnh.
- **Thời gian.** Benchmark test dùng CPU, batch 1, tối đa ba ảnh warm-up từ validation. Thời gian mỗi ảnh gồm đọc, giải mã, tiền xử lý và dự đoán với model đã nạp. Bảng ghi trung bình và JSON từng mô hình có thêm P95. Thời gian train của SVM gồm trích HOG, grid search và hiệu chỉnh nếu bật. Với mạng học sâu, đồng hồ tính từ lúc fit sau khi đã dựng mô hình và tải trọng số, gồm cả fine-tuning nếu bật. Kích thước là dung lượng tệp lưu thực tế.

Nếu không có mạng để tải trọng số, tải [tệp ImageNet MobileNetV2 no-top alpha 1.0, 224 chính thức](https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_224_no_top.h5) từ máy có mạng rồi truyền đường dẫn cho `--weights`. CLI dừng và báo lỗi nếu không nạp được.

## Chạy trên Google Colab

1. Chép dự án lên Drive hoặc tải một ZIP của mã nguồn. Giải nén hoặc chép dự án vào ổ cục bộ của phiên Colab, chẳng hạn `/content/FruitVegetableClassification`. Với dataset nhiều tệp, đọc từ ổ cục bộ giúp giảm thời gian I/O.
2. Chọn runtime GPU trong **Runtime → Change runtime type**. Thử `!python --version`. Có thể chọn runtime cũ dùng Python 3.11 nếu Colab còn cung cấp. Các dependency chính cũng có bản cho Python 3.12.
3. Dùng `%cd /content/FruitVegetableClassification` và `%pip install -r requirements.txt`. Khởi động lại runtime nếu Colab yêu cầu do thư viện đã được nạp. Kiểm tra GPU bằng `python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"`.
4. Tải dữ liệu bằng lệnh Kaggle ở trên nếu đã có token. Với Colab Secrets, lưu token ở secret `KAGGLE_API_TOKEN`, cấp quyền truy cập cho notebook rồi gán từ `google.colab.userdata.get("KAGGLE_API_TOKEN")` vào biến môi trường cùng tên. Có thể thay bằng ZIP đã giải nén đúng vị trí. Không in token trong notebook.
5. Chạy các lệnh bằng tiền tố `!` hoặc chạy sáu notebook theo thứ tự. Dùng `--batch-size 16` hoặc `8` nếu bộ nhớ GPU hạn chế. Lưu `models/`, `results/` và `data/processed/manifest.json` sang Drive trước khi phiên Colab kết thúc.

Tham khảo [Colab Runtime Versions](https://research.google.com/colaboratory/runtime-version-faq.html) và [Hướng dẫn cài TensorFlow](https://www.tensorflow.org/install/pip).

## Chạy demo, kiểm thử và đọc kết quả

```bash
python -m streamlit run app.py
python -m pytest -q
python -m src.smoke
python -m src.smoke --include-mobilenet
```

1. Vào **Nhận diện**, chọn một checkpoint và tải JPEG/PNG. Ứng dụng hiển thị nhãn dự đoán, tối đa ba lớp theo xác suất khi có, biểu đồ và thời gian xử lý.
2. Vào **So sánh mô hình** để xem `model_comparison.csv`, Accuracy và Macro F1, khuyến nghị theo dữ liệu đo, ma trận nhầm lẫn và ảnh dự đoán sai.
3. Để thử bằng CLI, chạy `python -m src.predict "duong-dan-anh.jpg" --model mobilenet`.

Smoke test mặc định dùng SVM và CNN một epoch trên ảnh hình học tổng hợp. Tùy chọn MobileNetV2 chạy một epoch head và một epoch fine-tune với ImageNet. Từng phiên tạo dữ liệu, checkpoint và báo cáo riêng dưới `.smoke/`. Muốn kiểm tra giao diện với các artifact này, đặt biến môi trường `FRUIT_VEG_WORKSPACE` trỏ đến thư mục `.smoke/run-...` do lệnh in ra. Xóa biến môi trường sau đó để trở lại dataset Kaggle. Giao diện hiển thị cảnh báo khi mở dữ liệu tổng hợp.

| Tình huống | Cách xử lý |
| --- | --- |
| Thiếu manifest hoặc tìm thấy nhiều bộ split | Kiểm tra `data/raw/` rồi chạy lại `python -m src.dataset`. |
| Ảnh đã thay đổi sau EDA | Chạy lại EDA. Huấn luyện lại nếu manifest đổi dấu vân tay. |
| Chưa có model hoặc model lệch metadata | Huấn luyện hoặc sao chép lại đủ checkpoint và JSON cùng lần chạy. |
| Thiếu metrics hoặc kết quả từ các checkpoint cũ | Chạy lại đánh giá chung rồi tạo lại bảng so sánh. |
| TensorFlow không thấy GPU trên Windows | Demo chạy CPU trực tiếp. Dùng Colab GPU hoặc WSL2 cho huấn luyện trên GPU. |
| Thiếu RAM/VRAM | Giảm batch size. Trên Colab, chỉnh loại phần cứng nếu cần. |
| Không nạp được ImageNet | Kiểm tra mạng hoặc dùng tùy chọn `--weights` với tệp chính thức. |
