# Đề cương báo cáo đồ án

## Chương 1 — Tổng quan đề tài

Mô tả bài toán phân loại ảnh đơn nhãn và mục tiêu tạo demo nhận diện. Đầu vào là ảnh RGB, đầu ra là một lớp thuộc danh sách học được từ tập train. Xây dựng ba mô hình để đo sự khác nhau giữa đặc trưng thủ công, mạng học từ đầu và mạng dùng trọng số có sẵn. So sánh cả chất lượng phân loại và chi phí triển khai.

## Chương 2 — Cơ sở lý thuyết

**HOG (Histogram of Oriented Gradients).** Với ảnh xám, tính gradient ngang $g_x$, dọc $g_y$, độ lớn $m$ và hướng $\theta$.

```math
m = \sqrt{g_x^2 + g_y^2}, \qquad \theta = \mathrm{atan2}(g_y, g_x).
```

Cộng phiếu có trọng số theo độ lớn gradient vào histogram hướng trong từng ô. Ghép các histogram trong mỗi khối và chuẩn hóa L2-Hys (chuẩn hóa L2, cắt giá trị lớn và chuẩn hóa lại). Với ảnh 64×64, ô 8×8, khối 2×2 ô và 9 bin, vector dài $7 \times 7 \times 2 \times 2 \times 9 = 1764$. [Dalal và Triggs, 2005](https://doi.org/10.1109/CVPR.2005.177).

**SVM.** Với nhãn nhị phân $y_i \in \{-1, 1\}$, soft-margin SVM tìm siêu phẳng có biên lớn, cho phép một số điểm vi phạm biên. Tham số $C$ quy định mức phạt.

```math
\min_{w,b,\xi}\; \frac{1}{2}\lVert w\rVert_2^2 + C\sum_i \xi_i,
\quad
y_i(w^\top\phi(x_i)+b) \ge 1-\xi_i, \qquad \xi_i \ge 0.
```

Kernel Linear dùng $K(x,x')=x^\top x'$. Kernel RBF dùng $K(x,x')=\exp(-\gamma\lVert x-x'\rVert_2^2)$, trong đó $\gamma$ điều khiển mức suy giảm tương đồng theo khoảng cách. SVC giải bài toán nhiều lớp bằng one-vs-one. [Cortes và Vapnik, 1995](https://doi.org/10.1007/BF00994018).

**CNN.** Mỗi bộ lọc kết hợp các giá trị trong một vùng ảnh cục bộ và áp dụng ReLU. Phép tính dùng chung trọng số ở mọi vị trí. MaxPooling giảm kích thước không gian, GlobalAveragePooling lấy trung bình từng kênh, Softmax chuyển logits $z$ thành phân phối $p$.

```math
a_{ij,k} = \max\left(0,b_k+\sum_{u,v,c} W_{uvck} X_{i+u,j+v,c}\right),
\qquad p_k = \frac{e^{z_k}}{\sum_j e^{z_j}}.
```

Sparse categorical cross-entropy dùng chỉ số nhãn thực $y_i$ và tối thiểu hóa $L=-\frac{1}{N}\sum_i \log p_{i,y_i}$. Dự án cập nhật trọng số bằng Adam. [LeCun và cộng sự, 1998](https://doi.org/10.1109/5.726791).

**Transfer learning.** Khởi tạo bộ trích đặc trưng $f_\theta$ bằng ImageNet và học đầu phân loại $g_\psi$ cho các lớp rau củ quả.

```math
\hat p(x) = \mathrm{softmax}\big(g_\psi(f_\theta(x))\big),
\qquad \hat\psi = \arg\min_\psi \mathcal{L}_{\mathrm{train}}(\theta_{\mathrm{ImageNet}}, \psi).
```

Khi fine-tune, cho phép một phần cuối của $\theta$ thay đổi với learning rate nhỏ. MobileNetV2 dùng depthwise convolution, inverted residual và linear bottleneck để giảm chi phí tính toán. Trình bày lý do giữ BatchNormalization ở chế độ suy luận khi fine-tune. [Sandler và cộng sự, 2018](https://openaccess.thecvf.com/content_cvpr_2018/html/Sandler_MobileNetV2_Inverted_Residuals_CVPR_2018_paper.html).

## Chương 3 — Phân tích và tiền xử lý dữ liệu

- Nêu nguồn, phiên bản archive, cấu trúc train/validation/test và số lớp rút ra từ `dataset_summary.json`.
- Dùng số ảnh trước và sau lọc từ `dataset_class_counts.csv`. Báo số ảnh lỗi, nhóm trùng nội tập và liên tập, xung đột nhãn, số ảnh còn lại và tỷ số mất cân bằng. Giải thích thứ tự ưu tiên khi loại ảnh trùng và giới hạn của so khớp chính xác pixel.
- Nếu validation cần bổ sung, ghi số ảnh và lớp thiếu sau khử trùng trước khi bổ sung. Trình bày quy tắc lấy mẫu từ train sạch, seed 42, kích thước validation mới, danh sách chuyển trong `dataset_validation_repair.csv` và cách giữ nguyên test đã làm sạch.
- Chèn `eda_class_counts.png`, `eda_splits.png`, `eda_samples.png` và `eda_dimensions.png`. Mô tả điều kiện chụp, nền, góc nhìn và phân bố kích thước dựa trên hình quan sát được.
- Mô tả HOG trên ảnh grayscale 64×64, CNN trên RGB 128×128 chuẩn hóa [0, 1] và MobileNetV2 trên RGB 224×224 chuẩn hóa [−1, 1]. Trình bày mapping nhãn, seed 42 và augmentation chỉ áp dụng trong train.

Trong lần chạy Windows ngày 19/11/2026, EDA đọc 3.825 ảnh thuộc 36 lớp. Có 357 nhóm trùng pixel, trong đó 334 nhóm xuất hiện ở nhiều split. Bước lọc loại 874 bản trùng cùng nhãn và 55 ảnh thuộc 17 nhóm xung đột nhãn. Cặp nhãn `corn` và `sweetcorn` cùng xuất hiện trong 11 nhóm xung đột. Validation gốc chỉ còn 2 ảnh sau lọc, nên manifest lấy thêm 253 ảnh từ train sạch. Ba tập thực nghiệm có 2.313, 255 và 328 ảnh, đều phủ đủ 36 lớp. Số ảnh train mỗi lớp nằm trong khoảng 50–77.

## Chương 4 — Xây dựng và huấn luyện mô hình

1. **SVM.** Dùng bảng `svm_validation_search.csv` để trình bày grid cho Linear, RBF, $C$ và $\gamma$. Giải thích scaler chỉ fit trên train và cách chọn theo Macro F1 validation.
2. **CNN.** Vẽ sơ đồ Input → 3 khối Conv/Pool → GlobalAveragePooling → Dense/Dropout → Softmax. Nêu số epoch thực chạy, batch size, Adam và tiêu chí chọn checkpoint.
3. **MobileNetV2.** Vẽ hai giai đoạn đóng băng backbone và fine-tune. Báo nguồn trọng số, số lớp được mở, learning rate, checkpoint được chọn và số epoch thực chạy.
4. **Đồ thị học.** Chèn `cnn_training.png` và `mobilenet_training.png`. Nếu train loss giảm còn validation loss tăng kéo dài, mô hình có thể overfit. Nếu hiệu suất cả hai tập đều thấp và không còn cải thiện, kiểm tra giới hạn năng lực mô hình, thời gian học và chất lượng dữ liệu trước khi kết luận underfit.

Trong lần chạy Windows, SVM chọn RBF với `C=10` và `gamma=scale`. CNN chạy 5 epoch, train loss giảm từ 3,535 xuống 2,906 và validation loss giảm từ 3,406 xuống 2,755. Accuracy ở cả hai tập còn thấp và loss vẫn giảm ở epoch cuối, cho thấy CNN còn học trong cấu hình này. Dữ liệu 5 epoch chưa đủ để kết luận kiến trúc bị underfit. MobileNetV2 chạy 3 epoch head và 1 epoch fine-tune. Checkpoint fine-tune được chọn theo validation loss 0,693, so với 0,798 của checkpoint head tốt nhất.

## Chương 5 — Kết quả thực nghiệm và đánh giá

Gọi $M_{ij}$ là số ảnh thực thuộc lớp $i$ và được dự đoán là $j$. Đường chéo của confusion matrix biểu thị dự đoán đúng. Với lớp $k$, $TP_k=M_{kk}$, $FP_k=\sum_{i\ne k}M_{ik}$, $FN_k=\sum_{j\ne k}M_{kj}$.

```math
\mathrm{Accuracy} = \frac{\sum_k M_{kk}}{\sum_{i,j}M_{ij}},
\qquad
P_k = \frac{TP_k}{TP_k+FP_k},
\qquad
R_k = \frac{TP_k}{TP_k+FN_k},
\qquad
F1_k = \frac{2P_kR_k}{P_k+R_k}.
```

Macro Precision, Macro Recall và Macro F1 là trung bình số học của chỉ số từng lớp. Macro F1 lấy trung bình $F1_k$ và mỗi lớp có trọng số bằng nhau. Quy ước trong mã là 0 khi mẫu số bằng 0.

```math
\mathrm{Macro\text{-}F1} = \frac{1}{K}\sum_{k=1}^{K}F1_k.
```

- Dùng `model_comparison.csv` để chèn bảng Model, Accuracy, Macro Precision, Macro Recall, Macro F1, Train Time, Inference Time và Model Size. Ghi số ảnh test hợp lệ, cấu hình CPU và quy trình đo. Thời gian huấn luyện chỉ có thể so sánh trực tiếp khi phần cứng và cấu hình phù hợp.
- Đọc `*_classification_report.json` để tìm lớp có recall thấp. Chọn các cặp đứng đầu trong `*_confusion_pairs.csv` và đối chiếu ảnh từ `*_mistakes.png` hoặc `*_mistakes.csv`. Với số đếm lớn, kiểm tra thêm tỷ lệ trên tổng ảnh thực của lớp để tránh kết luận từ chênh lệch support.
- Ghi ưu nhược điểm gắn với kết quả. HOG mất thông tin màu nhưng dễ giải thích, CNN học màu và texture nhưng cần dữ liệu đa dạng, MobileNetV2 tận dụng đặc trưng có sẵn nhưng phụ thuộc mức phù hợp giữa ImageNet và dữ liệu đích.
- Đề xuất mô hình cho demo dựa trên Macro F1, thời gian CPU và dung lượng. Nếu số epoch ít hoặc chưa có đủ kết quả, ghi đúng ngân sách đã chạy hoặc trạng thái chưa thực hiện. Ghi rõ nếu đo bằng smoke test và chỉ sử dụng phép đo đó để xác nhận pipeline hoạt động.

### Nhận xét từ lần đo trên Windows

MobileNetV2 đúng 274/328 ảnh, đạt Accuracy 83,54% và Macro F1 82,07%. Đây là lựa chọn cho demo khi ưu tiên chất lượng. Độ trễ trung bình của mô hình là 143,26 ms/ảnh, gồm đọc và giải mã ảnh. HOG + SVM có độ trễ thấp nhất trong lần đo này (38,71 ms/ảnh). CNN có tệp nhỏ nhất (1,37 MiB) và accuracy cao hơn HOG + SVM, nhưng Macro F1 thấp hơn do hiệu quả giữa các lớp chưa đồng đều. Xem bảng đầy đủ trong [model_comparison.csv](results/metrics/model_comparison.csv).

- **HOG + SVM** dự đoán 4/9 ảnh `cucumber` thành `carrot`. Ảnh minh họa có nhiều quả dài nằm chồng lên nhau. HOG ảnh xám giữ đường biên và hướng cạnh, nên thiếu màu xanh để phân biệt với cà rốt.
- **CNN** dự đoán 9/10 ảnh `pineapple` thành `garlic` và cả 8 ảnh `sweetcorn` thành `lemon`. Các lỗi tập trung theo cặp này cho thấy Macro F1 là chỉ số cần đọc cùng Accuracy và số mẫu thực của mỗi lớp.
- **MobileNetV2** dự đoán 5/6 ảnh `corn` thành `sweetcorn`. Ảnh minh họa là một bắp ngô vàng còn bẹ và EDA đã phát hiện ảnh giống hệt có cả hai nhãn. Cặp `bell pepper` và `capsicum` cũng bị nhầm theo hai chiều, mỗi chiều 2 ảnh. Gallery còn có logo Apple và ảnh chuối hoạt hình trong tập test, cho thấy nhãn thư mục đôi khi đi kèm hình ngoài phạm vi ảnh thực phẩm. Xem [gallery dự đoán sai](results/figures/mobilenet_mistakes.png).

Lần đo dùng một seed và ngân sách huấn luyện CPU rút gọn. Ảnh gần trùng và dữ liệu ngoài Kaggle cần các kiểm tra riêng khi diễn giải khả năng khái quát hóa.

## Chương 6 — Kết luận và hướng phát triển

Tóm tắt những gì đã đo được trên test và khả năng triển khai kèm ví dụ demo. Nêu giới hạn như ảnh gần trùng chưa phát hiện, ảnh nhiều loại rau củ cùng khung, ảnh ngoài danh sách lớp và khả năng suy rộng từ ảnh Kaggle sang điều kiện thực tế. Có thể đề xuất một tập ảnh độc lập để kiểm chứng sau khi lựa chọn mô hình.
