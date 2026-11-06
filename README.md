# BÁO CÁO ĐỒ ÁN: ĐÁNH GIÁ & THỰC THI GCN TRÊN TẬP DỮ LIỆU CORA

Thư mục chứa mã nguồn mở rộng và triển khai mô hình **Graph Convolutional Network (GCN)** dựa trên bài báo *"Semi-Supervised Classification with Graph Convolutional Networks"* (Kipf & Welling, 2017).

---

## 📁 Cấu Trúc Thư Mục Source Code

```plaintext
├── data/               # Chứa dữ liệu Cora (cora.cites, cora.content)
├── dataset.py          # Tiền xử lý dữ liệu, chuẩn hóa A_hat và phân chia tập dữ liệu
├── models.py           # Định nghĩa kiến trúc mô hình MLP Baseline và GCN
├── train.py            # Huấn luyện mô hình, đánh giá loss/accuracy, lưu checkpoint
├── visualize.py        # Vẽ biểu đồ Accuracy/Loss, biểu diễn t-SNE và phân tích Over-smoothing
├── requirements.txt    # Các thư viện phụ thuộc cần thiết
└── README.md           # Hướng dẫn chạy chương trình & Phân công công việc
```

---

## 👥 Phân Công Công Việc (4 Thành Viên)

| Thành viên | Trách nhiệm chính | Chi tiết công việc thực hiện | Output chính |
| :--- | :--- | :--- | :--- |
| **Thành viên 1** | **Data Engineering & Preprocessing** | • Thu thập và nạp dữ liệu Cora (`data/`).<br>• Xây dựng file `dataset.py` để xử lý ma trận kề $A$, thêm self-loops ($\tilde{A} = A + I_N$).<br>• Cài đặt công thức Renormalization Trick ($\hat{A} = \tilde{D}^{-1/2} \tilde{A} \tilde{D}^{-1/2}$).<br>• Phân chia tập Train/Validation/Test theo chuẩn đề bài. | `dataset.py`, dữ liệu trong `data/` |
| **Thành viên 2** | **Model Architecture & Design** | • Xây dựng module `models.py`.<br>• Thiết kế mô hình **MLP Baseline** (không tích hợp thông tin ma trận kề).<br>• Triển khai lớp **GCN Layer** và cấu trúc **GCN 2 lớp** chuẩn công thức Kipf & Welling ($Z = \text{softmax}(\hat{A} \text{ReLU}(\hat{A} X W^{(0)}) W^{(1)})$). | `models.py` |
| **Thành viên 3** | **Training & Evaluation Pipeline** | • Xây dựng luồng huấn luyện toàn cục trong `train.py`.<br>• Thiết lập Optimizer (Adam), Loss function (Cross-Entropy).<br>• Cài đặt cơ chế **Early Stopping** và kỹ thuật **Dropout/L2 Regularization**.<br>• Quản lý lưu giữweights/checkpoints kết quả huấn luyện tốt nhất. | `train.py`, file checkpoint mô hình |
| **Thành viên 4** | **Visualization & Empirical Analysis** | • Xây dựng các hàm vẽ đồ thị trong `visualize.py`.<br>• Trực quan hóa tiến trình huấn luyện (Loss & Accuracy curves).<br>• Biểu diễn biểu đồ **t-SNE** phân bố nhãn ẩn (Hidden Layer Activations).<br>• Thực nghiệm và trực quan hiện tượng **Over-smoothing** khi thay đổi độ sâu số lớp GCN. | `visualize.py`, hình ảnh biểu đồ kết quả |

---

## 🛠️ Cài Đặt Môi Trường

1. **Khởi tạo và kích hoạt môi trường ảo (Khuyên dùng):**
   ```bash
   python -m venv venv
   # Trên Linux/macOS:
   source venv/bin/activate
   # Trên Windows:
   venv\Scripts\activate
   ```

2. **Cài đặt các thư viện phụ thuộc:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 🚀 Hướng Dẫn Chạy Chương Trình

### 1. Huấn luyện mô hình
Chạy script `train.py` để bắt đầu quá trình huấn luyện MLP Baseline và GCN Model trên tập dữ liệu Cora:
```bash
python train.py --model gcn --epochs 200 --lr 0.01
```

### 2. Trực quan hóa kết quả
Sau khi huấn luyện xong, chạy script `visualize.py` để xuất các biểu đồ đánh giá:
```bash
python visualize.py
```
*Kết quả đồ thị bao gồm: đường cong Loss/Accuracy, biểu đồ không gian biểu diễn ẩn t-SNE, và biểu đồ so sánh Over-smoothing theo độ sâu của GCN.*