import torch
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid


class CoraDataset:
    def __init__(self, root="../data/Cora"):
        # split="public": chuẩn semi-supervised của Kipf & Welling
        # (20 nhãn/lớp = 140 train, 500 val, 1000 test)
        # NormalizeFeatures: chuẩn hóa L1 theo hàng cho đặc trưng node X
        self.dataset = Planetoid(
            root=root,
            name="Cora",
            split="public",
            transform=T.NormalizeFeatures(),
        )
        self.data = self.dataset[0]

        # Trích xuất các thành phần dữ liệu chính
        self.features = self.data.x              # Ma trận đặc trưng X (2708 x 1433)
        self.labels = self.data.y                # Nhãn phân lớp Y (2708)
        # Danh sách cạnh dạng COO (2 x 10556): 10556 cạnh CÓ HƯỚNG = 2 x 5278 cạnh vô hướng.
        # (Cora gốc ghi 5429 cạnh; PyG đã loại cạnh trùng lặp và self-loop.)
        self.edge_index = self.data.edge_index

        # Các tập mask phân chia dữ liệu
        self.train_mask = self.data.train_mask
        self.val_mask = self.data.val_mask
        self.test_mask = self.data.test_mask

        # Kích thước chiều đặc trưng và số phân lớp
        self.num_features = self.dataset.num_node_features
        self.num_classes = self.dataset.num_classes

    def verify(self):
        """Kiểm tra dataset đúng cấu hình đề bài; báo lỗi (AssertionError) nếu sai."""
        assert self.data.num_nodes == 2708, "Số node khác 2708"
        assert self.num_features == 1433, "Số feature khác 1433"
        assert self.num_classes == 7, "Số lớp khác 7"
        assert self.train_mask.sum().item() == 140, "Train khác 140"
        assert self.val_mask.sum().item() == 500, "Val khác 500"
        assert self.test_mask.sum().item() == 1000, "Test khác 1000"

        per_class = torch.bincount(
            self.labels[self.train_mask], minlength=self.num_classes
        ).tolist()
        assert per_class == [20] * self.num_classes, f"Train không đủ 20 nhãn/lớp: {per_class}"

        assert not (self.train_mask & self.val_mask).any(), "Train/Val trùng nhau"
        assert not (self.train_mask & self.test_mask).any(), "Train/Test trùng nhau"
        assert not (self.val_mask & self.test_mask).any(), "Val/Test trùng nhau"

        row_sums = self.features.sum(dim=1)
        assert torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-3), \
            "X chưa chuẩn hóa L1 theo hàng"
        assert not torch.isnan(self.features).any(), "X có NaN"
        return per_class


if __name__ == "__main__":
    # Chạy trực tiếp: python dataset.py
    cora = CoraDataset()
    per_class = cora.verify()
    print("=== ĐÃ TẢI, CHUẨN HÓA VÀ KIỂM TRA DATASET CORA THÀNH CÔNG ===")
    print(f"• Kích thước Features (X):      {tuple(cora.features.shape)}")
    print(f"• Số phân lớp (Classes):        {cora.num_classes}")
    n_dir = cora.edge_index.shape[1]
    print(f"• Số cạnh có hướng (PyG):       {n_dir}  (= {n_dir // 2} cạnh vô hướng)")
    print(f"• Tập Train: {cora.train_mask.sum().item()} | "
          f"Val: {cora.val_mask.sum().item()} | Test: {cora.test_mask.sum().item()}")
    print(f"• Nhãn/lớp ở tập Train:         {per_class}")
