import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    """MLP 2 lớp baseline: chỉ dùng đặc trưng node X, KHÔNG dùng ma trận kề A.

    Trả về log_softmax => phải dùng cùng F.nll_loss
    (không dùng CrossEntropyLoss, vì sẽ áp softmax hai lần).
    """

    def __init__(self, in_features, hidden_dim=16, num_classes=7,
                 dropout=0.5, input_dropout=True):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        self.dropout = dropout
        # GCN gốc (Kipf & Welling) áp dropout cả ở input; để True cho so sánh công bằng.
        # Nếu GCN của nhóm không có dropout input thì đặt False.
        self.input_dropout = input_dropout

    def forward(self, x):                      # KHÔNG nhận edge_index
        if self.input_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        # Lớp Linear 1 + ReLU
        x = F.relu(self.fc1(x))
        # Dropout giảm overfitting trong điều kiện tập Train nhỏ (140 nodes)
        x = F.dropout(x, p=self.dropout, training=self.training)
        # Lớp Linear 2
        x = self.fc2(x)
        # Log Softmax tương thích với NLLLoss
        return F.log_softmax(x, dim=1)
