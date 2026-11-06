"""Huấn luyện và đánh giá MLP baseline trên Cora.

Chạy: python train_mlp.py [--root ../data/Cora] [--seeds 10]
- Chọn model theo Validation accuracy; Test chỉ được đọc tại epoch Val tốt nhất.
- Báo cáo mean ± std qua nhiều seed.
"""
import argparse
import random

import numpy as np
import torch
import torch.nn.functional as F

from dataset import CoraDataset
from mlp_model import MLP


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def accuracy(model, x, y, mask):
    model.eval()
    pred = model(x).argmax(dim=1)
    return (pred[mask] == y[mask]).float().mean().item()


def run(cora, seed, hidden, dropout, lr, weight_decay, epochs, device, verbose=False):
    set_seed(seed)
    x = cora.features.to(device)                 # chỉ dùng X, không dùng edge_index
    y = cora.labels.to(device)
    train_mask = cora.train_mask.to(device)
    val_mask = cora.val_mask.to(device)
    test_mask = cora.test_mask.to(device)

    model = MLP(cora.num_features, hidden, cora.num_classes, dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val, test_at_best_val, best_epoch = 0.0, 0.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = F.nll_loss(model(x)[train_mask], y[train_mask])
        loss.backward()
        optimizer.step()

        val_acc = accuracy(model, x, y, val_mask)
        test_acc = accuracy(model, x, y, test_mask)

        # Chọn theo Validation: chỉ ghi lại test tại epoch có val tốt nhất
        if val_acc > best_val:
            best_val, test_at_best_val, best_epoch = val_acc, test_acc, epoch

        if verbose and (epoch == 1 or epoch % 20 == 0):
            print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | "
                  f"Val {val_acc:.4f} | Test {test_acc:.4f}")

    return best_val, test_at_best_val, best_epoch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="../data/Cora")
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--hidden", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--epochs", type=int, default=200)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cora = CoraDataset(root=args.root)
    cora.verify()
    print(f"Device: {device} | Dataset Cora đã kiểm tra đúng cấu hình 140/500/1000")

    results = []
    for seed in range(args.seeds):
        v, t, e = run(cora, seed, args.hidden, args.dropout, args.lr,
                      args.weight_decay, args.epochs, device,
                      verbose=(seed == 0))
        results.append((v * 100, t * 100))
        print(f"Seed {seed}: Val {v*100:.2f}% (epoch {e}) | Test tại epoch Val tốt nhất {t*100:.2f}%")

    val_accs, test_accs = np.array(results).T
    print("=" * 60)
    print(f"MLP Baseline ({args.seeds} seeds)")
    print(f"  Val  Acc: {val_accs.mean():.2f} ± {val_accs.std():.2f} %")
    print(f"  Test Acc (tại epoch Validation tốt nhất): {test_accs.mean():.2f} ± {test_accs.std():.2f} %")
    print("=" * 60)


if __name__ == "__main__":
    main()
