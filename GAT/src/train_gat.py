"""
Faz 5.7-5.8: GAT egitim ve validation.

Dataset'i yukle, modeli egit, validation accuracy olc.

Egitim dongusu:
  Her epoch:
    1) Her sample icin:
       - Graph'i model input formatina cevir
       - Forward pass: logits_ro, logits_rr = model(...)
       - Loss: BCEWithLogitsLoss(logits, labels)
       - Backward pass: gradients hesapla
       - Optimizer step: agirliklari guncelle
    2) Validation accuracy olc
    3) Sonuclari yazdir

Neden BCEWithLogitsLoss?
  - Binary cross-entropy: her binary icin P(b=1) tahmini ile gercek degeri karsilastir
  - "WithLogits": sigmoid icerde uygulanir, numerik olarak daha stabil
  - Loss = -[y*log(sigmoid(x)) + (1-y)*log(1-sigmoid(x))]
"""

import os
import pickle
import time
import numpy as np
import torch
import torch.nn as nn
from gat_model import GATBinaryPredictor


def load_dataset(path):
    """Pickle dataset yukle."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def graph_to_model_input(graph, device='cpu'):
    """Dataset'teki graph dict'ini model input formatina cevir.

    Onemli: Dataset'te obstacle edge index'leri 0..NO-1 arasi.
    Model'de ise node indexleme: robot 0..NR-1, obstacle NR..NR+NO-1.
    Bu yuzden obstacle index'lerine NR offset ekliyoruz.
    """
    NR = graph['node_feat_robot'].shape[0]
    NO = graph['node_feat_obstacle'].shape[0]
    H = graph['H']

    x_robot = torch.tensor(graph['node_feat_robot'], dtype=torch.float32, device=device)
    x_obstacle = torch.tensor(graph['node_feat_obstacle'], dtype=torch.float32, device=device)

    # --- RO edge'leri: robot -> obstacle ---
    # Dataset'te: src = robot index (0..NR-1), dst = obstacle index (0..NO-1)
    # Model'de:   src = robot index (0..NR-1), dst = obstacle index (NR..NR+NO-1)
    ro_src = graph['edge_index_RO'][0]  # robot indices, zaten dogru
    ro_dst = graph['edge_index_RO'][1] + NR  # obstacle'a NR offset ekle
    edge_index_ro = torch.tensor(np.array([ro_src, ro_dst]), dtype=torch.long, device=device)

    # --- RR edge'leri: robot -> robot ---
    # Index'ler zaten 0..NR-1, offset gerekmez
    edge_index_rr = torch.tensor(graph['edge_index_RR'], dtype=torch.long, device=device)

    # --- OR edge'leri: obstacle -> robot (bilgi akisi) ---
    or_src = graph['edge_index_OR'][0] + NR  # obstacle offset
    or_dst = graph['edge_index_OR'][1]  # robot, offset yok
    edge_index_or = torch.tensor(np.array([or_src, or_dst]), dtype=torch.long, device=device)

    # --- OO edge'leri: obstacle -> obstacle ---
    if graph['edge_index_OO'].shape[1] > 0:
        oo_src = graph['edge_index_OO'][0] + NR
        oo_dst = graph['edge_index_OO'][1] + NR
        edge_index_oo = torch.tensor(np.array([oo_src, oo_dst]), dtype=torch.long, device=device)
    else:
        edge_index_oo = torch.zeros((2, 0), dtype=torch.long, device=device)

    # Tum edge'leri birlestir (encoder icin)
    edge_index_all = torch.cat([edge_index_ro, edge_index_rr,
                                 edge_index_or, edge_index_oo], dim=1)

    # --- Labels ---
    labels_ro = torch.tensor(graph['edge_labels_RO'], dtype=torch.float32, device=device)
    labels_rr = torch.tensor(graph['edge_labels_RR'], dtype=torch.float32, device=device)

    return {
        'x_robot': x_robot,
        'x_obstacle': x_obstacle,
        'edge_index_all': edge_index_all,
        'edge_index_ro': edge_index_ro,
        'edge_index_rr': edge_index_rr,
        'labels_ro': labels_ro,
        'labels_rr': labels_rr,
    }


def compute_accuracy(logits, labels):
    """Binary accuracy: sigmoid(logit) > 0.5 → predicted 1, else 0."""
    if labels.numel() == 0:
        return float('nan'), 0
    preds = (torch.sigmoid(logits) > 0.5).float()
    correct = (preds == labels).sum().item()
    total = labels.numel()
    return correct / total, total


def train_one_epoch(model, dataset, optimizer, criterion, device='cpu'):
    """Bir epoch egitim."""
    model.train()
    total_loss = 0
    ro_correct, ro_total = 0, 0
    rr_correct, rr_total = 0, 0

    for graph in dataset:
        inp = graph_to_model_input(graph, device)
        optimizer.zero_grad()

        logits_ro, logits_rr = model(
            inp['x_robot'], inp['x_obstacle'], inp['edge_index_all'],
            inp['edge_index_ro'], inp['edge_index_rr']
        )

        # Loss: RO + RR
        loss_ro = criterion(logits_ro, inp['labels_ro'])
        if inp['labels_rr'].numel() > 0:
            loss_rr = criterion(logits_rr, inp['labels_rr'])
            loss = loss_ro + loss_rr
        else:
            loss = loss_ro

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        # Accuracy
        acc_ro, n_ro = compute_accuracy(logits_ro, inp['labels_ro'])
        ro_correct += acc_ro * n_ro
        ro_total += n_ro

        if inp['labels_rr'].numel() > 0:
            acc_rr, n_rr = compute_accuracy(logits_rr, inp['labels_rr'])
            rr_correct += acc_rr * n_rr
            rr_total += n_rr

    avg_loss = total_loss / len(dataset)
    ro_acc = ro_correct / max(ro_total, 1)
    rr_acc = rr_correct / max(rr_total, 1) if rr_total > 0 else float('nan')

    return avg_loss, ro_acc, rr_acc


@torch.no_grad()
def evaluate(model, dataset, criterion, device='cpu'):
    """Validation seti uzerinde evaluation."""
    model.eval()
    total_loss = 0
    ro_correct, ro_total = 0, 0
    rr_correct, rr_total = 0, 0

    for graph in dataset:
        inp = graph_to_model_input(graph, device)

        logits_ro, logits_rr = model(
            inp['x_robot'], inp['x_obstacle'], inp['edge_index_all'],
            inp['edge_index_ro'], inp['edge_index_rr']
        )

        loss_ro = criterion(logits_ro, inp['labels_ro'])
        if inp['labels_rr'].numel() > 0:
            loss_rr = criterion(logits_rr, inp['labels_rr'])
            loss = loss_ro + loss_rr
        else:
            loss = loss_ro

        total_loss += loss.item()

        acc_ro, n_ro = compute_accuracy(logits_ro, inp['labels_ro'])
        ro_correct += acc_ro * n_ro
        ro_total += n_ro

        if inp['labels_rr'].numel() > 0:
            acc_rr, n_rr = compute_accuracy(logits_rr, inp['labels_rr'])
            rr_correct += acc_rr * n_rr
            rr_total += n_rr

    avg_loss = total_loss / len(dataset)
    ro_acc = ro_correct / max(ro_total, 1)
    rr_acc = rr_correct / max(rr_total, 1) if rr_total > 0 else float('nan')

    return avg_loss, ro_acc, rr_acc


def train(n_epochs=50, lr=1e-3, hidden_dim=64, num_heads=4, num_layers=2,
          H=15, ff_hidden=128, device='cpu'):
    """Ana egitim fonksiyonu."""

    # Dataset yukle
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    train_data = load_dataset(os.path.join(data_dir, 'dataset_train.pkl'))
    val_data = load_dataset(os.path.join(data_dir, 'dataset_val.pkl'))

    print(f"Dataset: {len(train_data)} train, {len(val_data)} val")

    # Model
    model = GATBinaryPredictor(
        hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers,
        H=H, ff_hidden=ff_hidden
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {total_params} parametre")

    # Optimizer ve loss
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCEWithLogitsLoss()

    # Egitim dongusu
    print(f"\nEgitim: {n_epochs} epoch, lr={lr}")
    print(f"{'Epoch':>5} {'T_Loss':>8} {'T_RO%':>7} {'T_RR%':>7} "
          f"{'V_Loss':>8} {'V_RO%':>7} {'V_RR%':>7} {'Time':>6}")
    print("-" * 62)

    best_val_acc = 0
    model_dir = os.path.join(data_dir, '..', 'models')
    os.makedirs(model_dir, exist_ok=True)
    best_path = os.path.join(model_dir, 'best_model.pt')

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        # Train
        t_loss, t_ro, t_rr = train_one_epoch(model, train_data, optimizer, criterion, device)

        # Validate
        v_loss, v_ro, v_rr = evaluate(model, val_data, criterion, device)

        dt = time.time() - t0

        # Best model kaydet
        v_acc_avg = v_ro  # RO accuracy'yi ana metrik olarak kullan
        if v_acc_avg > best_val_acc:
            best_val_acc = v_acc_avg
            torch.save(model.state_dict(), best_path)
            marker = " *"
        else:
            marker = ""

        t_rr_str = f"{100*t_rr:6.1f}" if not np.isnan(t_rr) else "   N/A"
        v_rr_str = f"{100*v_rr:6.1f}" if not np.isnan(v_rr) else "   N/A"

        print(f"{epoch:>5} {t_loss:>8.4f} {100*t_ro:>6.1f} {t_rr_str} "
              f"{v_loss:>8.4f} {100*v_ro:>6.1f} {v_rr_str} {dt:>5.1f}s{marker}")

    print(f"\nEn iyi val RO accuracy: {100*best_val_acc:.1f}%")
    print(f"Model kaydedildi: {best_path}")

    return model


if __name__ == "__main__":
    train(n_epochs=50, lr=1e-3)
