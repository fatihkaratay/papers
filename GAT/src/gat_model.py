import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class ProjectionLayer(nn.Module):

    def __init__(self, robot_feat_dim=4, obstacle_feat_dim=5, hidden_dim=64):
        super().__init__()
        self.proj_robot = nn.Linear(robot_feat_dim, hidden_dim)
        self.proj_obstacle = nn.Linear(obstacle_feat_dim, hidden_dim)

    def forward(self, x_robot, x_obstacle):
        h_robot = self.proj_robot(x_robot)        # (NR, hidden_dim)
        h_obstacle = self.proj_obstacle(x_obstacle)  # (NO, hidden_dim)
        h = torch.cat([h_robot, h_obstacle], dim=0)  # (NR+NO, hidden_dim)
        return h


class GATEncoder(nn.Module):
    def __init__(self, robot_feat_dim=4, obstacle_feat_dim=5,
                 hidden_dim=64, num_heads=4, num_layers=2, dropout=0.0):
        super().__init__()
        self.projection = ProjectionLayer(robot_feat_dim, obstacle_feat_dim, hidden_dim)

        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // num_heads,
                        heads=num_heads, concat=True, dropout=dropout)
            )

        self.num_layers = num_layers

    def forward(self, x_robot, x_obstacle, edge_index):
        h = self.projection(x_robot, x_obstacle)  # (NR+NO, hidden_dim)

        for gat in self.gat_layers:
            h = gat(h, edge_index)  # attention + aggregation
            h = F.elu(h)            # non-linear aktivasyon

        return h

class EdgeDecoder(nn.Module):
    def __init__(self, hidden_dim=64, output_dim=60, ff_hidden=128, dropout=0.0):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim * 2, ff_hidden),  # [h_i || h_j] -> ff_hidden
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden, output_dim),       # ff_hidden -> H*4
        )

    def forward(self, h, edge_index):
        src = edge_index[0] 
        dst = edge_index[1] 
        h_src = h[src]
        h_dst = h[dst]
        edge_emb = torch.cat([h_src, h_dst], dim=1)
        return self.ff(edge_emb)

class GATBinaryPredictor(nn.Module):
    def __init__(self, robot_feat_dim=4, obstacle_feat_dim=5,
                 hidden_dim=64, num_heads=4, num_layers=2,
                 H=15, ff_hidden=128, dropout=0.0):
        super().__init__()
        self.encoder = GATEncoder(
            robot_feat_dim, obstacle_feat_dim,
            hidden_dim, num_heads, num_layers, dropout
        )
        output_dim = H * 4
        self.decoder_ro = EdgeDecoder(hidden_dim, output_dim, ff_hidden, dropout)
        self.decoder_rr = EdgeDecoder(hidden_dim, output_dim, ff_hidden, dropout)
        self.H = H

    def forward(self, x_robot, x_obstacle, edge_index_all,
                edge_index_ro, edge_index_rr):
        h = self.encoder(x_robot, x_obstacle, edge_index_all)

        logits_ro = self.decoder_ro(h, edge_index_ro)
        logits_rr = self.decoder_rr(h, edge_index_rr)

        return logits_ro, logits_rr

if __name__ == "__main__":
    torch.manual_seed(42)

    NR, NO, H = 3, 2, 15
    x_robot = torch.randn(NR, 4)
    x_obstacle = torch.randn(NO, 5)

    all_edges = []

    ro_edges = []
    for i in range(NR):
        for o in range(NO):
            ro_edges.append([i, NR + o])
    all_edges.extend(ro_edges)

    for o in range(NO):
        for i in range(NR):
            all_edges.append([NR + o, i])

    rr_edges = []
    for i in range(NR):
        for j in range(NR):
            if i != j:
                rr_edges.append([i, j])
    all_edges.extend(rr_edges)

    # OO: engel -> engel (bilgi akisi)
    for o1 in range(NO):
        for o2 in range(NO):
            if o1 != o2:
                all_edges.append([NR + o1, NR + o2])

    edge_index_all = torch.tensor(all_edges, dtype=torch.long).t()
    edge_index_ro = torch.tensor(ro_edges, dtype=torch.long).t()
    edge_index_rr = torch.tensor(rr_edges, dtype=torch.long).t()

    model = GATBinaryPredictor(hidden_dim=64, num_heads=4, num_layers=2, H=H)

    logits_ro, logits_rr = model(
        x_robot, x_obstacle, edge_index_all,
        edge_index_ro, edge_index_rr
    )

    print("=== GATBinaryPredictor Test ===")
    print(f"Input:  {NR} robot, {NO} engel, H={H}")
    print(f"Edges:  {edge_index_all.shape[1]} total, "
          f"{edge_index_ro.shape[1]} RO, {edge_index_rr.shape[1]} RR")
    print(f"\nOutput:")
    print(f"  logits_ro: {logits_ro.shape}  (E_ro={NR*NO}, H*4={H*4})")
    print(f"  logits_rr: {logits_rr.shape}  (E_rr={len(rr_edges)}, H*4={H*4})")

    probs_ro = torch.sigmoid(logits_ro)
    print(f"\nOrnek RO edge 0 (Robot0->Engel0):")
    print(f"  Ilk 8 olasilik: {probs_ro[0, :8].tolist()}")
    print(f"  Min: {probs_ro.min():.3f}, Max: {probs_ro.max():.3f}")

    total = sum(p.numel() for p in model.parameters())
    print(f"\nToplam parametre: {total}")
    print(f"  Encoder: {sum(p.numel() for p in model.encoder.parameters())}")
    print(f"  Decoder RO: {sum(p.numel() for p in model.decoder_ro.parameters())}")
    print(f"  Decoder RR: {sum(p.numel() for p in model.decoder_rr.parameters())}")
