"""
Faz 5: Heterogeneous GAT modeli.

Mimari (makaledeki Section IV-B):
  1. Projection: farki node tiplerini ayni boyuta tasi (robot 4D, obstacle 5D -> hidden_dim)
  2. GAT Encoder: 2 katman GAT, attention ile komsulardan bilgi topla
  3. Decoder: edge embedding'den binary tahmin (ayri RO ve RR decoder'lar)

Adim adim build ediyoruz — her faz yeni bir parca ekliyor.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


# =============================================================================
# Faz 5.2: Projection Layer
# =============================================================================

class ProjectionLayer(nn.Module):
    """Farklı node tiplerini aynı feature space'e taşır.

    Robot features  [4D] --Linear--> [hidden_dim]
    Obstacle features [5D] --Linear--> [hidden_dim]

    Neden gerekli?
      GAT katmanında tüm node'lar birbirleriyle etkileşime giriyor.
      Attention hesabı için aynı boyutta olmaları şart.
    """

    def __init__(self, robot_feat_dim=4, obstacle_feat_dim=5, hidden_dim=64):
        super().__init__()
        self.proj_robot = nn.Linear(robot_feat_dim, hidden_dim)
        self.proj_obstacle = nn.Linear(obstacle_feat_dim, hidden_dim)

    def forward(self, x_robot, x_obstacle):
        """
        Args:
            x_robot:    (NR, 4) robot features
            x_obstacle: (NO, 5) obstacle features

        Returns:
            h: (NR + NO, hidden_dim) tüm node'ların ortak embedding'i
               ilk NR satır robot, sonraki NO satır obstacle
        """
        h_robot = self.proj_robot(x_robot)        # (NR, hidden_dim)
        h_obstacle = self.proj_obstacle(x_obstacle)  # (NO, hidden_dim)
        h = torch.cat([h_robot, h_obstacle], dim=0)  # (NR+NO, hidden_dim)
        return h


# =============================================================================
# Faz 5.3-5.4: GAT Encoder
# =============================================================================

class GATEncoder(nn.Module):
    """Projection + 2-layer GAT.

    Akis:
      x_robot, x_obstacle
        |
        v
      ProjectionLayer  -->  h (NR+NO, hidden_dim)
        |
        v
      GATConv layer 1  -->  h (NR+NO, hidden_dim)   + ELU aktivasyon
        |
        v
      GATConv layer 2  -->  h (NR+NO, hidden_dim)   + ELU aktivasyon

    GAT katmaninda ne oluyor?
      Her node icin:
        1) Komsularina bak (edge_index'ten)
        2) Her komsu icin attention skoru hesapla: "bu komsu ne kadar onemli?"
        3) Skorlari softmax ile normalize et
        4) Komsularin embedding'lerini agirlikli topla
        5) Sonuc: o node'un yeni, zenginlestirilmis embedding'i

    edge_index neden gerekli?
      GAT'e "kim kimin komsusu" bilgisini verir.
      Bizim 4 edge tipimizi (RO, RR, OR, OO) tek bir edge_index'te birlestiriyoruz
      cunku projection sonrasi tum node'lar ayni space'te.
    """

    def __init__(self, robot_feat_dim=4, obstacle_feat_dim=5,
                 hidden_dim=64, num_heads=4, num_layers=2, dropout=0.0):
        super().__init__()
        self.projection = ProjectionLayer(robot_feat_dim, obstacle_feat_dim, hidden_dim)

        # GAT katmanlari
        # Multi-head attention: her head bagimsiz attention ogrenir
        # num_heads=4, hidden_dim=64 -> her head 16 boyut uretir -> concat -> 64
        self.gat_layers = nn.ModuleList()
        for i in range(num_layers):
            # concat=True: head'leri birlestir (4 head * 16 = 64)
            self.gat_layers.append(
                GATConv(hidden_dim, hidden_dim // num_heads,
                        heads=num_heads, concat=True, dropout=dropout)
            )

        self.num_layers = num_layers

    def forward(self, x_robot, x_obstacle, edge_index):
        """
        Args:
            x_robot:    (NR, 4) robot features
            x_obstacle: (NO, 5) obstacle features
            edge_index: (2, E) tum edge'ler (RO+RR+OR+OO birlesmis)

        Returns:
            h: (NR+NO, hidden_dim) her node'un son embedding'i
        """
        # 1) Projection: farki tipleri ayni boyuta tasi
        h = self.projection(x_robot, x_obstacle)  # (NR+NO, hidden_dim)

        # 2) GAT katmanlari
        for gat in self.gat_layers:
            h = gat(h, edge_index)  # attention + aggregation
            h = F.elu(h)            # non-linear aktivasyon

        return h


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    # === Ornek: 3 robot, 2 engel ===
    NR, NO = 3, 2
    x_robot = torch.randn(NR, 4)     # [px, py, gx, gy]
    x_obstacle = torch.randn(NO, 5)  # [cx, cy, angle, L, W]

    # --- 5.2: Projection test ---
    proj = ProjectionLayer(robot_feat_dim=4, obstacle_feat_dim=5, hidden_dim=64)
    h = proj(x_robot, x_obstacle)
    print("=== Projection Layer Test ===")
    print(f"Input:  x_robot {x_robot.shape}, x_obstacle {x_obstacle.shape}")
    print(f"Output: h {h.shape}")

    # --- 5.3-5.4: GAT Encoder test ---
    # Edge index olustur: tum edge tiplerini birlestir
    # Node indexleme: robotlar 0..NR-1, engeller NR..NR+NO-1
    edges = []
    # RO: her robot -> her engel
    for i in range(NR):
        for o in range(NO):
            edges.append([i, NR + o])
    # OR: her engel -> her robot (ters yon, bilgi akisi)
    for o in range(NO):
        for i in range(NR):
            edges.append([NR + o, i])
    # RR: robot-robot (tam bagli)
    for i in range(NR):
        for j in range(NR):
            if i != j:
                edges.append([i, j])
    # OO: engel-engel
    for o1 in range(NO):
        for o2 in range(NO):
            if o1 != o2:
                edges.append([NR + o1, NR + o2])

    edge_index = torch.tensor(edges, dtype=torch.long).t()  # (2, E)

    print(f"\n=== GAT Encoder Test ===")
    print(f"Nodes: {NR} robot + {NO} engel = {NR+NO} toplam")
    print(f"Edges: {edge_index.shape[1]} (RO + OR + RR + OO)")

    encoder = GATEncoder(hidden_dim=64, num_heads=4, num_layers=2)
    h = encoder(x_robot, x_obstacle, edge_index)

    print(f"Output: h {h.shape}")
    print(f"  h[0] (Robot 0):   [{h[0, :4].tolist()}, ...]")
    print(f"  h[{NR}] (Engel 0): [{h[NR, :4].tolist()}, ...]")

    total = sum(p.numel() for p in encoder.parameters())
    print(f"\nToplam parametre: {total}")
    print(f"\nKatmanlar:")
    for name, p in encoder.named_parameters():
        print(f"  {name}: {p.shape}")
