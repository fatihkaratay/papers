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
# Faz 5.5-5.6: Decoder
# =============================================================================

class EdgeDecoder(nn.Module):
    """Bir edge tipi icin binary tahmin yapan decoder.

    Akis:
      h_i (src node embedding, 64D)  ─┐
                                        ├─ concat ─► [128D] ─► FF ─► [H*4] ─► sigmoid
      h_j (dst node embedding, 64D)  ─┘

    Neden iki node'u birlestiriyoruz?
      Binary karar, iki node ARASINDAKI iliskiye bagli.
      Ornegin robot-engel: "robot engelin sagından mi solundan mi gececek?"
      Bu karar tek basina ne robottan ne de engelden anlasilir — ikisine birden bakmak lazim.

    Neden ayri decoder'lar (5.6)?
      - RO decoder (Omega_RO): robot-engel binary'leri
        Robot engelin hangi tarafindan gececek? (4 yon x H adim)
      - RR decoder (Omega_R): robot-robot binary'leri
        Iki robot birbirinin hangi tarafinda kalacak? (4 yon x H adim)
      Fiziksel anlam farkli → farkli agirliklar ogrenmeli.
    """

    def __init__(self, hidden_dim=64, output_dim=60, ff_hidden=128):
        """
        Args:
            hidden_dim: encoder'dan gelen node embedding boyutu
            output_dim: H * 4 (ornegin 15 * 4 = 60)
            ff_hidden:  feedforward katmanin gizli boyutu
        """
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim * 2, ff_hidden),  # [h_i || h_j] -> ff_hidden
            nn.ReLU(),
            nn.Linear(ff_hidden, output_dim),       # ff_hidden -> H*4
        )

    def forward(self, h, edge_index):
        """
        Args:
            h:          (N, hidden_dim) tum node embedding'leri
            edge_index: (2, E_type) bu tip icin edge'ler

        Returns:
            pred: (E_type, output_dim) her edge icin binary tahminler (logits)
        """
        src = edge_index[0]  # kaynak node indexleri
        dst = edge_index[1]  # hedef node indexleri
        h_src = h[src]       # (E_type, hidden_dim)
        h_dst = h[dst]       # (E_type, hidden_dim)
        edge_emb = torch.cat([h_src, h_dst], dim=1)  # (E_type, hidden_dim*2)
        return self.ff(edge_emb)  # (E_type, output_dim) — logits (sigmoid oncesi)


# =============================================================================
# Faz 5.5-5.6: Tam Model
# =============================================================================

class GATBinaryPredictor(nn.Module):
    """Encoder + iki ayri decoder = tam model.

    Encoder (paylasilan):
      Tum node'lar icin ortak embedding ogrenir.

    Decoder RO (Omega_RO):
      Robot-engel edge'leri icin binary tahmin.

    Decoder RR (Omega_R):
      Robot-robot edge'leri icin binary tahmin.
    """

    def __init__(self, robot_feat_dim=4, obstacle_feat_dim=5,
                 hidden_dim=64, num_heads=4, num_layers=2,
                 H=15, ff_hidden=128, dropout=0.0):
        super().__init__()
        self.encoder = GATEncoder(
            robot_feat_dim, obstacle_feat_dim,
            hidden_dim, num_heads, num_layers, dropout
        )
        output_dim = H * 4  # her edge icin H zaman adimi x 4 yon
        self.decoder_ro = EdgeDecoder(hidden_dim, output_dim, ff_hidden)
        self.decoder_rr = EdgeDecoder(hidden_dim, output_dim, ff_hidden)
        self.H = H

    def forward(self, x_robot, x_obstacle, edge_index_all,
                edge_index_ro, edge_index_rr):
        """
        Args:
            x_robot:        (NR, 4)
            x_obstacle:     (NO, 5)
            edge_index_all: (2, E) tum edge'ler — encoder icin
            edge_index_ro:  (2, E_ro) robot-engel edge'leri — decoder icin
            edge_index_rr:  (2, E_rr) robot-robot edge'leri — decoder icin

        Returns:
            logits_ro: (E_ro, H*4) robot-engel binary logits
            logits_rr: (E_rr, H*4) robot-robot binary logits
        """
        # Encoder: tum node embedding'lerini hesapla
        h = self.encoder(x_robot, x_obstacle, edge_index_all)

        # Decoder: her edge tipi icin ayri binary tahmin
        logits_ro = self.decoder_ro(h, edge_index_ro)
        logits_rr = self.decoder_rr(h, edge_index_rr)

        return logits_ro, logits_rr


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    # === Ornek senaryo: 3 robot, 2 engel, H=15 ===
    NR, NO, H = 3, 2, 15
    x_robot = torch.randn(NR, 4)
    x_obstacle = torch.randn(NO, 5)

    # --- Edge index'leri olustur ---
    # Node indexleme: robotlar 0..NR-1, engeller NR..NR+NO-1
    all_edges = []

    # RO: robot -> engel (binary tahmin edilecek)
    ro_edges = []
    for i in range(NR):
        for o in range(NO):
            ro_edges.append([i, NR + o])
    all_edges.extend(ro_edges)

    # OR: engel -> robot (bilgi akisi, binary yok)
    for o in range(NO):
        for i in range(NR):
            all_edges.append([NR + o, i])

    # RR: robot -> robot (binary tahmin edilecek)
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

    # --- Tam model testi ---
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

    # Sigmoid -> olasilik
    probs_ro = torch.sigmoid(logits_ro)
    print(f"\nOrnek RO edge 0 (Robot0->Engel0):")
    print(f"  Ilk 8 olasilik: {probs_ro[0, :8].tolist()}")
    print(f"  Min: {probs_ro.min():.3f}, Max: {probs_ro.max():.3f}")

    # Parametre sayisi
    total = sum(p.numel() for p in model.parameters())
    print(f"\nToplam parametre: {total}")
    print(f"  Encoder: {sum(p.numel() for p in model.encoder.parameters())}")
    print(f"  Decoder RO: {sum(p.numel() for p in model.decoder_ro.parameters())}")
    print(f"  Decoder RR: {sum(p.numel() for p in model.decoder_rr.parameters())}")
