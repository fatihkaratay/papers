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
# Test
# =============================================================================

if __name__ == "__main__":
    torch.manual_seed(42)

    # Ornek: 3 robot, 2 engel
    x_robot = torch.randn(3, 4)     # [px, py, gx, gy]
    x_obstacle = torch.randn(2, 5)  # [cx, cy, angle, L, W]

    proj = ProjectionLayer(robot_feat_dim=4, obstacle_feat_dim=5, hidden_dim=64)
    h = proj(x_robot, x_obstacle)

    print("=== Projection Layer Test ===")
    print(f"Input:  x_robot {x_robot.shape}, x_obstacle {x_obstacle.shape}")
    print(f"Output: h {h.shape}")  # (5, 64)
    print(f"  h[:3] = robot embeddings  (3, 64)")
    print(f"  h[3:] = obstacle embeddings (2, 64)")
    print(f"\nParametre sayisi:")
    for name, p in proj.named_parameters():
        print(f"  {name}: {p.shape} ({p.numel()} param)")
    total = sum(p.numel() for p in proj.parameters())
    print(f"  Toplam: {total} parametre")
