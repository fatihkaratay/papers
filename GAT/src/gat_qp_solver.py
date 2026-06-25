"""
Faz 6.1-6.2: GAT ile binary tahmin et, kalan QP'yi coz.

Framework'un online pipeline'i:
  1. Senaryo bilgisinden graph olustur
  2. GAT ile binary degiskenleri tahmin et (cok hizli, ~ms)
  3. Binary'leri sabitle, kalan convex QP'yi GUROBI ile coz (cok hizli, ~ms)
  4. Sonuc: trajectory

Neden hizli?
  MICP = NP-hard (binary + continuous, branch-and-bound)
  QP   = polynomial (sadece continuous, interior-point)
  GAT binary'leri tahmin edince MICP → QP donusumu bedava!
"""

import os
import time
import pickle
import numpy as np
import torch
import gurobipy as gp
from gurobipy import GRB

from gat_model import GATBinaryPredictor
from train_gat import graph_to_model_input, compute_normalization


def _enforce_sum_constraint(probs):
    """Post-processing: b1+b2+b3+b4 <= 3 kisitini zorla.

    MICP'de her zaman adiminda 4 binary'nin toplami <= 3 olmali,
    yani en az 1 tanesi 0 (aktif kisit) olmali.
    Yoksa "robot engelin hem saginda hem solunda hem ustunde hem altinda ol"
    denmis olur — imkansiz!

    Yontem:
      1. Threshold 0.5 ile binary'ye cevir
      2. Eger 4'u de 0 ise → en yuksek olasilikli yonu 1 yap (en az emin oldugu kisiti gevset)
      3. Eger toplam > 3 ise zaten ok (en az 1 aktif)

    Args:
        probs: (H, 4) float — sigmoid olasililiklari

    Returns:
        binaries: (H, 4) int — duzeltilmis binary'ler
    """
    binaries = (probs > 0.5).astype(int)

    for k in range(binaries.shape[0]):
        row = binaries[k]
        if row.sum() > 3:
            # 4'u de 1 → en dusuk olasilikli yonu 0 yap (en emin oldugu kisiti aktif birak)
            worst = np.argmin(probs[k])
            row[worst] = 0
        elif row.sum() == 0:
            # Hepsi 0 → en yuksek olasilikli yonu 1 yap (en az emin oldugu kisiti gevset)
            best = np.argmax(probs[k])
            row[best] = 1

    return binaries


def load_model(model_dir=None, device='cpu'):
    """Egitilmis GAT modelini ve norm stats'i yukle."""
    if model_dir is None:
        model_dir = os.path.join(os.path.dirname(__file__), '..', 'models')

    model = GATBinaryPredictor(hidden_dim=64, num_heads=4, num_layers=2,
                                H=15, ff_hidden=128, dropout=0.0)
    model.load_state_dict(torch.load(
        os.path.join(model_dir, 'best_model.pt'),
        map_location=device, weights_only=True
    ))
    model.eval()

    with open(os.path.join(model_dir, 'norm_stats.pkl'), 'rb') as f:
        norm_stats = pickle.load(f)

    return model, norm_stats


@torch.no_grad()
def predict_binaries(model, graph, norm_stats, device='cpu'):
    """GAT ile binary degiskenleri tahmin et.

    Args:
        model: egitilmis GATBinaryPredictor
        graph: collect_data.sample_to_graph() ciktisi
        norm_stats: normalization istatistikleri

    Returns:
        b_obs_pred: dict, (robot_i, obs_o) -> (H, 4) binary array
        b_rob_pred: dict, (robot_i, robot_j) -> (H, 4) binary array
    """
    inp = graph_to_model_input(graph, device, norm_stats)

    logits_ro, logits_rr = model(
        inp['x_robot'], inp['x_obstacle'], inp['edge_index_all'],
        inp['edge_index_ro'], inp['edge_index_rr']
    )

    # Sigmoid olasiliklarini al
    probs_ro = torch.sigmoid(logits_ro).cpu().numpy()  # (E_ro, H*4)
    probs_rr = torch.sigmoid(logits_rr).cpu().numpy()  # (E_rr, H*4)

    H = graph['H']
    NR = graph['node_feat_robot'].shape[0]
    NO = graph['node_feat_obstacle'].shape[0]

    # RO: edge_index_RO[0]=robot, edge_index_RO[1]=obstacle (lokal index)
    b_obs_pred = {}
    for e in range(probs_ro.shape[0]):
        ri = graph['edge_index_RO'][0, e]
        oi = graph['edge_index_RO'][1, e]
        probs = probs_ro[e].reshape(H, 4)
        b_obs_pred[ri, oi] = _enforce_sum_constraint(probs)

    # RR: edge_index_RR[0]=robot_i, edge_index_RR[1]=robot_j
    b_rob_pred = {}
    for e in range(probs_rr.shape[0]):
        ri = graph['edge_index_RR'][0, e]
        rj = graph['edge_index_RR'][1, e]
        probs = probs_rr[e].reshape(H, 4)
        b_rob_pred[ri, rj] = _enforce_sum_constraint(probs)

    return b_obs_pred, b_rob_pred


def solve_qp_with_fixed_binaries(robots_p, robots_v, robots_goal,
                                  obstacles, H, tau,
                                  bounds, vmax, amax, dmin,
                                  b_obs_fixed, b_rob_fixed, robot_edges,
                                  M=100.0, w_pt=10.0, w_p=1.0, w_u=1.0,
                                  soft_penalty=1000.0):
    """Binary'ler sabitlenmis QP'yi coz (her zaman soft constraints ile).

    Bu fonksiyon multi_robot_micp ile ayni, ama binary degiskenler
    GUROBI'nin karar vermesi yerine GAT'in tahmini ile sabitlenmis.
    Dolayisiyla problem artik MICP degil, sadece QP (convex).

    Her zaman soft constraints kullaniyoruz cunku:
      - GAT'in tahmini %100 dogru degil
      - Yanlis tahmin → infeasible QP → tekrar cozme suresi
      - Soft ile tek seferde cozmek daha hizli
      - Slack cezasi yuksek (1000) → mumkunse kisitlara uyar, mecbursa az gevsetir

    Args:
        b_obs_fixed: dict, (robot_i, obs_o) -> (H, 4) int array
        b_rob_fixed: dict, (robot_i, robot_j) -> (H, 4) int array
        robot_edges: list of (i, j) tuples
        soft_penalty: slack degiskenlerinin ceza agirligi

    Returns:
        p_trajs, v_trajs, u_trajs, total_slack
    """
    NR = len(robots_p)
    n_obs = len(obstacles)
    px_min, px_max, py_min, py_max = bounds

    model = gp.Model("qp_fixed_binary")
    model.setParam('OutputFlag', 0)

    # --- Surekli degiskenler (MICP ile ayni) ---
    p = {}
    v = {}
    u = {}
    for i in range(NR):
        p[i] = model.addMVar((H + 1, 2),
                              lb=[px_min, py_min], ub=[px_max, py_max],
                              name=f"p_{i}")
        v[i] = model.addMVar((H + 1, 2), lb=-vmax, ub=vmax, name=f"v_{i}")
        u[i] = model.addMVar((H, 2), lb=-amax, ub=amax, name=f"u_{i}")

    # --- Slack degiskenleri (her zaman aktif) ---
    # Slack = kisiti ne kadar gevsettigimiz. Ceza yuksek oldugu icin
    # optimizer mumkun oldugunca slack=0 tutar, ama mecbursa kullanir.
    slack_cost = 0
    s_obs = {}
    for i in range(NR):
        for o_idx in range(n_obs):
            s_obs[i, o_idx] = model.addMVar((H, 4), lb=0, name=f"s_obs_{i}_{o_idx}")
            slack_cost += soft_penalty * s_obs[i, o_idx].sum()

    s_rob = {}
    for (i, j) in robot_edges:
        s_rob[i, j] = model.addMVar((H, 4), lb=0, name=f"s_rob_{i}_{j}")
        slack_cost += soft_penalty * s_rob[i, j].sum()

    # === KISITLAR ===
    for i in range(NR):
        # Baslangic durumu
        model.addConstr(p[i][0, :] == robots_p[i])
        model.addConstr(v[i][0, :] == robots_v[i])

        # Dinamik
        for k in range(H):
            model.addConstr(
                p[i][k+1, :] == p[i][k, :] + tau * v[i][k, :]
                + 0.5 * tau**2 * u[i][k, :])
            model.addConstr(
                v[i][k+1, :] == v[i][k, :] + tau * u[i][k, :])

        # Robot-engel collision avoidance (binary'ler sabit!)
        for o_idx, obs in enumerate(obstacles):
            ca = np.cos(obs.angle)
            sa = np.sin(obs.angle)
            ox, oy = obs.center
            L = obs.half_length
            W = obs.half_width
            b = b_obs_fixed[i, o_idx]  # (H, 4) sabit integer

            for k in range(H):
                pk = k + 1
                # Sadece b=0 olan yonlerin kisiti AKTIF
                # b=1 olan yonler zaten M ile relaxed
                rhs_vals = [L + dmin, W + dmin, L + dmin, W + dmin]
                lhs_exprs = [
                    ca * (p[i][pk, 0] - ox) + sa * (p[i][pk, 1] - oy),
                    -sa * (p[i][pk, 0] - ox) + ca * (p[i][pk, 1] - oy),
                    -ca * (p[i][pk, 0] - ox) - sa * (p[i][pk, 1] - oy),
                    sa * (p[i][pk, 0] - ox) - ca * (p[i][pk, 1] - oy),
                ]
                for m in range(4):
                    rhs = rhs_vals[m] - M * float(b[k, m])
                    model.addConstr(lhs_exprs[m] >= rhs - s_obs[i, o_idx][k, m])

    # Robot-robot collision avoidance (binary'ler sabit!)
    for (i, j) in robot_edges:
        b = b_rob_fixed[i, j]  # (H, 4) sabit integer
        for k in range(H):
            pk = k + 1
            lhs_exprs = [
                p[i][pk, 0] - p[j][pk, 0],
                p[j][pk, 0] - p[i][pk, 0],
                p[i][pk, 1] - p[j][pk, 1],
                p[j][pk, 1] - p[i][pk, 1],
            ]
            for m in range(4):
                rhs = 2 * dmin - M * float(b[k, m])
                model.addConstr(lhs_exprs[m] >= rhs - s_rob[i, j][k, m])

    # === AMAC FONKSIYONU ===
    obj = 0
    for i in range(NR):
        p_err_T = p[i][H, :] - robots_goal[i]
        obj += w_pt * (p_err_T @ p_err_T)
        for k in range(H):
            p_err = p[i][k, :] - robots_goal[i]
            obj += w_p * (p_err @ p_err)
            obj += w_u * (u[i][k, :] @ u[i][k, :])

    obj += slack_cost
    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status == GRB.INFEASIBLE:
        return None

    p_trajs = [p[i].X for i in range(NR)]
    v_trajs = [v[i].X for i in range(NR)]
    u_trajs = [u[i].X for i in range(NR)]

    # Slack toplami
    total_slack = 0
    for i in range(NR):
        for o_idx in range(n_obs):
            total_slack += s_obs[i, o_idx].X.sum()
    for (i, j) in robot_edges:
        total_slack += s_rob[i, j].X.sum()

    return p_trajs, v_trajs, u_trajs, total_slack


# --- Demo: tek bir senaryo uzerinde GAT+QP vs MICP karsilastirmasi ---
if __name__ == "__main__":
    from scenario_generator import generate_scenario
    from multi_robot_micp import solve_multi_robot_micp
    from collect_data import sample_to_graph, refine_obs_binaries, refine_rob_binaries

    # Parametreler
    H, tau = 15, 0.2
    vmax, amax, dmin, dprox = 0.5, 0.5, 0.2, 5.0
    bounds = (-4.0, 4.0, -4.0, 4.0)

    # Rastgele senaryo
    rng = np.random.RandomState(100)
    env = generate_scenario(3, 2, bounds, dmin, rng)
    NR = len(env.robots)
    robots_p = [r.position.copy() for r in env.robots]
    robots_v = [r.velocity.copy() for r in env.robots]
    robots_goal = [r.goal.copy() for r in env.robots]

    print(f"Senaryo: {NR} robot, {len(env.obstacles)} engel")
    for i in range(NR):
        print(f"  R{i}: {robots_p[i]} -> {robots_goal[i]}")

    # --- 1) MICP (ground truth) ---
    print("\n--- MICP (GUROBI) ---")
    t0 = time.time()
    p_micp, v_micp, u_micp, edges, b_obs_gt, b_rob_gt = \
        solve_multi_robot_micp(robots_p, robots_v, robots_goal,
                                env.obstacles, H, tau, bounds,
                                vmax, amax, dmin, dprox)
    t_micp = time.time() - t0
    print(f"  Sure: {t_micp*1000:.1f} ms")
    print(f"  Edges: {edges}")

    # --- 2) GAT + QP ---
    print("\n--- GAT + QP ---")
    model, norm_stats = load_model()

    # Graph olustur (ayni formatta)
    # Oncelikle sample_to_graph icin gerekli format
    obstacle_data = []
    for obs in env.obstacles:
        obstacle_data.append({
            'center': obs.center.copy(),
            'half_length': obs.half_length,
            'half_width': obs.half_width,
            'angle': obs.angle,
        })
    b_obs_list = []
    for i in range(NR):
        for o in range(len(env.obstacles)):
            b_obs_list.append({
                'robot': i, 'obstacle': o,
                'binaries': np.zeros((H, 4)),  # dummy — sadece graph yapisi icin
            })
    b_rob_list = []
    for (i, j) in edges:
        b_rob_list.append({
            'robot_i': i, 'robot_j': j,
            'binaries': np.zeros((H, 4)),
        })

    sample = {
        'n_robots': NR,
        'n_obstacles': len(env.obstacles),
        'robots_p': robots_p,
        'robots_goal': robots_goal,
        'obstacles': obstacle_data,
        'robot_edges': edges,
        'b_obs': b_obs_list,
        'b_rob': b_rob_list,
    }
    graph = sample_to_graph(sample)

    # GAT tahmin
    t0 = time.time()
    b_obs_pred, b_rob_pred = predict_binaries(model, graph, norm_stats)
    t_gat = time.time() - t0
    print(f"  GAT inference: {t_gat*1000:.1f} ms")

    # QP coz
    t0 = time.time()
    result = solve_qp_with_fixed_binaries(
        robots_p, robots_v, robots_goal,
        env.obstacles, H, tau, bounds, vmax, amax, dmin,
        b_obs_pred, b_rob_pred, edges
    )
    t_qp = time.time() - t0

    if result is None:
        print("  QP: INFEASIBLE (soft ile de cozulemedi)")
    else:
        p_gat, v_gat, u_gat, slack = result
        print(f"  QP solve: {t_qp*1000:.1f} ms")
        print(f"  Toplam: {(t_gat+t_qp)*1000:.1f} ms (vs MICP {t_micp*1000:.1f} ms)")
        print(f"  Speedup: {t_micp/(t_gat+t_qp):.1f}x")
        if slack > 0:
            print(f"  Slack kullanildi: {slack:.4f} (soft constraints)")

    # --- Binary karsilastirmasi ---
    print("\n--- Binary Karsilastirmasi ---")
    ro_match, ro_total = 0, 0
    for (i, o) in b_obs_pred:
        match = (b_obs_pred[i, o] == b_obs_gt[i, o]).sum()
        total = b_obs_pred[i, o].size
        ro_match += match
        ro_total += total
    print(f"  RO accuracy: {ro_match}/{ro_total} = {100*ro_match/ro_total:.1f}%")

    rr_match, rr_total = 0, 0
    for (i, j) in b_rob_pred:
        if (i, j) in b_rob_gt:
            match = (b_rob_pred[i, j] == b_rob_gt[i, j]).sum()
            total = b_rob_pred[i, j].size
            rr_match += match
            rr_total += total
    if rr_total > 0:
        print(f"  RR accuracy: {rr_match}/{rr_total} = {100*rr_match/rr_total:.1f}%")
