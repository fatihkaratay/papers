"""
Faz 3.1-3.4: Coklu robot MICP — robot-robot + robot-engel collision avoidance.

Faz 2'deki tek robot MICP'ye ek olarak:
  - Robot-robot collision avoidance (Denklem 4):
      px_i - px_j >= 2*dmin - M*b1
      px_j - px_i >= 2*dmin - M*b2
      py_i - py_j >= 2*dmin - M*b3
      py_j - py_i >= 2*dmin - M*b4
      b1 + b2 + b3 + b4 <= 3

  - Proximity-based edge olusturma (Denklem 5):
      Sadece ||p_i(t) - p_j(t)|| <= dprox olan ciftler arasinda kisit eklenir.
"""

import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from environment import Environment


def solve_multi_robot_micp(robots_p, robots_v, robots_goal,
                           obstacles, H, tau,
                           bounds, vmax, amax, dmin,
                           dprox=3.0, M=100.0,
                           w_pt=10.0, w_p=1.0, w_u=1.0):
    """Coklu robot + engeller icin optimal trajectory (MICP).

    Args:
        robots_p: list of np.array — her robotun baslangic pozisyonu
        robots_v: list of np.array — her robotun baslangic hizi
        robots_goal: list of np.array — her robotun hedefi
        obstacles: list of Obstacle nesneleri
        H: horizon length
        tau: sampling period (s)
        bounds: (px_min, px_max, py_min, py_max)
        vmax, amax, dmin: limitler
        dprox: proximity esigi — sadece bu kadar yakin robotlar arasi kisit eklenir
        M: big-M sabiti
        w_pt, w_p, w_u: cost weights

    Returns:
        p_trajs: list of np.array, her robot icin (H+1, 2) trajectory
        v_trajs: list of np.array, her robot icin (H+1, 2) hiz
        u_trajs: list of np.array, her robot icin (H, 2) kontrol
        robot_edges: list of (i, j) — hangi robot ciftleri arasinda kisit var
    """
    NR = len(robots_p)
    n_obs = len(obstacles)
    px_min, px_max, py_min, py_max = bounds

    model = gp.Model("multi_robot_micp")
    model.setParam('OutputFlag', 0)

    # --- Her robot icin surekli degiskenler ---
    p = {}  # p[i] shape (H+1, 2)
    v = {}  # v[i] shape (H+1, 2)
    u = {}  # u[i] shape (H, 2)
    for i in range(NR):
        p[i] = model.addMVar((H + 1, 2),
                              lb=[px_min, py_min], ub=[px_max, py_max],
                              name=f"p_{i}")
        v[i] = model.addMVar((H + 1, 2), lb=-vmax, ub=vmax, name=f"v_{i}")
        u[i] = model.addMVar((H, 2), lb=-amax, ub=amax, name=f"u_{i}")

    # --- Robot-engel binary degiskenleri ---
    b_obs = {}  # b_obs[i, o] shape (H, 4)
    for i in range(NR):
        for o_idx in range(n_obs):
            b_obs[i, o_idx] = model.addMVar(
                (H, 4), vtype=GRB.BINARY, name=f"b_obs_{i}_{o_idx}")

    # --- Proximity-based robot-robot edges (Denklem 5) ---
    robot_edges = []
    for i in range(NR):
        for j in range(i + 1, NR):
            dist = np.linalg.norm(robots_p[i] - robots_p[j])
            if dist <= dprox:
                robot_edges.append((i, j))

    # --- Robot-robot binary degiskenleri (Denklem 4) ---
    b_rob = {}  # b_rob[i, j] shape (H, 4)
    for (i, j) in robot_edges:
        b_rob[i, j] = model.addMVar(
            (H, 4), vtype=GRB.BINARY, name=f"b_rob_{i}_{j}")

    # === KISITLAR ===

    for i in range(NR):
        # --- Baslangic durumu ---
        model.addConstr(p[i][0, :] == robots_p[i], name=f"p_init_{i}")
        model.addConstr(v[i][0, :] == robots_v[i], name=f"v_init_{i}")

        # --- Dinamik kisitlar (Denklem 1) ---
        for k in range(H):
            model.addConstr(
                p[i][k+1, :] == p[i][k, :] + tau * v[i][k, :]
                + 0.5 * tau**2 * u[i][k, :],
                name=f"dyn_p_{i}_{k}")
            model.addConstr(
                v[i][k+1, :] == v[i][k, :] + tau * u[i][k, :],
                name=f"dyn_v_{i}_{k}")

        # --- Robot-engel collision avoidance (Denklem 6) ---
        for o_idx, obs in enumerate(obstacles):
            ca = np.cos(obs.angle)
            sa = np.sin(obs.angle)
            ox, oy = obs.center
            L = obs.half_length
            W = obs.half_width

            for k in range(H):
                pk = k + 1  # p[pk] = p[k+1], yani k=1..H

                model.addConstr(
                    ca * (p[i][pk, 0] - ox) + sa * (p[i][pk, 1] - oy)
                    >= L + dmin - M * b_obs[i, o_idx][k, 0],
                    name=f"obs_{i}_{o_idx}_k{k}_right")
                model.addConstr(
                    -sa * (p[i][pk, 0] - ox) + ca * (p[i][pk, 1] - oy)
                    >= W + dmin - M * b_obs[i, o_idx][k, 1],
                    name=f"obs_{i}_{o_idx}_k{k}_top")
                model.addConstr(
                    -ca * (p[i][pk, 0] - ox) - sa * (p[i][pk, 1] - oy)
                    >= L + dmin - M * b_obs[i, o_idx][k, 2],
                    name=f"obs_{i}_{o_idx}_k{k}_left")
                model.addConstr(
                    sa * (p[i][pk, 0] - ox) - ca * (p[i][pk, 1] - oy)
                    >= W + dmin - M * b_obs[i, o_idx][k, 3],
                    name=f"obs_{i}_{o_idx}_k{k}_bottom")
                model.addConstr(
                    b_obs[i, o_idx][k, 0] + b_obs[i, o_idx][k, 1]
                    + b_obs[i, o_idx][k, 2] + b_obs[i, o_idx][k, 3] <= 3,
                    name=f"obs_{i}_{o_idx}_k{k}_sum")

    # --- Robot-robot collision avoidance (Denklem 4) ---
    for (i, j) in robot_edges:
        for k in range(H):
            pk = k + 1  # collision avoidance p[1]..p[H] icin

            # Yon 0: px_i - px_j >= 2*dmin - M*b1
            model.addConstr(
                p[i][pk, 0] - p[j][pk, 0] >= 2 * dmin - M * b_rob[i, j][k, 0],
                name=f"rob_{i}_{j}_k{k}_xpos")
            # Yon 1: px_j - px_i >= 2*dmin - M*b2
            model.addConstr(
                p[j][pk, 0] - p[i][pk, 0] >= 2 * dmin - M * b_rob[i, j][k, 1],
                name=f"rob_{i}_{j}_k{k}_xneg")
            # Yon 2: py_i - py_j >= 2*dmin - M*b3
            model.addConstr(
                p[i][pk, 1] - p[j][pk, 1] >= 2 * dmin - M * b_rob[i, j][k, 2],
                name=f"rob_{i}_{j}_k{k}_ypos")
            # Yon 3: py_j - py_i >= 2*dmin - M*b4
            model.addConstr(
                p[j][pk, 1] - p[i][pk, 1] >= 2 * dmin - M * b_rob[i, j][k, 3],
                name=f"rob_{i}_{j}_k{k}_yneg")
            # En az 1 yon aktif
            model.addConstr(
                b_rob[i, j][k, 0] + b_rob[i, j][k, 1]
                + b_rob[i, j][k, 2] + b_rob[i, j][k, 3] <= 3,
                name=f"rob_{i}_{j}_k{k}_sum")

    # === AMAC FONKSIYONU (Denklem 7-8, tum robotlarin toplami) ===
    obj = 0
    for i in range(NR):
        # Terminal cost
        p_err_T = p[i][H, :] - robots_goal[i]
        obj += w_pt * (p_err_T @ p_err_T)
        # Running cost
        for k in range(H):
            p_err = p[i][k, :] - robots_goal[i]
            obj += w_p * (p_err @ p_err)
            obj += w_u * (u[i][k, :] @ u[i][k, :])

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        print(f"Warning: MICP status = {model.status}")

    # Sonuclari topla
    p_trajs = [p[i].X for i in range(NR)]
    v_trajs = [v[i].X for i in range(NR)]
    u_trajs = [u[i].X for i in range(NR)]

    # Binary cozumleri topla
    b_obs_sol = {}
    for i in range(NR):
        for o_idx in range(n_obs):
            b_obs_sol[i, o_idx] = np.round(b_obs[i, o_idx].X).astype(int)

    b_rob_sol = {}
    for (i, j) in robot_edges:
        b_rob_sol[i, j] = np.round(b_rob[i, j].X).astype(int)

    return p_trajs, v_trajs, u_trajs, robot_edges, b_obs_sol, b_rob_sol


# --- Demo: 2 robot + 1 engel ---
if __name__ == "__main__":
    tau = 0.2
    H = 20
    env_bounds = (-3.0, 1.0, -0.5, 2.5)
    vmax = 0.5
    amax = 0.5
    dmin = 0.2
    dprox = 5.0  # tum robotlar arasi kisit olsun (kucuk senaryo)

    env = Environment(bounds=env_bounds)
    env.add_obstacle(center=(-1.0, 1.0), half_length=0.3, half_width=0.2)

    # Robot 1: soldan saga
    env.add_robot(position=(-2.5, 1.0), goal=(0.5, 1.0))
    # Robot 2: sagdan sola (karsidan karsilasacaklar!)
    env.add_robot(position=(0.5, 1.0), goal=(-2.5, 1.0))

    robots_p = [r.position.copy() for r in env.robots]
    robots_v = [r.velocity.copy() for r in env.robots]
    robots_goal = [r.goal.copy() for r in env.robots]

    print(f"Robots: {len(env.robots)}, Obstacles: {len(env.obstacles)}")
    print("Solving multi-robot MICP...")

    p_trajs, v_trajs, u_trajs, edges, b_obs_sol, b_rob_sol = \
        solve_multi_robot_micp(
            robots_p, robots_v, robots_goal,
            env.obstacles, H, tau, env_bounds,
            vmax, amax, dmin, dprox
        )

    print(f"Robot-robot edges: {edges}")

    # --- Gorsellestirme ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange']

    # 1) Trajectory
    ax = axes[0]
    env.plot(ax=ax)
    for i in range(len(env.robots)):
        ax.plot(p_trajs[i][:, 0], p_trajs[i][:, 1], '-o',
                color=colors[i], markersize=3, linewidth=2, alpha=0.7,
                label=f'Robot {i+1}')
        # Guvenli daire (baslangic ve bitis)
        for t_idx in [0, H]:
            circle = plt.Circle(p_trajs[i][t_idx], dmin,
                                fill=False, edgecolor=colors[i],
                                linestyle=':', alpha=0.4)
            ax.add_patch(circle)
    ax.set_title('Multi-Robot MICP Trajectories')
    ax.legend()

    # 2) Robotlar arasi mesafe
    ax = axes[1]
    time = np.arange(H + 1) * tau
    for (i, j) in edges:
        dists = np.linalg.norm(p_trajs[i] - p_trajs[j], axis=1)
        ax.plot(time, dists, '-o', markersize=3,
                label=f'Robot {i+1}-{j+1}')
    ax.axhline(y=2 * dmin, color='red', linestyle='--',
               label=f'2*dmin={2*dmin}', alpha=0.7)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('distance (m)')
    ax.set_title('Inter-Robot Distance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../plots/faz3_multi_robot_micp.png', dpi=150)
    plt.show()
    print("Saved: plots/faz3_multi_robot_micp.png")
