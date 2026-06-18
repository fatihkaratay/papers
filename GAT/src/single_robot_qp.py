"""
Faz 1.3-1.5: Tek robot, engelsiz — QP ile optimal trajectory planlama.

Makaledeki Denklem (7-8)'in basitlestirilmis hali:
  - Engel yok, binary yok, sadece convex QP
  - GUROBI ile cozuyoruz

Amac fonksiyonu:
  minimize  w_pt * ||p(H) - p_goal||^2                   (terminal cost)
          + sum_k [ w_p * ||p(k) - p_goal||^2            (tracking cost)
                  + w_u * ||u(k)||^2 ]                    (effort cost)

Kisitlar:
  p(k+1) = p(k) + tau*v(k) + 0.5*tau^2*u(k)   (dinamik - pozisyon)
  v(k+1) = v(k) + tau*u(k)                      (dinamik - hiz)
  p(0) = p_init, v(0) = v_init                  (baslangic durumu)
  px_min <= px(k) <= px_max                      (ortam siniri, Denklem 2)
  py_min <= py(k) <= py_max
  -vmax <= vx(k), vy(k) <= vmax                 (hiz limiti)
  -amax <= ux(k), uy(k) <= amax                 (ivme limiti)
"""

import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from environment import Environment


def solve_single_robot_qp(p_init, v_init, p_goal, H, tau,
                           bounds=None, vmax=None, amax=None,
                           w_pt=10.0, w_p=1.0, w_u=1.0):
    """Tek robot icin optimal trajectory hesapla (QP).

    Args:
        p_init: baslangic pozisyonu (px, py)
        v_init: baslangic hizi (vx, vy)
        p_goal: hedef pozisyonu (px_g, py_g)
        H: kontrol ufku (horizon length)
        tau: sampling period (s)
        bounds: (px_min, px_max, py_min, py_max) ortam siniri — None ise sinirsiz
        vmax: maksimum hiz (m/s) — None ise sinirsiz
        amax: maksimum ivme (m/s^2) — None ise sinirsiz
        w_pt: terminal cost weight
        w_p: tracking cost weight
        w_u: effort cost weight

    Returns:
        p_traj: pozisyon trajectory'si, shape (H+1, 2)
        v_traj: hiz trajectory'si, shape (H+1, 2)
        u_traj: kontrol inputlari, shape (H, 2)
    """
    model = gp.Model("single_robot_qp")
    model.setParam('OutputFlag', 0)  # GUROBI ciktisini sustur

    # --- Degiskenler (Denklem 2'deki bound'lar burada) ---
    # Pozisyon: p[k, dim] for k=0..H
    if bounds is not None:
        px_min, px_max, py_min, py_max = bounds
        p_lb = np.array([px_min, py_min])
        p_ub = np.array([px_max, py_max])
        p = model.addMVar((H + 1, 2), lb=p_lb, ub=p_ub, name="p")
    else:
        p = model.addMVar((H + 1, 2), lb=-GRB.INFINITY, name="p")

    # Hiz: v[k, dim] for k=0..H
    if vmax is not None:
        v = model.addMVar((H + 1, 2), lb=-vmax, ub=vmax, name="v")
    else:
        v = model.addMVar((H + 1, 2), lb=-GRB.INFINITY, name="v")

    # Kontrol (ivme): u[k, dim] for k=0..H-1
    if amax is not None:
        u = model.addMVar((H, 2), lb=-amax, ub=amax, name="u")
    else:
        u = model.addMVar((H, 2), lb=-GRB.INFINITY, name="u")

    # --- Kisitlar ---
    # Baslangic durumu: p(0) = p_init, v(0) = v_init
    model.addConstr(p[0, :] == p_init, name="p_init")
    model.addConstr(v[0, :] == v_init, name="v_init")

    # Dinamik kisitlar (Denklem 1):
    for k in range(H):
        # p(k+1) = p(k) + tau*v(k) + 0.5*tau^2*u(k)
        model.addConstr(
            p[k + 1, :] == p[k, :] + tau * v[k, :] + 0.5 * tau**2 * u[k, :],
            name=f"dyn_p_{k}"
        )
        # v(k+1) = v(k) + tau*u(k)
        model.addConstr(
            v[k + 1, :] == v[k, :] + tau * u[k, :],
            name=f"dyn_v_{k}"
        )

    # --- Amac fonksiyonu (Denklem 7-8) ---
    obj = 0

    # Terminal cost: w_pt * ||p(H) - p_goal||^2
    p_err_terminal = p[H, :] - p_goal
    obj += w_pt * (p_err_terminal @ p_err_terminal)

    # Tracking + effort cost over horizon
    for k in range(H):
        # w_p * ||p(k) - p_goal||^2
        p_err = p[k, :] - p_goal
        obj += w_p * (p_err @ p_err)
        # w_u * ||u(k)||^2
        obj += w_u * (u[k, :] @ u[k, :])

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"QP solve failed, status: {model.status}")

    # Sonuclari cikart
    p_traj = p.X  # shape (H+1, 2)
    v_traj = v.X  # shape (H+1, 2)
    u_traj = u.X  # shape (H, 2)

    return p_traj, v_traj, u_traj


# --- Demo: bound'lu vs bound'suz karsilastirma ---
if __name__ == "__main__":
    # Parametreler (makaledeki degerler, Section V-A)
    tau = 0.2       # sampling period (s)
    H = 20          # horizon length
    w_pt = 10.0     # terminal cost weight
    w_p = 1.0       # tracking cost weight
    w_u = 1.0       # effort cost weight

    env_bounds = (0.0, 4.0, 0.0, 3.0)
    vmax = 0.5      # m/s  (makaledeki deger)
    amax = 0.5      # m/s^2 (makaledeki deger)

    p_init = np.array([0.5, 0.5])
    v_init = np.array([0.0, 0.0])
    p_goal = np.array([3.5, 2.5])

    # Iki durum: bound'suz ve bound'lu
    p1, v1, u1 = solve_single_robot_qp(
        p_init, v_init, p_goal, H, tau, w_pt=w_pt, w_p=w_p, w_u=w_u
    )
    p2, v2, u2 = solve_single_robot_qp(
        p_init, v_init, p_goal, H, tau,
        bounds=env_bounds, vmax=vmax, amax=amax,
        w_pt=w_pt, w_p=w_p, w_u=w_u
    )

    # --- Gorsellestirme ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    time = np.arange(H + 1) * tau
    time_u = np.arange(H) * tau

    # 1) Trajectory
    ax = axes[0]
    env = Environment(bounds=env_bounds)
    env.add_robot(position=p_init, goal=p_goal)
    env.plot(ax=ax)
    ax.plot(p1[:, 0], p1[:, 1], '-o', color='tab:blue',
            markersize=3, linewidth=2, alpha=0.5, label='unconstrained')
    ax.plot(p2[:, 0], p2[:, 1], '-s', color='tab:red',
            markersize=3, linewidth=2, alpha=0.7, label='constrained')
    ax.set_title('Trajectory Comparison')
    ax.legend()

    # 2) Speed profile
    ax = axes[1]
    ax.plot(time, np.linalg.norm(v1, axis=1), '-o', markersize=3,
            color='tab:blue', alpha=0.5, label='unconstrained')
    ax.plot(time, np.linalg.norm(v2, axis=1), '-s', markersize=3,
            color='tab:red', label='constrained')
    ax.axhline(y=vmax, color='gray', linestyle='--', label=f'vmax={vmax}')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('speed (m/s)')
    ax.set_title('Speed Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3) Control inputs (constrained only)
    ax = axes[2]
    ax.plot(time_u, u2[:, 0], '-o', markersize=3, label='ux')
    ax.plot(time_u, u2[:, 1], '-o', markersize=3, label='uy')
    ax.axhline(y=amax, color='gray', linestyle='--', label=f'amax={amax}')
    ax.axhline(y=-amax, color='gray', linestyle='--')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('acceleration (m/s^2)')
    ax.set_title('Control Inputs (constrained)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../plots/faz1_5_bounded_qp.png', dpi=150)
    plt.show()
    print("Bounded QP saved: plots/faz1_5_bounded_qp.png")
