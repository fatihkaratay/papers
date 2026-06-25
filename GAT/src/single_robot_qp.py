import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from environment import Environment


def solve_single_robot_qp(p_init, v_init, p_goal, H, tau,
                           bounds=None, vmax=None, amax=None,
                           w_pt=10.0, w_p=1.0, w_u=1.0):
    model = gp.Model("single_robot_qp")
    model.setParam('OutputFlag', 0)  # GUROBI

    if bounds is not None:
        px_min, px_max, py_min, py_max = bounds
        p_lb = np.array([px_min, py_min])
        p_ub = np.array([px_max, py_max])
        p = model.addMVar((H + 1, 2), lb=p_lb, ub=p_ub, name="p")
    else:
        p = model.addMVar((H + 1, 2), lb=-GRB.INFINITY, name="p")

    # v[k, dim] for k=0..H
    if vmax is not None:
        v = model.addMVar((H + 1, 2), lb=-vmax, ub=vmax, name="v")
    else:
        v = model.addMVar((H + 1, 2), lb=-GRB.INFINITY, name="v")

    # u[k, dim] for k=0..H-1
    if amax is not None:
        u = model.addMVar((H, 2), lb=-amax, ub=amax, name="u")
    else:
        u = model.addMVar((H, 2), lb=-GRB.INFINITY, name="u")

    model.addConstr(p[0, :] == p_init, name="p_init")
    model.addConstr(v[0, :] == v_init, name="v_init")

    for k in range(H):
        model.addConstr(
            p[k + 1, :] == p[k, :] + tau * v[k, :] + 0.5 * tau**2 * u[k, :],
            name=f"dyn_p_{k}"
        )
        model.addConstr(
            v[k + 1, :] == v[k, :] + tau * u[k, :],
            name=f"dyn_v_{k}"
        )

    obj = 0

    p_err_terminal = p[H, :] - p_goal
    obj += w_pt * (p_err_terminal @ p_err_terminal)

    for k in range(H):
        p_err = p[k, :] - p_goal
        obj += w_p * (p_err @ p_err)
        obj += w_u * (u[k, :] @ u[k, :])

    model.setObjective(obj, GRB.MINIMIZE)
    model.optimize()

    if model.status != GRB.OPTIMAL:
        raise RuntimeError(f"QP solve failed, status: {model.status}")

    p_traj = p.X  # shape (H+1, 2)
    v_traj = v.X  # shape (H+1, 2)
    u_traj = u.X  # shape (H, 2)

    return p_traj, v_traj, u_traj


if __name__ == "__main__":
    tau = 0.2       # sampling period (s)
    H = 20          # horizon length
    w_pt = 10.0     # terminal cost weight
    w_p = 1.0       # tracking cost weight
    w_u = 1.0       # effort cost weight

    env_bounds = (0.0, 4.0, 0.0, 3.0)
    vmax = 0.5      # m/s 
    amax = 0.5      # m/s^2

    p_init = np.array([0.5, 0.5])
    v_init = np.array([0.0, 0.0])
    p_goal = np.array([3.5, 2.5])

    p1, v1, u1 = solve_single_robot_qp(
        p_init, v_init, p_goal, H, tau, w_pt=w_pt, w_p=w_p, w_u=w_u
    )
    p2, v2, u2 = solve_single_robot_qp(
        p_init, v_init, p_goal, H, tau,
        bounds=env_bounds, vmax=vmax, amax=amax,
        w_pt=w_pt, w_p=w_p, w_u=w_u
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    time = np.arange(H + 1) * tau
    time_u = np.arange(H) * tau

    # Trajectory
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

    # Speed profile
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

    # Control inputs (constrained only)
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
    plt.savefig('../plots/phase1_5_bounded_qp.png', dpi=150)
    plt.show()
    print("Bounded QP saved: plots/phase1_5_bounded_qp.png")
