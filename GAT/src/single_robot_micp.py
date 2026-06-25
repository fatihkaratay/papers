import numpy as np
import matplotlib.pyplot as plt
import gurobipy as gp
from gurobipy import GRB
from environment import Environment


def solve_single_robot_micp(p_init, v_init, p_goal, obstacles, H, tau,
                             bounds, vmax, amax, dmin,
                             M=100.0, w_pt=10.0, w_p=1.0, w_u=1.0):
    model = gp.Model("single_robot_micp")
    model.setParam('OutputFlag', 0)

    px_min, px_max, py_min, py_max = bounds

    p = model.addMVar((H + 1, 2),
                       lb=[px_min, py_min], ub=[px_max, py_max], name="p")
    v = model.addMVar((H + 1, 2), lb=-vmax, ub=vmax, name="v")
    u = model.addMVar((H, 2), lb=-amax, ub=amax, name="u")

    n_obs = len(obstacles)
    b = model.addMVar((n_obs, H, 4), vtype=GRB.BINARY, name="b")

    model.addConstr(p[0, :] == p_init, name="p_init")
    model.addConstr(v[0, :] == v_init, name="v_init")

    for k in range(H):
        model.addConstr(
            p[k+1, :] == p[k, :] + tau * v[k, :] + 0.5 * tau**2 * u[k, :],
            name=f"dyn_p_{k}")
        model.addConstr(
            v[k+1, :] == v[k, :] + tau * u[k, :],
            name=f"dyn_v_{k}")

    for o_idx, obs in enumerate(obstacles):
        ca = np.cos(obs.angle)
        sa = np.sin(obs.angle)
        ox, oy = obs.center
        L = obs.half_length
        W = obs.half_width

        for k in range(H):  # k=0..H-1 → p[k+1] icin (k+1 = 1..H)
            pk = k + 1
            
            model.addConstr(
                ca * (p[pk, 0] - ox) + sa * (p[pk, 1] - oy)
                >= L + dmin - M * b[o_idx, k, 0],
                name=f"obs{o_idx}_k{k}_right")

            model.addConstr(
                -sa * (p[pk, 0] - ox) + ca * (p[pk, 1] - oy)
                >= W + dmin - M * b[o_idx, k, 1],
                name=f"obs{o_idx}_k{k}_top")

            model.addConstr(
                -ca * (p[pk, 0] - ox) - sa * (p[pk, 1] - oy)
                >= L + dmin - M * b[o_idx, k, 2],
                name=f"obs{o_idx}_k{k}_left")

            model.addConstr(
                sa * (p[pk, 0] - ox) - ca * (p[pk, 1] - oy)
                >= W + dmin - M * b[o_idx, k, 3],
                name=f"obs{o_idx}_k{k}_bottom")

            model.addConstr(
                b[o_idx, k, 0] + b[o_idx, k, 1]
                + b[o_idx, k, 2] + b[o_idx, k, 3] <= 3,
                name=f"obs{o_idx}_k{k}_sum")

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
        print(f"Warning: MICP status = {model.status}")

    p_traj = p.X
    v_traj = v.X
    u_traj = u.X
    binaries = b.X

    return p_traj, v_traj, u_traj, binaries


if __name__ == "__main__":
    tau = 0.2
    H = 20
    env_bounds = (-3.0, 0.5, -0.5, 2.0)
    vmax = 0.5
    amax = 0.5
    dmin = 0.2

    env = Environment(bounds=env_bounds)
    obs1 = env.add_obstacle(center=(-1.5, 0.8), half_length=0.3, half_width=0.2)
    obs2 = env.add_obstacle(center=(-0.5, 1.2), half_length=0.2, half_width=0.15,
                             angle=np.radians(20))

    p_init = np.array([-2.5, 0.3])
    v_init = np.array([0.0, 0.0])
    p_goal = np.array([0.0, 1.5])
    env.add_robot(position=p_init, goal=p_goal)

    # MICP
    p_traj, v_traj, u_traj, binaries = solve_single_robot_micp(
        p_init, v_init, p_goal, env.obstacles, H, tau,
        env_bounds, vmax, amax, dmin
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    time = np.arange(H + 1) * tau
    time_u = np.arange(H) * tau

    ax = axes[0]
    env.plot(ax=ax)
    ax.plot(p_traj[:, 0], p_traj[:, 1], '-o', color='tab:blue',
            markersize=3, linewidth=2, alpha=0.7, label='MICP trajectory')
    ax.set_title('MICP Trajectory (with obstacles)')
    ax.legend()

    ax = axes[1]
    speeds = np.linalg.norm(v_traj, axis=1)
    ax.plot(time, speeds, '-o', markersize=3)
    ax.axhline(y=vmax, color='gray', linestyle='--', label=f'vmax={vmax}')
    ax.set_xlabel('time (s)')
    ax.set_ylabel('speed (m/s)')
    ax.set_title('Speed Profile')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    labels = ['right', 'top', 'left', 'bottom']
    for m in range(4):
        ax.step(time_u, binaries[0, :, m], where='mid', label=labels[m], alpha=0.7)
    ax.set_xlabel('time (s)')
    ax.set_ylabel('binary value')
    ax.set_title('Binary Decisions (obstacle 1)')
    ax.set_yticks([0, 1])
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../plots/phase2_micp_trajectory.png', dpi=150)
    plt.show()
    print("MICP trajectory saved: plots/phase2_micp_trajectory.png")
