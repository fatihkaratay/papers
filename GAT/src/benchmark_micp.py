"""
Faz 3.5: Robot sayisi arttikca MICP cozum suresini olcme.

Amac: Neden GAT'e ihtiyac var? Cunku MICP cozum suresi robot sayisiyla
hizla artiyor. Binary degisken sayisi patlayinca GUROBI bile yavaslliyor.
Bu script bunu somut olarak gosteriyor.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from environment import Environment
from multi_robot_micp import solve_multi_robot_micp


def create_scenario(n_robots, n_obstacles=2, seed=42):
    """N robot + M engel senaryosu olustur.

    Robotlar daire uzerinde yerlesiyor, hedefleri karsida.
    Boylece yollar kesisiyor ve collision avoidance gerekiyor.
    """
    rng = np.random.RandomState(seed)
    env = Environment(bounds=(-4.0, 4.0, -4.0, 4.0))

    # Engelleri merkez civarina yerlestir
    for _ in range(n_obstacles):
        cx = rng.uniform(-1.0, 1.0)
        cy = rng.uniform(-1.0, 1.0)
        env.add_obstacle(center=(cx, cy),
                         half_length=rng.uniform(0.2, 0.4),
                         half_width=rng.uniform(0.15, 0.3),
                         angle=rng.uniform(0, np.pi))

    # Robotlar daire uzerinde, hedefler karsida
    radius = 3.0
    for i in range(n_robots):
        angle = 2 * np.pi * i / n_robots
        px = radius * np.cos(angle)
        py = radius * np.sin(angle)
        # Hedef: dairenin karsi tarafi
        gx = -px
        gy = -py
        env.add_robot(position=(px, py), goal=(gx, gy))

    return env


def count_variables(n_robots, n_obstacles, H):
    """Degisken sayilarini hesapla."""
    # Her robot: (H+1)*2 pozisyon + (H+1)*2 hiz + H*2 kontrol
    continuous = n_robots * ((H + 1) * 2 + (H + 1) * 2 + H * 2)
    # Robot-engel binary: n_robots * n_obstacles * H * 4
    bin_obs = n_robots * n_obstacles * H * 4
    # Robot-robot binary (worst case, tum ciftler): C(n,2) * H * 4
    n_pairs = n_robots * (n_robots - 1) // 2
    bin_rob = n_pairs * H * 4
    return continuous, bin_obs, bin_rob


def run_benchmark():
    H = 15
    tau = 0.2
    vmax = 0.5
    amax = 0.5
    dmin = 0.2
    dprox = 10.0  # tum robotlar arasi edge olsun

    robot_counts = [2, 3, 4, 5]
    n_obstacles = 2
    results = []

    print(f"{'N_robots':>8} {'N_obs':>5} {'Cont.vars':>10} {'Bin(obs)':>9} "
          f"{'Bin(rob)':>9} {'Total_bin':>10} {'Time(s)':>8} {'Status':>8}")
    print("-" * 78)

    for nr in robot_counts:
        env = create_scenario(nr, n_obstacles)

        robots_p = [r.position.copy() for r in env.robots]
        robots_v = [r.velocity.copy() for r in env.robots]
        robots_goal = [r.goal.copy() for r in env.robots]

        cont, b_obs, b_rob = count_variables(nr, n_obstacles, H)

        t0 = time.time()
        try:
            p_trajs, v_trajs, u_trajs, edges = solve_multi_robot_micp(
                robots_p, robots_v, robots_goal,
                env.obstacles, H, tau, env.bounds,
                vmax, amax, dmin, dprox
            )
            elapsed = time.time() - t0
            status = "OK"
        except Exception as e:
            elapsed = time.time() - t0
            status = "FAIL"
            edges = []

        results.append({
            'n_robots': nr,
            'continuous': cont,
            'bin_obs': b_obs,
            'bin_rob': b_rob,
            'total_bin': b_obs + b_rob,
            'time': elapsed,
            'status': status,
            'n_edges': len(edges),
        })

        print(f"{nr:>8} {n_obstacles:>5} {cont:>10} {b_obs:>9} "
              f"{b_rob:>9} {b_obs+b_rob:>10} {elapsed:>8.2f} {status:>8}")

    # --- Grafik ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    nrs = [r['n_robots'] for r in results]
    times = [r['time'] for r in results]
    total_bins = [r['total_bin'] for r in results]

    # Sol: cozum suresi
    ax1.bar(nrs, times, color='steelblue', alpha=0.8, width=0.6)
    for i, (n, t) in enumerate(zip(nrs, times)):
        ax1.text(n, t + 0.02, f'{t:.2f}s', ha='center', fontsize=10)
    ax1.set_xlabel('Number of Robots')
    ax1.set_ylabel('Solve Time (s)')
    ax1.set_title('MICP Solve Time vs Robot Count')
    ax1.set_xticks(nrs)
    ax1.grid(True, alpha=0.3, axis='y')

    # Sag: binary degisken sayisi
    b_obs_list = [r['bin_obs'] for r in results]
    b_rob_list = [r['bin_rob'] for r in results]
    ax2.bar(nrs, b_obs_list, color='tab:orange', alpha=0.8, width=0.6,
            label='Robot-Obstacle')
    ax2.bar(nrs, b_rob_list, bottom=b_obs_list, color='tab:red', alpha=0.8,
            width=0.6, label='Robot-Robot')
    for i, (n, tb) in enumerate(zip(nrs, total_bins)):
        ax2.text(n, tb + 5, str(tb), ha='center', fontsize=10)
    ax2.set_xlabel('Number of Robots')
    ax2.set_ylabel('Number of Binary Variables')
    ax2.set_title('Binary Variable Count vs Robot Count')
    ax2.set_xticks(nrs)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('../plots/faz3_benchmark.png', dpi=150)
    plt.show()
    print("\nSaved: plots/faz3_benchmark.png")


if __name__ == "__main__":
    run_benchmark()
