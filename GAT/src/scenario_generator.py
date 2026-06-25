import numpy as np
from environment import Environment


def _point_inside_obstacle(point, obs, margin=0.0):

    dx = point[0] - obs.center[0]
    dy = point[1] - obs.center[1]

    ca = np.cos(-obs.angle)
    sa = np.sin(-obs.angle)
    local_x = ca * dx - sa * dy
    local_y = sa * dx + ca * dy

    return (abs(local_x) < obs.half_length + margin and
            abs(local_y) < obs.half_width + margin)


def _obstacles_overlap(obs1, obs2, margin=0.1):
    r1 = np.sqrt(obs1.half_length**2 + obs1.half_width**2)
    r2 = np.sqrt(obs2.half_length**2 + obs2.half_width**2)
    dist = np.linalg.norm(obs1.center - obs2.center)
    return dist < r1 + r2 + margin


def generate_scenario(n_robots, n_obstacles, bounds, dmin=0.2,
                      rng=None, max_attempts=200):
    if rng is None:
        rng = np.random.RandomState()

    px_min, px_max, py_min, py_max = bounds
    pad = 0.3
    env = Environment(bounds=bounds)

    for _ in range(n_obstacles):
        placed = False
        for attempt in range(max_attempts):
            cx = rng.uniform(px_min + pad, px_max - pad)
            cy = rng.uniform(py_min + pad, py_max - pad)
            half_l = rng.uniform(0.15, 0.4)
            half_w = rng.uniform(0.1, 0.3)
            angle = rng.uniform(0, np.pi)

            obs = env.add_obstacle(center=(cx, cy),
                                       half_length=half_l,
                                       half_width=half_w,
                                       angle=angle)

            overlap = False
            for prev_obs in env.obstacles[:-1]:
                if _obstacles_overlap(obs, prev_obs):
                    overlap = True
                    break

            if overlap:
                env.obstacles.pop()
            else:
                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"Engel yerlestirilemedi ({n_obstacles} engel, "
                f"{max_attempts} deneme)")

    for _ in range(n_robots):
        placed = False
        for attempt in range(max_attempts):
            px = rng.uniform(px_min + pad, px_max - pad)
            py = rng.uniform(py_min + pad, py_max - pad)
            start = np.array([px, py])

            gx = rng.uniform(px_min + pad, px_max - pad)
            gy = rng.uniform(py_min + pad, py_max - pad)
            goal = np.array([gx, gy])

            inside_obs = False
            for obs in env.obstacles:
                if (_point_inside_obstacle(start, obs, margin=dmin) or
                        _point_inside_obstacle(goal, obs, margin=dmin)):
                    inside_obs = True
                    break
            if inside_obs:
                continue

            too_close = False
            for robot in env.robots:
                if np.linalg.norm(start - robot.position) < 2 * dmin:
                    too_close = True
                    break
            if too_close:
                continue

            if np.linalg.norm(start - goal) < 1.0:
                continue

            env.add_robot(position=start, goal=goal)
            placed = True
            break

        if not placed:
            raise RuntimeError(
                f"Robot yerlestirilemedi ({n_robots} robot, "
                f"{max_attempts} deneme)")

    return env


def generate_batch(n_scenarios, n_robots_range=(2, 5),
                   n_obstacles_range=(1, 3),
                   bounds=(-4.0, 4.0, -4.0, 4.0),
                   dmin=0.2, base_seed=0):
    scenarios = []
    failed = 0

    for i in range(n_scenarios):
        rng = np.random.RandomState(base_seed + i)
        nr = rng.randint(n_robots_range[0], n_robots_range[1] + 1)
        no = rng.randint(n_obstacles_range[0], n_obstacles_range[1] + 1)

        try:
            env = generate_scenario(nr, no, bounds, dmin, rng)
            scenarios.append(env)
        except RuntimeError as e:
            failed += 1

    if failed > 0:
        print(f"Uyari: {failed}/{n_scenarios} senaryo uretilemedi.")

    return scenarios


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=== Rastgele Senaryo Uretici Demo ===\n")

    bounds = (-4.0, 4.0, -4.0, 4.0)
    rng = np.random.RandomState(42)
    env = generate_scenario(n_robots=3, n_obstacles=2, bounds=bounds,
                            dmin=0.2, rng=rng)

    print(f"Robotlar ({len(env.robots)}):")
    for i, r in enumerate(env.robots):
        dist = np.linalg.norm(r.position - r.goal)
        print(f"  R{i+1}: {r.position} -> {r.goal}  (mesafe: {dist:.2f}m)")

    print(f"\nEngeller ({len(env.obstacles)}):")
    for i, o in enumerate(env.obstacles):
        print(f"  O{i+1}: {o}")

    print("\n--- Batch: 12 senaryo ---")
    scenarios = generate_batch(12, n_robots_range=(2, 4),
                               n_obstacles_range=(1, 3),
                               base_seed=100)
    for i, sc in enumerate(scenarios):
        print(f"  Senaryo {i+1}: {len(sc.robots)} robot, "
              f"{len(sc.obstacles)} engel")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for idx, ax in enumerate(axes.flat):
        if idx < len(scenarios):
            scenarios[idx].plot(ax=ax)
            ax.set_title(f'Senaryo {idx+1}: '
                         f'{len(scenarios[idx].robots)}R + '
                         f'{len(scenarios[idx].obstacles)}O')
        else:
            ax.axis('off')

    plt.suptitle('Rastgele Uretilmis Senaryolar (phase 4.1)', fontsize=14)
    plt.tight_layout()
    plt.savefig('../plots/phase4_random_scenarios.png', dpi=150)
    plt.show()
    print("\nSaved: plots/phase4_random_scenarios.png")
