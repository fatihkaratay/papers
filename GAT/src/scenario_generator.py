"""
Faz 4.1: Rastgele senaryo uretici — GAT egitimi icin dataset olusturmanin ilk adimi.

Amac: N robot + M engel iceren, fiziksel olarak gecerli (overlap yok) ve
cesitli senaryolar uretmek. Her senaryo bir MICP problemi tanimlayacak,
GUROBI ile cozulecek, ve binary cozumler GAT'in egitim verisi olacak.

Gecerlilik kurallari:
  1. Robotlar engellerin icinde baslamaz / bitmez
  2. Robotlar birbirinin ustunde baslamaz (min 2*dmin mesafe)
  3. Engeller birbirinin ustune binmez
  4. Her sey ortam sinirlari icinde kalir
"""

import numpy as np
from environment import Environment


def _point_inside_obstacle(point, obs, margin=0.0):
    """Bir nokta engelin icinde mi? (margin kadar buyutulmus engel.)

    Engel dondurulmus dikdortgen. Noktayi engelin lokal koordinatina
    cevirip kontrol ediyoruz:
      1. Noktayi engel merkezine gore otelele
      2. Engelin aci kadar ters dondir
      3. Lokal koordinatta |x| < L+margin VE |y| < W+margin ise icindedir
    """
    dx = point[0] - obs.center[0]
    dy = point[1] - obs.center[1]

    # Ters rotasyon (engelin lokal frame'ine gec)
    ca = np.cos(-obs.angle)
    sa = np.sin(-obs.angle)
    local_x = ca * dx - sa * dy
    local_y = sa * dx + ca * dy

    return (abs(local_x) < obs.half_length + margin and
            abs(local_y) < obs.half_width + margin)


def _obstacles_overlap(obs1, obs2, margin=0.1):
    """Iki engel arasinda yeterli bosluk var mi? (basitlesitirilmis kontrol.)

    Tam dikdortgen-dikdortgen kesisim kontrolu karmasik. Burada
    merkezler arasi mesafeyi bounding circle'lar ile kontrol ediyoruz:
    her engelin bounding radius = sqrt(L^2 + W^2).
    """
    r1 = np.sqrt(obs1.half_length**2 + obs1.half_width**2)
    r2 = np.sqrt(obs2.half_length**2 + obs2.half_width**2)
    dist = np.linalg.norm(obs1.center - obs2.center)
    return dist < r1 + r2 + margin


def generate_scenario(n_robots, n_obstacles, bounds, dmin=0.2,
                      rng=None, max_attempts=200):
    """Rastgele gecerli bir senaryo uret.

    Args:
        n_robots: robot sayisi
        n_obstacles: engel sayisi
        bounds: (px_min, px_max, py_min, py_max) — ortam sinirlari
        dmin: robot guvenli yaricap (m)
        rng: numpy RandomState (tekrarlanabilirlik icin)
        max_attempts: her eleman icin max deneme sayisi

    Returns:
        env: Environment nesnesi (robotlar ve engeller yerlestirilmis)

    Raises:
        RuntimeError: gecerli yerlestirme bulunamazsa
    """
    if rng is None:
        rng = np.random.RandomState()

    px_min, px_max, py_min, py_max = bounds
    # Sinir icinde kalabilmek icin kenar payı birak
    pad = 0.3
    env = Environment(bounds=bounds)

    # === 1. Engelleri yerlestir ===
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

            # Mevcut engellerle cakisiyor mu?
            overlap = False
            for prev_obs in env.obstacles[:-1]:
                if _obstacles_overlap(obs, prev_obs):
                    overlap = True
                    break

            if overlap:
                env.obstacles.pop()  # geri al, tekrar dene
            else:
                placed = True
                break

        if not placed:
            raise RuntimeError(
                f"Engel yerlestirilemedi ({n_obstacles} engel, "
                f"{max_attempts} deneme)")

    # === 2. Robotlari yerlestir (baslangic + hedef) ===
    for _ in range(n_robots):
        placed = False
        for attempt in range(max_attempts):
            # Rastgele baslangic pozisyonu
            px = rng.uniform(px_min + pad, px_max - pad)
            py = rng.uniform(py_min + pad, py_max - pad)
            start = np.array([px, py])

            # Rastgele hedef pozisyonu
            gx = rng.uniform(px_min + pad, px_max - pad)
            gy = rng.uniform(py_min + pad, py_max - pad)
            goal = np.array([gx, gy])

            # Kontrol 1: Engellerin icinde mi?
            inside_obs = False
            for obs in env.obstacles:
                if (_point_inside_obstacle(start, obs, margin=dmin) or
                        _point_inside_obstacle(goal, obs, margin=dmin)):
                    inside_obs = True
                    break
            if inside_obs:
                continue

            # Kontrol 2: Mevcut robotlara cok yakin mi?
            too_close = False
            for robot in env.robots:
                if np.linalg.norm(start - robot.position) < 2 * dmin:
                    too_close = True
                    break
            if too_close:
                continue

            # Kontrol 3: Baslangic-hedef arasi anlamli bir mesafe olsun
            # (cok yakin olursa trivial problem olur, GAT icin faydasz)
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
    """Birden fazla senaryo uret.

    Args:
        n_scenarios: kac senaryo uretilecek
        n_robots_range: (min, max) robot sayisi (dahil)
        n_obstacles_range: (min, max) engel sayisi (dahil)
        bounds: ortam sinirlari
        dmin: robot guvenli yaricap
        base_seed: ilk seed (her senaryo base_seed + i kullanir)

    Returns:
        scenarios: list of Environment nesneleri
    """
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


# --- Demo ---
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("=== Rastgele Senaryo Uretici Demo ===\n")

    # Tek senaryo ornegi
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

    # Batch ornegi
    print("\n--- Batch: 12 senaryo ---")
    scenarios = generate_batch(12, n_robots_range=(2, 4),
                               n_obstacles_range=(1, 3),
                               base_seed=100)
    for i, sc in enumerate(scenarios):
        print(f"  Senaryo {i+1}: {len(sc.robots)} robot, "
              f"{len(sc.obstacles)} engel")

    # 6 tanesini gorsellelstir
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    for idx, ax in enumerate(axes.flat):
        if idx < len(scenarios):
            scenarios[idx].plot(ax=ax)
            ax.set_title(f'Senaryo {idx+1}: '
                         f'{len(scenarios[idx].robots)}R + '
                         f'{len(scenarios[idx].obstacles)}O')
        else:
            ax.axis('off')

    plt.suptitle('Rastgele Uretilmis Senaryolar (Faz 4.1)', fontsize=14)
    plt.tight_layout()
    plt.savefig('../plots/faz4_random_scenarios.png', dpi=150)
    plt.show()
    print("\nSaved: plots/faz4_random_scenarios.png")
