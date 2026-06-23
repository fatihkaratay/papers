"""
Faz 4.2-4.3: MICP coz, binary cozumleri topla ve refine et.

Akis:
  1. scenario_generator ile rastgele senaryo uret
  2. solve_multi_robot_micp ile MICP coz
  3. Binary refinement: ill-posed binary'leri duzelt
  4. Tum sonuclari pickle ile diske yaz

Refinement neden gerekli?
  GUROBI bir binary'yi b=1 (relaxed) verebilir, ama o kisit zaten
  saglaniyordur. Bu durumda b=0 da b=1 de optimal — GUROBI'nin
  umurunda degil. Ama GAT icin bu tutarsizlik yaratir.
  Cozum: b=1 ama kisit zaten saglaniyor → b=0'a cevir.
"""

import time
import pickle
import numpy as np
from scenario_generator import generate_scenario
from multi_robot_micp import solve_multi_robot_micp


def refine_obs_binaries(b_obs_sol, p_trajs, obstacles, dmin, NR):
    """Robot-engel binary'lerini refine et.

    Her (robot i, engel o, zaman k, yon m) icin:
      b=1 (relaxed) ama kisit zaten saglaniyor → b=0 yap.

    Kisitlar (Denklem 6):
      m=0 (sag):   cos(a)*(px-ox) + sin(a)*(py-oy) >= L + dmin
      m=1 (ust):  -sin(a)*(px-ox) + cos(a)*(py-oy) >= W + dmin
      m=2 (sol):  -cos(a)*(px-ox) - sin(a)*(py-oy) >= L + dmin
      m=3 (alt):   sin(a)*(px-ox) - cos(a)*(py-oy) >= W + dmin

    Returns:
        b_refined: ayni yapida dict, refine edilmis binary'ler
        stats: kac binary 1'den 0'a cevirildi
    """
    b_refined = {}
    total_flipped = 0
    total_checked = 0

    for i in range(NR):
        for o_idx, obs in enumerate(obstacles):
            b = b_obs_sol[i, o_idx].copy()  # (H, 4)
            H_len = b.shape[0]

            ca = np.cos(obs.angle)
            sa = np.sin(obs.angle)
            ox, oy = obs.center
            L = obs.half_length
            W = obs.half_width
            # Her yon icin: RHS (saglanmasi gereken esik)
            rhs = [L + dmin, W + dmin, L + dmin, W + dmin]

            for k in range(H_len):
                pk = k + 1  # p_trajs[i][pk] = pozisyon at time k+1
                px = p_trajs[i][pk, 0]
                py = p_trajs[i][pk, 1]

                # 4 yonun LHS degerlerini hesapla
                dx = px - ox
                dy = py - oy
                lhs = [
                    ca * dx + sa * dy,   # m=0: sag
                    -sa * dx + ca * dy,  # m=1: ust
                    -ca * dx - sa * dy,  # m=2: sol
                    sa * dx - ca * dy,   # m=3: alt
                ]

                for m in range(4):
                    total_checked += 1
                    if b[k, m] == 1 and lhs[m] >= rhs[m] - 1e-6:
                        # Kisit zaten saglaniyor, b=0 yapabiliriz
                        b[k, m] = 0
                        total_flipped += 1

            b_refined[i, o_idx] = b

    return b_refined, total_flipped, total_checked


def refine_rob_binaries(b_rob_sol, p_trajs, robot_edges, dmin):
    """Robot-robot binary'lerini refine et.

    Kisitlar (Denklem 4):
      m=0: px_i - px_j >= 2*dmin
      m=1: px_j - px_i >= 2*dmin
      m=2: py_i - py_j >= 2*dmin
      m=3: py_j - py_i >= 2*dmin

    Returns:
        b_refined, total_flipped, total_checked
    """
    b_refined = {}
    total_flipped = 0
    total_checked = 0
    threshold = 2 * dmin

    for (i, j) in robot_edges:
        b = b_rob_sol[i, j].copy()  # (H, 4)
        H_len = b.shape[0]

        for k in range(H_len):
            pk = k + 1
            dx = p_trajs[i][pk, 0] - p_trajs[j][pk, 0]
            dy = p_trajs[i][pk, 1] - p_trajs[j][pk, 1]

            lhs = [dx, -dx, dy, -dy]

            for m in range(4):
                total_checked += 1
                if b[k, m] == 1 and lhs[m] >= threshold - 1e-6:
                    b[k, m] = 0
                    total_flipped += 1

        b_refined[i, j] = b

    return b_refined, total_flipped, total_checked


def solve_and_collect(env, H, tau, vmax, amax, dmin, dprox, M=100.0):
    """Bir senaryo icin MICP coz ve tum sonuclari topla.

    Args:
        env: Environment nesnesi (robotlar ve engeller yerlestirilmis)
        H, tau, vmax, amax, dmin, dprox, M: MICP parametreleri

    Returns:
        sample: dict — senaryo + cozum bilgileri, veya None (cozum bulunamazsa)
    """
    NR = len(env.robots)
    NO = len(env.obstacles)

    robots_p = [r.position.copy() for r in env.robots]
    robots_v = [r.velocity.copy() for r in env.robots]
    robots_goal = [r.goal.copy() for r in env.robots]

    t0 = time.time()
    try:
        p_trajs, v_trajs, u_trajs, robot_edges, b_obs_sol, b_rob_sol = \
            solve_multi_robot_micp(
                robots_p, robots_v, robots_goal,
                env.obstacles, H, tau, env.bounds,
                vmax, amax, dmin, dprox, M
            )
    except Exception as e:
        return None
    solve_time = time.time() - t0

    # === Binary refinement (Faz 4.3) ===
    b_obs_sol, obs_flipped, obs_checked = refine_obs_binaries(
        b_obs_sol, p_trajs, env.obstacles, dmin, NR)
    b_rob_sol, rob_flipped, rob_checked = refine_rob_binaries(
        b_rob_sol, p_trajs, robot_edges, dmin)

    # Engel bilgilerini topla
    obstacle_data = []
    for obs in env.obstacles:
        obstacle_data.append({
            'center': obs.center.copy(),
            'half_length': obs.half_length,
            'half_width': obs.half_width,
            'angle': obs.angle,
        })

    # Robot-engel binary'leri: her (robot_i, obstacle_o) cifti icin (H, 4)
    b_obs_list = []
    for i in range(NR):
        for o in range(NO):
            b_obs_list.append({
                'robot': i,
                'obstacle': o,
                'binaries': b_obs_sol[i, o],  # (H, 4)
            })

    # Robot-robot binary'leri: her (i, j) edge icin (H, 4)
    b_rob_list = []
    for (i, j) in robot_edges:
        b_rob_list.append({
            'robot_i': i,
            'robot_j': j,
            'binaries': b_rob_sol[i, j],  # (H, 4)
        })

    sample = {
        # Senaryo
        'n_robots': NR,
        'n_obstacles': NO,
        'bounds': env.bounds,
        'robots_p': robots_p,
        'robots_v': robots_v,
        'robots_goal': robots_goal,
        'obstacles': obstacle_data,

        # Cozum
        'p_trajs': p_trajs,
        'v_trajs': v_trajs,
        'u_trajs': u_trajs,
        'solve_time': solve_time,

        # Binary cozumler (GAT'in ogrenecegi hedef, refined)
        'robot_edges': robot_edges,
        'b_obs': b_obs_list,
        'b_rob': b_rob_list,

        # Refinement istatistikleri
        'refinement': {
            'obs_flipped': obs_flipped,
            'obs_checked': obs_checked,
            'rob_flipped': rob_flipped,
            'rob_checked': rob_checked,
        },
    }
    return sample


def collect_dataset(n_samples, H=15, tau=0.2, vmax=0.5, amax=0.5,
                    dmin=0.2, dprox=5.0,
                    n_robots_range=(2, 5), n_obstacles_range=(1, 3),
                    bounds=(-4.0, 4.0, -4.0, 4.0),
                    base_seed=0, save_path=None):
    """Birden fazla senaryo coz ve dataset olustur.

    Args:
        n_samples: kac senaryo uretilip cozulecek
        save_path: sonuclarin kaydedilecegi dosya yolu (pickle)
        (diger args: MICP ve senaryo parametreleri)

    Returns:
        dataset: list of sample dicts
    """
    dataset = []
    failed = 0

    print(f"Veri toplama: {n_samples} senaryo")
    print(f"  Robotlar: {n_robots_range}, Engeller: {n_obstacles_range}")
    print(f"  H={H}, tau={tau}, dprox={dprox}")
    print(f"{'':>4} {'NR':>3} {'NO':>3} {'Edges':>5} {'Time':>7} {'Status':>7}")
    print("-" * 35)

    for i in range(n_samples):
        rng = np.random.RandomState(base_seed + i)
        nr = rng.randint(n_robots_range[0], n_robots_range[1] + 1)
        no = rng.randint(n_obstacles_range[0], n_obstacles_range[1] + 1)

        # Senaryo uret
        try:
            env = generate_scenario(nr, no, bounds, dmin, rng)
        except RuntimeError:
            failed += 1
            print(f"{i+1:>4} {nr:>3} {no:>3} {'---':>5} {'---':>7} {'GEN_FAIL':>7}")
            continue

        # MICP coz
        sample = solve_and_collect(env, H, tau, vmax, amax, dmin, dprox)

        if sample is None:
            failed += 1
            print(f"{i+1:>4} {nr:>3} {no:>3} {'---':>5} {'---':>7} {'SOLVE_FAIL':>7}")
            continue

        sample['seed'] = base_seed + i
        dataset.append(sample)

        n_edges = len(sample['robot_edges'])
        st = sample['solve_time']
        print(f"{i+1:>4} {nr:>3} {no:>3} {n_edges:>5} {st:>6.2f}s {'OK':>7}")

    print(f"\nToplam: {len(dataset)}/{n_samples} basarili, {failed} basarisiz")

    # Istatistikler
    if dataset:
        times = [s['solve_time'] for s in dataset]
        print(f"Cozum suresi: min={min(times):.2f}s, max={max(times):.2f}s, "
              f"mean={np.mean(times):.2f}s")

        n_obs_bins = sum(len(s['b_obs']) for s in dataset)
        n_rob_bins = sum(len(s['b_rob']) for s in dataset)
        print(f"Binary edge toplami: {n_obs_bins} robot-engel, "
              f"{n_rob_bins} robot-robot")

    # Kaydet
    if save_path and dataset:
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)
        print(f"\nDataset kaydedildi: {save_path}")

    return dataset


# --- Demo: kucuk bir dataset topla ---
if __name__ == "__main__":
    dataset = collect_dataset(
        n_samples=10,
        H=15,
        tau=0.2,
        n_robots_range=(2, 3),
        n_obstacles_range=(1, 2),
        base_seed=42,
        save_path='../data/demo_dataset.pkl',
    )

    # Refinement istatistikleri
    if dataset:
        print(f"\n=== Refinement Istatistikleri ===")
        total_obs_flipped = 0
        total_obs_checked = 0
        total_rob_flipped = 0
        total_rob_checked = 0
        for s in dataset:
            r = s['refinement']
            total_obs_flipped += r['obs_flipped']
            total_obs_checked += r['obs_checked']
            total_rob_flipped += r['rob_flipped']
            total_rob_checked += r['rob_checked']

        print(f"  Robot-engel: {total_obs_flipped}/{total_obs_checked} binary "
              f"1->0 cevirildi "
              f"({100*total_obs_flipped/max(total_obs_checked,1):.1f}%)")
        if total_rob_checked > 0:
            print(f"  Robot-robot: {total_rob_flipped}/{total_rob_checked} binary "
                  f"1->0 cevirildi "
                  f"({100*total_rob_flipped/max(total_rob_checked,1):.1f}%)")

        # Bir sample detayi
        s = dataset[0]
        print(f"\n=== Ornek sample (seed={s['seed']}) ===")
        print(f"  Robotlar: {s['n_robots']}, Engeller: {s['n_obstacles']}")
        print(f"  Robot-robot edges: {s['robot_edges']}")
        print(f"  Ornek binary (robot0-obs0, ilk 5 adim):")
        print(f"    {s['b_obs'][0]['binaries'][:5]}")
        r = s['refinement']
        print(f"  Refinement: {r['obs_flipped']}+{r['rob_flipped']} "
              f"binary cevirildi")
