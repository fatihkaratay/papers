"""
Faz 4.2-4.4: MICP coz, binary cozumleri topla, refine et, graf formatina cevir.

Akis:
  1. scenario_generator ile rastgele senaryo uret
  2. solve_multi_robot_micp ile MICP coz
  3. Binary refinement: ill-posed binary'leri duzelt
  4. Graf formatina cevir: node features + edge index + edge labels
  5. Tum sonuclari pickle ile diske yaz

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


def sample_to_graph(sample):
    """Raw sample'i GAT icin graf formatina cevir.

    Heterogeneous graf yapisi (Definition 3):
      Node tipleri:
        - robot:    feature = [px, py, gx, gy]           (4D)
        - obstacle: feature = [cx, cy, angle, L, W]      (5D)

      Edge tipleri (binary tahmin gerektiren):
        - RO: robot -> obstacle (her robot-engel cifti)
        - RR: robot -> robot    (proximity-based)

      Edge tipleri (sadece bilgi akisi, binary yok):
        - OR: obstacle -> robot (RO'nun tersi)
        - OO: obstacle -> obstacle (tum engel ciftleri)

    Her edge'in label'i: binary vektor, shape (H*4,)
      H zaman adimi x 4 yon = GAT'in o edge icin tahmin edecegi binary'ler.

    Returns:
        graph: dict with keys:
          - node_feat_robot:    (NR, 4) float
          - node_feat_obstacle: (NO, 5) float
          - edge_index_RO:      (2, NR*NO) int — [src_robot, dst_obstacle]
          - edge_index_RR:      (2, n_rr_edges) int — [src_robot, dst_robot]
          - edge_index_OR:      (2, NR*NO) int — bilgi akisi
          - edge_index_OO:      (2, NO*(NO-1)) int — bilgi akisi
          - edge_labels_RO:     (NR*NO, H*4) float — binary labels
          - edge_labels_RR:     (n_rr_edges, H*4) float — binary labels
          - H:                  int — horizon length
    """
    NR = sample['n_robots']
    NO = sample['n_obstacles']

    # === Node features ===
    # Robot: [px, py, gx, gy]
    robot_feats = np.zeros((NR, 4))
    for i in range(NR):
        robot_feats[i, :2] = sample['robots_p'][i]
        robot_feats[i, 2:] = sample['robots_goal'][i]

    # Obstacle: [cx, cy, angle, L, W]
    obs_feats = np.zeros((NO, 5))
    for o in range(NO):
        od = sample['obstacles'][o]
        obs_feats[o] = [od['center'][0], od['center'][1],
                        od['angle'], od['half_length'], od['half_width']]

    # === Edge index + labels: Robot-Obstacle (RO) ===
    # Her robot-engel cifti bir edge. Toplam NR*NO edge.
    # Node indexleme: robotlar 0..NR-1, engeller NR..NR+NO-1
    ro_src = []  # robot index (0..NR-1)
    ro_dst = []  # obstacle index (0..NO-1)
    ro_labels = []
    for entry in sample['b_obs']:
        ro_src.append(entry['robot'])
        ro_dst.append(entry['obstacle'])
        # (H, 4) -> (H*4,) flatten
        ro_labels.append(entry['binaries'].flatten())

    edge_index_RO = np.array([ro_src, ro_dst], dtype=int)  # (2, NR*NO)
    edge_labels_RO = np.array(ro_labels, dtype=float)       # (NR*NO, H*4)

    # === Edge index + labels: Robot-Robot (RR) ===
    rr_src = []
    rr_dst = []
    rr_labels = []
    for entry in sample['b_rob']:
        # Bidirectional: (i,j) ve (j,i) — ayni binary'ler
        rr_src.append(entry['robot_i'])
        rr_dst.append(entry['robot_j'])
        rr_labels.append(entry['binaries'].flatten())

    if rr_src:
        edge_index_RR = np.array([rr_src, rr_dst], dtype=int)
        edge_labels_RR = np.array(rr_labels, dtype=float)
    else:
        edge_index_RR = np.zeros((2, 0), dtype=int)
        edge_labels_RR = np.zeros((0, sample['b_obs'][0]['binaries'].size), dtype=float)

    # === Bilgi akisi edge'leri (label yok) ===
    # OR: obstacle -> robot (RO'nun tersi)
    or_src = []
    or_dst = []
    for o in range(NO):
        for i in range(NR):
            or_src.append(o)
            or_dst.append(i)
    edge_index_OR = np.array([or_src, or_dst], dtype=int)

    # OO: obstacle -> obstacle (tum ciftler)
    oo_src = []
    oo_dst = []
    for o1 in range(NO):
        for o2 in range(NO):
            if o1 != o2:
                oo_src.append(o1)
                oo_dst.append(o2)
    if oo_src:
        edge_index_OO = np.array([oo_src, oo_dst], dtype=int)
    else:
        edge_index_OO = np.zeros((2, 0), dtype=int)

    H = sample['b_obs'][0]['binaries'].shape[0]

    graph = {
        'node_feat_robot': robot_feats,
        'node_feat_obstacle': obs_feats,
        'edge_index_RO': edge_index_RO,
        'edge_index_RR': edge_index_RR,
        'edge_index_OR': edge_index_OR,
        'edge_index_OO': edge_index_OO,
        'edge_labels_RO': edge_labels_RO,
        'edge_labels_RR': edge_labels_RR,
        'H': H,
    }
    return graph


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

        # Bir sample'i graf formatina cevir ve goster
        s = dataset[0]
        g = sample_to_graph(s)
        print(f"\n=== Graf Formati (seed={s['seed']}) ===")
        print(f"  node_feat_robot:    shape {g['node_feat_robot'].shape}")
        print(f"  node_feat_obstacle: shape {g['node_feat_obstacle'].shape}")
        print(f"  edge_index_RO:      shape {g['edge_index_RO'].shape}")
        print(f"  edge_index_RR:      shape {g['edge_index_RR'].shape}")
        print(f"  edge_index_OR:      shape {g['edge_index_OR'].shape}")
        print(f"  edge_index_OO:      shape {g['edge_index_OO'].shape}")
        print(f"  edge_labels_RO:     shape {g['edge_labels_RO'].shape}")
        print(f"  edge_labels_RR:     shape {g['edge_labels_RR'].shape}")
        print(f"  H = {g['H']}")

        # Label dagilimi
        ro_ones = g['edge_labels_RO'].sum()
        ro_total = g['edge_labels_RO'].size
        print(f"\n  RO label dagilimi: "
              f"{int(ro_ones)} ones / {ro_total} total "
              f"({100*ro_ones/max(ro_total,1):.1f}% ones)")
        if g['edge_labels_RR'].size > 0:
            rr_ones = g['edge_labels_RR'].sum()
            rr_total = g['edge_labels_RR'].size
            print(f"  RR label dagilimi: "
                  f"{int(rr_ones)} ones / {rr_total} total "
                  f"({100*rr_ones/max(rr_total,1):.1f}% ones)")

        # Ornek robot ve engel feature'lari
        print(f"\n  Robot features (ilk 2):")
        for i in range(min(2, g['node_feat_robot'].shape[0])):
            f = g['node_feat_robot'][i]
            print(f"    R{i}: pos=({f[0]:.2f},{f[1]:.2f}) "
                  f"goal=({f[2]:.2f},{f[3]:.2f})")
        print(f"  Obstacle features (ilk 2):")
        for o in range(min(2, g['node_feat_obstacle'].shape[0])):
            f = g['node_feat_obstacle'][o]
            print(f"    O{o}: center=({f[0]:.2f},{f[1]:.2f}) "
                  f"angle={np.degrees(f[2]):.0f}deg L={f[3]:.2f} W={f[4]:.2f}")
