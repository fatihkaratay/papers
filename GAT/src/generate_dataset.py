"""
Faz 4.5-4.6: Buyuk dataset uret ve train/validation split yap.

Kullanim:
    python generate_dataset.py              # 2000 sample, varsayilan ayarlar
    python generate_dataset.py --n 5000     # 5000 sample
    python generate_dataset.py --resume     # onceki calismayi devam ettir

Cikti:
    data/dataset_full.pkl   — tum sample'lar (raw)
    data/dataset_train.pkl  — %90 train split (graph formati)
    data/dataset_val.pkl    — %10 validation split (graph formati)
"""

import argparse
import os
import sys
import time
import pickle
import numpy as np

from scenario_generator import generate_scenario
from multi_robot_micp import solve_multi_robot_micp
from collect_data import refine_obs_binaries, refine_rob_binaries, sample_to_graph, solve_and_collect


def collect_dataset_large(n_samples, n_robots_range=(2, 5), n_obstacles_range=(1, 3),
                          H=15, tau=0.2, vmax=0.5, amax=0.5,
                          dmin=0.2, dprox=5.0,
                          bounds=(-4.0, 4.0, -4.0, 4.0),
                          base_seed=0, save_path=None, resume_path=None):
    """Buyuk dataset topla, periyodik olarak diske kaydet.

    Her 50 sample'da bir diske yazar, boylece crash olursa kayip az olur.
    """
    # Resume: onceki dataset'i yukle
    dataset = []
    start_idx = 0
    if resume_path and os.path.exists(resume_path):
        with open(resume_path, 'rb') as f:
            dataset = pickle.load(f)
        start_idx = len(dataset)
        print(f"Onceki calismadan devam: {start_idx} sample yuklendi")
        if start_idx >= n_samples:
            print("Zaten yeterli sample var!")
            return dataset

    failed = 0
    save_interval = 50  # her 50 sample'da kaydet

    print(f"\nVeri toplama: {n_samples} senaryo (baslangiC: {start_idx})")
    print(f"  Robotlar: {n_robots_range}, Engeller: {n_obstacles_range}")
    print(f"  H={H}, tau={tau}, dprox={dprox}")
    print(f"  Kayit: her {save_interval} sample'da -> {save_path}")
    print(f"\n{'#':>5} {'NR':>3} {'NO':>3} {'Edges':>5} {'Time':>7} {'Status':>10} {'Total':>6}")
    print("-" * 50)

    t_start = time.time()

    for i in range(start_idx, n_samples):
        rng = np.random.RandomState(base_seed + i)
        nr = rng.randint(n_robots_range[0], n_robots_range[1] + 1)
        no = rng.randint(n_obstacles_range[0], n_obstacles_range[1] + 1)

        # Senaryo uret
        try:
            env = generate_scenario(nr, no, bounds, dmin, rng)
        except RuntimeError:
            failed += 1
            print(f"{i+1:>5} {nr:>3} {no:>3} {'---':>5} {'---':>7} {'GEN_FAIL':>10} {len(dataset):>6}")
            continue

        # MICP coz + refine
        sample = solve_and_collect(env, H, tau, vmax, amax, dmin, dprox)

        if sample is None:
            failed += 1
            print(f"{i+1:>5} {nr:>3} {no:>3} {'---':>5} {'---':>7} {'SOLVE_FAIL':>10} {len(dataset):>6}")
            continue

        sample['seed'] = base_seed + i
        dataset.append(sample)

        n_edges = len(sample['robot_edges'])
        st = sample['solve_time']
        print(f"{i+1:>5} {nr:>3} {no:>3} {n_edges:>5} {st:>6.2f}s {'OK':>10} {len(dataset):>6}")

        # Periyodik kayit
        if save_path and len(dataset) % save_interval == 0:
            with open(save_path, 'wb') as f:
                pickle.dump(dataset, f)
            elapsed = time.time() - t_start
            rate = len(dataset) / max(elapsed, 1)
            remaining = (n_samples - i - 1) / max(rate, 0.01)
            print(f"  >>> Kaydedildi: {len(dataset)} sample | "
                  f"Gecen: {elapsed/60:.1f}dk | "
                  f"Kalan tahmini: {remaining/60:.1f}dk")

    # Son kayit
    if save_path and dataset:
        with open(save_path, 'wb') as f:
            pickle.dump(dataset, f)

    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"Tamamlandi: {len(dataset)}/{n_samples} basarili, {failed} basarisiz")
    print(f"Toplam sure: {elapsed/60:.1f} dakika ({elapsed:.0f}s)")
    if dataset:
        times = [s['solve_time'] for s in dataset]
        print(f"Solve suresi: min={min(times):.2f}s, max={max(times):.2f}s, "
              f"mean={np.mean(times):.2f}s")

    return dataset


def make_splits(dataset, train_ratio=0.9, seed=42):
    """Dataset'i train/val olarak bol ve graph formatina cevir.

    Args:
        dataset: list of raw sample dicts
        train_ratio: train orani (0.9 = %90 train, %10 val)
        seed: shuffle icin random seed

    Returns:
        train_graphs, val_graphs: list of graph dicts
    """
    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(dataset))
    split = int(len(dataset) * train_ratio)

    train_idx = indices[:split]
    val_idx = indices[split:]

    train_graphs = [sample_to_graph(dataset[i]) for i in train_idx]
    val_graphs = [sample_to_graph(dataset[i]) for i in val_idx]

    return train_graphs, val_graphs


def print_dataset_stats(graphs, name):
    """Dataset istatistiklerini yazdir."""
    print(f"\n--- {name} ({len(graphs)} sample) ---")
    if not graphs:
        return

    n_robots = [g['node_feat_robot'].shape[0] for g in graphs]
    n_obs = [g['node_feat_obstacle'].shape[0] for g in graphs]
    n_ro = [g['edge_index_RO'].shape[1] for g in graphs]
    n_rr = [g['edge_index_RR'].shape[1] for g in graphs]

    print(f"  Robotlar: min={min(n_robots)}, max={max(n_robots)}, "
          f"mean={np.mean(n_robots):.1f}")
    print(f"  Engeller: min={min(n_obs)}, max={max(n_obs)}, "
          f"mean={np.mean(n_obs):.1f}")
    print(f"  RO edges: min={min(n_ro)}, max={max(n_ro)}, "
          f"mean={np.mean(n_ro):.1f}")
    print(f"  RR edges: min={min(n_rr)}, max={max(n_rr)}, "
          f"mean={np.mean(n_rr):.1f}")

    # Label dagilimi (binary balance)
    all_ro = np.concatenate([g['edge_labels_RO'].flatten() for g in graphs])
    ro_ones_pct = 100 * all_ro.sum() / max(all_ro.size, 1)
    print(f"  RO label dagilimi: {ro_ones_pct:.1f}% ones (relaxed)")

    all_rr_flat = [g['edge_labels_RR'].flatten() for g in graphs if g['edge_labels_RR'].size > 0]
    if all_rr_flat:
        all_rr = np.concatenate(all_rr_flat)
        rr_ones_pct = 100 * all_rr.sum() / max(all_rr.size, 1)
        print(f"  RR label dagilimi: {rr_ones_pct:.1f}% ones (relaxed)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GAT dataset uretici")
    parser.add_argument("--n", type=int, default=2000, help="Sample sayisi")
    parser.add_argument("--resume", action="store_true", help="Onceki calismayi devam ettir")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--min-robots", type=int, default=2)
    parser.add_argument("--max-robots", type=int, default=5)
    parser.add_argument("--min-obs", type=int, default=1)
    parser.add_argument("--max-obs", type=int, default=3)
    args = parser.parse_args()

    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(data_dir, exist_ok=True)

    full_path = os.path.join(data_dir, 'dataset_full.pkl')
    train_path = os.path.join(data_dir, 'dataset_train.pkl')
    val_path = os.path.join(data_dir, 'dataset_val.pkl')

    # 1) Dataset topla
    dataset = collect_dataset_large(
        n_samples=args.n,
        n_robots_range=(args.min_robots, args.max_robots),
        n_obstacles_range=(args.min_obs, args.max_obs),
        base_seed=args.seed,
        save_path=full_path,
        resume_path=full_path if args.resume else None,
    )

    if not dataset:
        print("Hic sample toplanamadi!")
        sys.exit(1)

    # 2) Train/val split + graph formatina cevir
    print(f"\n{'='*50}")
    print("Train/validation split (%90/%10) ve graph donusumu...")
    train_graphs, val_graphs = make_splits(dataset, train_ratio=0.9)

    # Kaydet
    with open(train_path, 'wb') as f:
        pickle.dump(train_graphs, f)
    with open(val_path, 'wb') as f:
        pickle.dump(val_graphs, f)

    print(f"Train: {len(train_graphs)} sample -> {train_path}")
    print(f"Val:   {len(val_graphs)} sample -> {val_path}")

    # Istatistikler
    print_dataset_stats(train_graphs, "TRAIN")
    print_dataset_stats(val_graphs, "VALIDATION")
