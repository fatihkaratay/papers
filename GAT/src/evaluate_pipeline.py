import time
import numpy as np
import pickle
import os

from scenario_generator import generate_scenario
from multi_robot_micp import solve_multi_robot_micp
from collect_data import sample_to_graph
from gat_qp_solver import load_model, predict_binaries, solve_qp_with_fixed_binaries


def build_graph_for_prediction(env, edges, H):
    NR = len(env.robots)
    NO = len(env.obstacles)

    robots_p = [r.position.copy() for r in env.robots]
    robots_goal = [r.goal.copy() for r in env.robots]

    obstacle_data = []
    for obs in env.obstacles:
        obstacle_data.append({
            'center': obs.center.copy(),
            'half_length': obs.half_length,
            'half_width': obs.half_width,
            'angle': obs.angle,
        })

    b_obs_list = []
    for i in range(NR):
        for o in range(NO):
            b_obs_list.append({
                'robot': i, 'obstacle': o,
                'binaries': np.zeros((H, 4)),
            })
    b_rob_list = []
    for (i, j) in edges:
        b_rob_list.append({
            'robot_i': i, 'robot_j': j,
            'binaries': np.zeros((H, 4)),
        })

    sample = {
        'n_robots': NR, 'n_obstacles': NO,
        'robots_p': robots_p, 'robots_goal': robots_goal,
        'obstacles': obstacle_data,
        'robot_edges': edges,
        'b_obs': b_obs_list, 'b_rob': b_rob_list,
    }
    return sample_to_graph(sample)


def check_collisions(p_trajs, obstacles, robot_edges, dmin):
    NR = len(p_trajs)
    H = p_trajs[0].shape[0] - 1
    obs_collisions = 0
    rob_collisions = 0

    # Robot-engel carpmasi
    for i in range(NR):
        for obs in obstacles:
            ca = np.cos(obs.angle)
            sa = np.sin(obs.angle)
            ox, oy = obs.center
            L, W = obs.half_length, obs.half_width

            for k in range(1, H + 1):
                px, py = p_trajs[i][k]
                dx, dy = px - ox, py - oy
                lx = abs(ca * dx + sa * dy)
                ly = abs(-sa * dx + ca * dy)
                if lx < L + dmin - 1e-4 and ly < W + dmin - 1e-4:
                    obs_collisions += 1

    for (i, j) in robot_edges:
        for k in range(1, H + 1):
            dist = np.linalg.norm(p_trajs[i][k] - p_trajs[j][k])
            if dist < 2 * dmin - 1e-4:
                rob_collisions += 1

    return obs_collisions, rob_collisions


def check_goal_reached(p_trajs, robots_goal, threshold=0.5):
    reached = 0
    for i in range(len(p_trajs)):
        dist = np.linalg.norm(p_trajs[i][-1] - robots_goal[i])
        if dist < threshold:
            reached += 1
    return reached


def evaluate(n_scenarios=100, n_robots_range=(2, 5), n_obstacles_range=(1, 3),
             H=15, tau=0.2, vmax=0.5, amax=0.5, dmin=0.2, dprox=5.0,
             bounds=(-4.0, 4.0, -4.0, 4.0), base_seed=9999):

    model, norm_stats = load_model()

    results = {
        'micp_times': [], 'gat_times': [], 'qp_times': [],
        'ro_accs': [], 'rr_accs': [],
        'micp_obs_collisions': [], 'micp_rob_collisions': [],
        'gat_obs_collisions': [], 'gat_rob_collisions': [],
        'micp_goals': [], 'gat_goals': [],
        'gat_infeasible': 0, 'gat_soft': 0,
        'micp_fail': 0, 'gen_fail': 0,
        'total_robots': [],
        'n_scenarios_ok': 0,
    }

    print(f"Degerlendirme: {n_scenarios} senaryo")
    print(f"  Robotlar: {n_robots_range}, Engeller: {n_obstacles_range}")
    print(f"\n{'#':>4} {'NR':>3} {'NO':>3} {'MICP_ms':>8} {'GAT+QP_ms':>9} "
          f"{'Speed':>6} {'RO%':>5} {'RR%':>5} {'Coll':>5} {'Note':>8}")
    print("-" * 65)

    for idx in range(n_scenarios):
        rng = np.random.RandomState(base_seed + idx)
        nr = rng.randint(n_robots_range[0], n_robots_range[1] + 1)
        no = rng.randint(n_obstacles_range[0], n_obstacles_range[1] + 1)

        try:
            env = generate_scenario(nr, no, bounds, dmin, rng)
        except RuntimeError:
            results['gen_fail'] += 1
            continue

        robots_p = [r.position.copy() for r in env.robots]
        robots_v = [r.velocity.copy() for r in env.robots]
        robots_goal = [r.goal.copy() for r in env.robots]

        t0 = time.time()
        try:
            p_micp, v_micp, u_micp, edges, b_obs_gt, b_rob_gt = \
                solve_multi_robot_micp(robots_p, robots_v, robots_goal,
                                        env.obstacles, H, tau, bounds,
                                        vmax, amax, dmin, dprox)
        except Exception:
            results['micp_fail'] += 1
            continue
        t_micp = time.time() - t0

        graph = build_graph_for_prediction(env, edges, H)

        t0 = time.time()
        b_obs_pred, b_rob_pred = predict_binaries(model, graph, norm_stats)
        t_gat = time.time() - t0

        t0 = time.time()
        qp_result = solve_qp_with_fixed_binaries(
            robots_p, robots_v, robots_goal,
            env.obstacles, H, tau, bounds, vmax, amax, dmin,
            b_obs_pred, b_rob_pred, edges
        )
        t_qp = time.time() - t0

        if qp_result is None:
            results['gat_infeasible'] += 1
            print(f"{idx+1:>4} {nr:>3} {no:>3} {t_micp*1000:>7.0f} {'---':>9} "
                  f"{'---':>6} {'---':>5} {'---':>5} {'---':>5} {'INFEAS':>8}")
            continue

        p_gat, v_gat, u_gat, slack = qp_result
        used_soft = slack > 1e-6

        if used_soft:
            results['gat_soft'] += 1

        results['n_scenarios_ok'] += 1
        results['total_robots'].append(nr)
        results['micp_times'].append(t_micp)
        results['gat_times'].append(t_gat)
        results['qp_times'].append(t_qp)

        ro_match, ro_total = 0, 0
        for key in b_obs_pred:
            if key in b_obs_gt:
                ro_match += (b_obs_pred[key] == b_obs_gt[key]).sum()
                ro_total += b_obs_pred[key].size
        ro_acc = ro_match / max(ro_total, 1)
        results['ro_accs'].append(ro_acc)

        rr_match, rr_total = 0, 0
        for key in b_rob_pred:
            if key in b_rob_gt:
                rr_match += (b_rob_pred[key] == b_rob_gt[key]).sum()
                rr_total += b_rob_pred[key].size
        rr_acc = rr_match / max(rr_total, 1) if rr_total > 0 else float('nan')
        results['rr_accs'].append(rr_acc)

        m_oc, m_rc = check_collisions(p_micp, env.obstacles, edges, dmin)
        g_oc, g_rc = check_collisions(p_gat, env.obstacles, edges, dmin)
        results['micp_obs_collisions'].append(m_oc)
        results['micp_rob_collisions'].append(m_rc)
        results['gat_obs_collisions'].append(g_oc)
        results['gat_rob_collisions'].append(g_rc)

        m_goals = check_goal_reached(p_micp, robots_goal)
        g_goals = check_goal_reached(p_gat, robots_goal)
        results['micp_goals'].append(m_goals)
        results['gat_goals'].append(g_goals)

        speedup = t_micp / (t_gat + t_qp)
        coll_str = f"{g_oc+g_rc}" if (g_oc + g_rc) > 0 else "0"
        note = "SOFT" if used_soft else "OK"
        rr_str = f"{100*rr_acc:4.0f}" if not np.isnan(rr_acc) else " N/A"

        print(f"{idx+1:>4} {nr:>3} {no:>3} {t_micp*1000:>7.0f} "
              f"{(t_gat+t_qp)*1000:>8.0f} {speedup:>5.1f}x "
              f"{100*ro_acc:>4.0f} {rr_str} {coll_str:>5} {note:>8}")

    print(f"\n{'='*65}")
    print(f"OZET ({results['n_scenarios_ok']}/{n_scenarios} senaryo basarili)")
    print(f"  Gen fail: {results['gen_fail']}, MICP fail: {results['micp_fail']}, "
          f"GAT infeasible: {results['gat_infeasible']}")

    if results['n_scenarios_ok'] == 0:
        print("Hic basarili senaryo yok!")
        return results

    micp_ms = np.array(results['micp_times']) * 1000
    gat_ms = np.array(results['gat_times']) * 1000
    qp_ms = np.array(results['qp_times']) * 1000
    total_ms = gat_ms + qp_ms

    print(f"\n--- Sure (ms) ---")
    print(f"  MICP:   mean={micp_ms.mean():.0f}, median={np.median(micp_ms):.0f}, "
          f"max={micp_ms.max():.0f}")
    print(f"  GAT+QP: mean={total_ms.mean():.0f}, median={np.median(total_ms):.0f}, "
          f"max={total_ms.max():.0f}")
    print(f"  GAT:    mean={gat_ms.mean():.1f}")
    print(f"  QP:     mean={qp_ms.mean():.0f}")
    speedups = micp_ms / total_ms
    print(f"  Speedup: mean={speedups.mean():.1f}x, median={np.median(speedups):.1f}x")

    print(f"\n--- Binary Accuracy ---")
    print(f"  RO: mean={100*np.mean(results['ro_accs']):.1f}%")
    rr_valid = [a for a in results['rr_accs'] if not np.isnan(a)]
    if rr_valid:
        print(f"  RR: mean={100*np.mean(rr_valid):.1f}%")

    print(f"\n--- Collision ---")
    gat_coll_scenarios = sum(1 for oc, rc in zip(results['gat_obs_collisions'],
                                                   results['gat_rob_collisions'])
                              if oc + rc > 0)
    micp_coll_scenarios = sum(1 for oc, rc in zip(results['micp_obs_collisions'],
                                                    results['micp_rob_collisions'])
                               if oc + rc > 0)
    n_ok = results['n_scenarios_ok']
    print(f"  MICP: {micp_coll_scenarios}/{n_ok} senaryoda carpma "
          f"({100*micp_coll_scenarios/n_ok:.1f}%)")
    print(f"  GAT+QP: {gat_coll_scenarios}/{n_ok} senaryoda carpma "
          f"({100*gat_coll_scenarios/n_ok:.1f}%)")
    print(f"  Soft constraints kullanilan: {results['gat_soft']}/{n_ok}")

    print(f"\n--- Hedefe Ulasma ---")
    total_robots_micp = sum(results['micp_goals'])
    total_robots_gat = sum(results['gat_goals'])
    total_robots_all = sum(results['total_robots'])
    print(f"  MICP: {total_robots_micp}/{total_robots_all} robot "
          f"({100*total_robots_micp/total_robots_all:.1f}%)")
    print(f"  GAT+QP: {total_robots_gat}/{total_robots_all} robot "
          f"({100*total_robots_gat/total_robots_all:.1f}%)")

    print(f"\n--- Robot Sayisina Gore Speedup ---")
    robots_arr = np.array(results['total_robots'])
    for nr_val in sorted(set(robots_arr)):
        mask = robots_arr == nr_val
        if mask.sum() > 0:
            sp = speedups[mask]
            print(f"  NR={nr_val}: mean speedup={sp.mean():.1f}x "
                  f"({mask.sum()} senaryo)")

    return results


if __name__ == "__main__":
    results = evaluate(n_scenarios=100, base_seed=9999)
