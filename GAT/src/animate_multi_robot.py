"""
Faz 3: Multi-robot MICP animasyonu — interaktif kontroller ile.
Play/Pause, Step, Slider, Replay destegi.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider
from environment import Environment
from multi_robot_micp import solve_multi_robot_micp


def animate_multi_robot(env, p_trajs, v_trajs, u_trajs, robot_edges,
                        tau, dmin):
    """Multi-robot MICP sonucunu interaktif animasyon olarak goster.

    Args:
        env: Environment nesnesi
        p_trajs: list of (H+1, 2) — her robotun trajectory'si
        v_trajs: list of (H+1, 2) — her robotun hiz profili
        u_trajs: list of (H, 2) — her robotun kontrol inputlari
        robot_edges: list of (i, j) — robot-robot edge'leri
        tau: sampling period
        dmin: guvenli yaricap
    """
    NR = len(p_trajs)
    H = len(u_trajs[0])
    n_frames = H + 1
    time_all = np.arange(n_frames) * tau
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']

    state = {'frame': 0, 'playing': True}

    # --- Layout: 2 panel + slider ---
    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.08], hspace=0.35,
                          left=0.06, right=0.97, top=0.92, bottom=0.12)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_slider = fig.add_subplot(gs[1, :])

    # === Panel 1: Ortam + robotlar ===
    px_min, px_max, py_min, py_max = env.bounds
    ax1.set_xlim(px_min - 0.3, px_max + 0.3)
    ax1.set_ylim(py_min - 0.3, py_max + 0.3)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('Multi-Robot MICP')

    # Sinir
    border = patches.Rectangle(
        (px_min, py_min), px_max - px_min, py_max - py_min,
        linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
    ax1.add_patch(border)

    # Engeller
    for obs in env.obstacles:
        w = 2 * obs.half_length
        h = 2 * obs.half_width
        rect = patches.Rectangle(
            (-obs.half_length, -obs.half_width), w, h,
            linewidth=1, edgecolor='dimgray', facecolor='gray', alpha=0.7)
        t = (plt.matplotlib.transforms.Affine2D()
             .rotate(obs.angle)
             .translate(obs.center[0], obs.center[1])
             + ax1.transData)
        rect.set_transform(t)
        ax1.add_patch(rect)

    # Her robot icin: hedef, tam trajectory (soluk), trail, daire
    trail_lines = []
    robot_circles = []
    for i in range(NR):
        c = colors[i % len(colors)]
        # Hedef (yildiz)
        ax1.plot(*env.robots[i].goal, '*', color=c, markersize=15,
                 markeredgecolor='black', markeredgewidth=0.5, zorder=5)
        # Tam trajectory (soluk)
        ax1.plot(p_trajs[i][:, 0], p_trajs[i][:, 1], '-', color=c,
                 alpha=0.12, linewidth=1)
        # Canli trail
        line, = ax1.plot([], [], '-o', color=c, markersize=2,
                         linewidth=2, alpha=0.6, label=f'Robot {i+1}')
        trail_lines.append(line)
        # Guvenli daire
        circle = patches.Circle((0, 0), dmin, fill=False,
                                edgecolor=c, linewidth=2, alpha=0.8)
        ax1.add_patch(circle)
        robot_circles.append(circle)

    ax1.legend(loc='upper right', fontsize=9)
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    # === Panel 2: Inter-robot mesafe ===
    # Onceden hesapla
    edge_dists = {}
    for (i, j) in robot_edges:
        edge_dists[i, j] = np.linalg.norm(p_trajs[i] - p_trajs[j], axis=1)

    ax2.set_xlim(0, time_all[-1] + tau)
    if edge_dists:
        max_dist = max(d.max() for d in edge_dists.values())
    else:
        max_dist = 1.0
    ax2.set_ylim(0, max_dist * 1.1)
    ax2.axhline(y=2 * dmin, color='red', linestyle='--',
                alpha=0.5, label=f'2*dmin={2*dmin}')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('distance (m)')
    ax2.set_title('Inter-Robot Distance')
    ax2.grid(True, alpha=0.3)

    dist_lines = {}
    dist_markers = {}
    for (i, j) in robot_edges:
        line, = ax2.plot([], [], '-o', markersize=3,
                         label=f'R{i+1}-R{j+1}')
        marker, = ax2.plot([], [], 'o', color=line.get_color(),
                           markersize=8, zorder=5)
        dist_lines[i, j] = line
        dist_markers[i, j] = marker
    ax2.legend(loc='upper right', fontsize=9)

    # === Slider ===
    slider = Slider(ax_slider, 'Step', 0, H, valinit=0, valstep=1,
                    color='tab:blue', alpha=0.5)

    # === Butonlar ===
    ax_play = fig.add_axes([0.35, 0.01, 0.08, 0.04])
    ax_prev = fig.add_axes([0.44, 0.01, 0.05, 0.04])
    ax_next = fig.add_axes([0.50, 0.01, 0.05, 0.04])
    ax_replay = fig.add_axes([0.56, 0.01, 0.08, 0.04])

    btn_play = Button(ax_play, 'Pause')
    btn_prev = Button(ax_prev, '<<')
    btn_next = Button(ax_next, '>>')
    btn_replay = Button(ax_replay, 'Replay')

    def draw_frame(frame):
        frame = int(frame)
        state['frame'] = frame

        # Robotlari guncelle
        for i in range(NR):
            trail_lines[i].set_data(p_trajs[i][:frame+1, 0],
                                    p_trajs[i][:frame+1, 1])
            robot_circles[i].center = (p_trajs[i][frame, 0],
                                       p_trajs[i][frame, 1])

        # Mesafe grafiklerini guncelle
        for (i, j) in robot_edges:
            dist_lines[i, j].set_data(time_all[:frame+1],
                                      edge_dists[i, j][:frame+1])
            dist_markers[i, j].set_data([time_all[frame]],
                                        [edge_dists[i, j][frame]])

        # Bilgi kutusu
        info = f't = {frame * tau:.1f}s'
        for (i, j) in robot_edges:
            info += f'\nR{i+1}-R{j+1}: {edge_dists[i,j][frame]:.2f}m'
        time_text.set_text(info)

        slider.eventson = False
        slider.set_val(frame)
        slider.eventson = True
        fig.canvas.draw_idle()

    def animate_step(_):
        if not state['playing']:
            return
        f = state['frame'] + 1
        if f >= n_frames:
            state['playing'] = False
            btn_play.label.set_text('Play')
            return
        draw_frame(f)

    def on_play(event):
        if state['frame'] >= H:
            state['frame'] = 0
            draw_frame(0)
        state['playing'] = not state['playing']
        btn_play.label.set_text('Pause' if state['playing'] else 'Play')
        fig.canvas.draw_idle()

    def on_prev(event):
        state['playing'] = False
        btn_play.label.set_text('Play')
        draw_frame(max(0, state['frame'] - 1))

    def on_next(event):
        state['playing'] = False
        btn_play.label.set_text('Play')
        draw_frame(min(H, state['frame'] + 1))

    def on_replay(event):
        state['frame'] = 0
        state['playing'] = True
        btn_play.label.set_text('Pause')
        draw_frame(0)

    slider.on_changed(lambda val: draw_frame(int(val)))
    btn_play.on_clicked(on_play)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_replay.on_clicked(on_replay)

    timer = fig.canvas.new_timer(interval=200)
    timer.add_callback(animate_step, 0)
    timer.start()

    draw_frame(0)
    plt.show()


def run_scenario(name, env, H, tau, vmax, amax, dmin, dprox):
    """Senaryo coz ve animasyonu goster."""
    robots_p = [r.position.copy() for r in env.robots]
    robots_v = [r.velocity.copy() for r in env.robots]
    robots_goal = [r.goal.copy() for r in env.robots]

    print(f"\n=== {name} ===")
    print(f"Robots: {len(env.robots)}, Obstacles: {len(env.obstacles)}")
    print("Solving multi-robot MICP...")

    p_trajs, v_trajs, u_trajs, edges, _, _ = solve_multi_robot_micp(
        robots_p, robots_v, robots_goal,
        env.obstacles, H, tau, env.bounds,
        vmax, amax, dmin, dprox
    )
    print(f"Robot-robot edges: {edges}")
    print("Launching animation...")

    animate_multi_robot(env, p_trajs, v_trajs, u_trajs, edges, tau, dmin)


if __name__ == "__main__":
    import sys

    tau = 0.2
    H = 20
    vmax = 0.5
    amax = 0.5
    dmin = 0.2
    dprox = 5.0

    # Hangi senaryoyu calistiracagiz?
    scenario = sys.argv[1] if len(sys.argv) > 1 else "3r2o"

    if scenario == "2r1o":
        # --- Senaryo A: 2 robot + 1 engel (karsidan karsilasma) ---
        env = Environment(bounds=(-3.0, 1.0, -0.5, 2.5))
        env.add_obstacle(center=(-1.0, 1.0), half_length=0.3, half_width=0.2)
        env.add_robot(position=(-2.5, 1.0), goal=(0.5, 1.0))
        env.add_robot(position=(0.5, 1.0), goal=(-2.5, 1.0))
        run_scenario("2 Robot + 1 Engel", env, H, tau, vmax, amax, dmin, dprox)

    elif scenario == "3r2o":
        # --- Senaryo B: 3 robot + 2 engel (capraz gecis) ---
        env = Environment(bounds=(-3.0, 3.0, -1.0, 3.0))
        env.add_obstacle(center=(-0.5, 1.0), half_length=0.35, half_width=0.2)
        env.add_obstacle(center=(0.8, 1.8), half_length=0.25, half_width=0.2,
                         angle=np.radians(30))
        # Robot 1: sol-alt → sag-ust
        env.add_robot(position=(-2.5, 0.0), goal=(2.5, 2.5))
        # Robot 2: sag-alt → sol-ust
        env.add_robot(position=(2.5, 0.0), goal=(-2.5, 2.5))
        # Robot 3: sol-ust → sag-alt (digerlerini keser)
        env.add_robot(position=(-2.0, 2.5), goal=(2.0, 0.0))
        run_scenario("3 Robot + 2 Engel", env, H, tau, vmax, amax, dmin, dprox)

    else:
        print(f"Unknown scenario: {scenario}")
        print("Usage: python animate_multi_robot.py [2r1o|3r2o]")
