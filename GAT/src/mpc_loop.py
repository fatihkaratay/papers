"""
Faz 2.6: Receding Horizon (MPC) loop.

Her adimda:
  1. Mevcut durumu al: p(t), v(t)
  2. H adimlik MICP coz
  3. Sadece ILK kontrol inputunu uygula: u(0)
  4. Robot bir adim ilerler
  5. Hedefe ulasana kadar tekrarla
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider
from environment import Environment
from single_robot_micp import solve_single_robot_micp


def run_mpc(env, robot_idx, H, tau, vmax, amax, dmin,
            max_steps=100, goal_tol=0.15, M=100.0,
            w_pt=10.0, w_p=1.0, w_u=1.0):
    """MPC loop: her adimda MICP coz, ilk inputu uygula.

    Args:
        env: Environment nesnesi (obstacles ve robots iceriyor)
        robot_idx: hangi robot
        H: horizon length
        tau: sampling period
        max_steps: maksimum adim sayisi
        goal_tol: hedefe bu kadar yaklasinca dur

    Returns:
        p_history: gercek pozisyon gecmisi, shape (T+1, 2)
        v_history: gercek hiz gecmisi, shape (T+1, 2)
        u_history: uygulanan kontrol inputlari, shape (T, 2)
        planned_trajectories: her adimda planlanan trajectory listesi
    """
    robot = env.robots[robot_idx]
    p = robot.position.copy()
    v = robot.velocity.copy()
    p_goal = robot.goal.copy()

    p_history = [p.copy()]
    v_history = [v.copy()]
    u_history = []
    planned_trajectories = []

    for step in range(max_steps):
        # Hedefe ulastik mi?
        dist = np.linalg.norm(p - p_goal)
        if dist < goal_tol:
            print(f"  Goal reached at step {step}! (dist={dist:.3f}m)")
            break

        # MICP coz
        try:
            p_traj, v_traj, u_traj, _ = solve_single_robot_micp(
                p, v, p_goal, env.obstacles, H, tau,
                env.bounds, vmax, amax, dmin, M, w_pt, w_p, w_u
            )
        except Exception as e:
            print(f"  MICP failed at step {step}: {e}")
            break

        planned_trajectories.append(p_traj.copy())

        # Sadece ILK kontrol inputunu uygula
        u0 = u_traj[0]
        u_history.append(u0.copy())

        # Robot bir adim ilerle (double-integrator)
        p = p + tau * v + 0.5 * tau**2 * u0
        v = v + tau * u0

        p_history.append(p.copy())
        v_history.append(v.copy())

    else:
        print(f"  Max steps ({max_steps}) reached. dist={dist:.3f}m")

    return (np.array(p_history), np.array(v_history),
            np.array(u_history), planned_trajectories)


def animate_mpc(env, p_history, v_history, u_history,
                planned_trajectories, tau, dmin):
    """MPC sonucunu interaktif animasyon olarak goster."""
    n_frames = len(p_history)
    time_all = np.arange(n_frames) * tau
    speeds = np.linalg.norm(v_history, axis=1)

    state = {'frame': 0, 'playing': True}

    fig = plt.figure(figsize=(14, 6.5))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 0.08], hspace=0.35,
                          left=0.06, right=0.97, top=0.92, bottom=0.12)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax_slider = fig.add_subplot(gs[1, :])

    # --- Panel 1: Ortam + robot ---
    px_min, px_max, py_min, py_max = env.bounds
    ax1.set_xlim(px_min - 0.2, px_max + 0.2)
    ax1.set_ylim(py_min - 0.2, py_max + 0.2)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('MPC — Receding Horizon')

    border = patches.Rectangle(
        (px_min, py_min), px_max - px_min, py_max - py_min,
        linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
    ax1.add_patch(border)

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

    goal = env.robots[0].goal
    ax1.plot(*goal, '*', color='tab:blue', markersize=15,
             markeredgecolor='black', markeredgewidth=0.5, zorder=5)

    # Tam gercek trajectory (soluk)
    ax1.plot(p_history[:, 0], p_history[:, 1], '-', color='tab:blue',
             alpha=0.15, linewidth=1)

    trail_line, = ax1.plot([], [], '-o', color='tab:blue', markersize=3,
                           linewidth=2, alpha=0.6)
    # Planlanan trajectory (her adimda guncellenen, soluk kirmizi)
    plan_line, = ax1.plot([], [], '--', color='tab:red', linewidth=1.5,
                          alpha=0.5, label='planned')
    robot_circle = patches.Circle((0, 0), dmin, fill=False,
                                   edgecolor='tab:blue', linewidth=2)
    ax1.add_patch(robot_circle)
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    ax1.legend(loc='upper right')

    # --- Panel 2: Speed profile ---
    ax2.set_xlim(0, time_all[-1] + tau)
    ax2.set_ylim(0, max(speeds) * 1.15 + 0.05)
    ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='vmax=0.5')
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('speed (m/s)')
    ax2.set_title('Speed Profile')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    speed_line, = ax2.plot([], [], '-o', color='tab:blue', markersize=3)
    speed_marker, = ax2.plot([], [], 'o', color='red', markersize=8, zorder=5)

    # --- Slider ---
    slider = Slider(ax_slider, 'Step', 0, n_frames - 1, valinit=0,
                    valstep=1, color='tab:blue', alpha=0.5)

    # --- Butonlar ---
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

        trail_line.set_data(p_history[:frame+1, 0], p_history[:frame+1, 1])
        robot_circle.center = (p_history[frame, 0], p_history[frame, 1])

        # Planlanan trajectory goster (o adimda MICP'nin plani)
        if frame < len(planned_trajectories):
            pt = planned_trajectories[frame]
            plan_line.set_data(pt[:, 0], pt[:, 1])
        else:
            plan_line.set_data([], [])

        dist = np.linalg.norm(p_history[frame] - goal)
        time_text.set_text(f't = {frame * tau:.1f}s\n'
                           f'speed = {speeds[frame]:.2f} m/s\n'
                           f'dist = {dist:.2f}m')

        speed_line.set_data(time_all[:frame+1], speeds[:frame+1])
        speed_marker.set_data([time_all[frame]], [speeds[frame]])

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
        if state['frame'] >= n_frames - 1:
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
        draw_frame(min(n_frames - 1, state['frame'] + 1))

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


if __name__ == "__main__":
    tau = 0.2
    H = 20
    env_bounds = (-3.0, 0.5, -0.5, 2.0)
    vmax = 0.5
    amax = 0.5
    dmin = 0.2

    env = Environment(bounds=env_bounds)
    env.add_obstacle(center=(-1.5, 0.8), half_length=0.3, half_width=0.2)
    env.add_obstacle(center=(-0.5, 1.2), half_length=0.2, half_width=0.15,
                      angle=np.radians(20))

    p_init = np.array([-2.5, 0.3])
    v_init = np.array([0.0, 0.0])
    p_goal = np.array([0.0, 1.5])
    env.add_robot(position=p_init, goal=p_goal)

    print("Running MPC loop...")
    p_hist, v_hist, u_hist, plans = run_mpc(
        env, robot_idx=0, H=H, tau=tau,
        vmax=vmax, amax=amax, dmin=dmin,
        max_steps=60, goal_tol=0.15
    )

    print(f"Total steps: {len(u_hist)}")
    animate_mpc(env, p_hist, v_hist, u_hist, plans, tau, dmin)
