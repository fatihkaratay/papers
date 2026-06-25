import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider
from matplotlib.animation import FuncAnimation
from environment import Environment
from single_robot_micp import solve_single_robot_micp


def animate_trajectory(env, p_traj, v_traj, u_traj, binaries, tau, dmin,
                        save_gif=None):
    H = len(u_traj)
    n_frames = H + 1
    time_all = np.arange(n_frames) * tau
    speeds = np.linalg.norm(v_traj, axis=1)

    state = {'frame': 0, 'playing': True, 'anim': None}

    fig = plt.figure(figsize=(16, 6.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 0.08], hspace=0.35,
                          left=0.05, right=0.97, top=0.92, bottom=0.12)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])
    ax_slider = fig.add_subplot(gs[1, :])

    px_min, px_max, py_min, py_max = env.bounds
    ax1.set_xlim(px_min - 0.2, px_max + 0.2)
    ax1.set_ylim(py_min - 0.2, py_max + 0.2)
    ax1.set_aspect('equal')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel('x (m)')
    ax1.set_ylabel('y (m)')
    ax1.set_title('MICP Trajectory')

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
    ax1.plot(p_traj[:, 0], p_traj[:, 1], '-', color='tab:blue',
             alpha=0.15, linewidth=1)

    trail_line, = ax1.plot([], [], '-o', color='tab:blue', markersize=2,
                           linewidth=2, alpha=0.5)
    robot_circle = patches.Circle((0, 0), dmin, fill=False,
                                   edgecolor='tab:blue', linewidth=2)
    ax1.add_patch(robot_circle)
    time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes,
                         fontsize=10, verticalalignment='top',
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

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

    labels = ['right', 'top', 'left', 'bottom']
    colors_b = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
    time_u = np.arange(H) * tau
    ax3.set_xlim(0, time_all[-1] + tau)
    ax3.set_ylim(-0.1, 1.3)
    ax3.set_xlabel('time (s)')
    ax3.set_ylabel('binary value')
    ax3.set_title('Binary Decisions (obs 1)')
    ax3.set_yticks([0, 1])
    ax3.grid(True, alpha=0.3)
    binary_lines = []
    for m in range(4):
        line, = ax3.plot([], [], '-o', color=colors_b[m], markersize=3,
                         label=labels[m], alpha=0.7)
        binary_lines.append(line)
    ax3.legend(loc='upper right', fontsize=8)

    slider = Slider(ax_slider, 'Step', 0, H, valinit=0, valstep=1,
                    color='tab:blue', alpha=0.5)

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

        trail_line.set_data(p_traj[:frame+1, 0], p_traj[:frame+1, 1])
        robot_circle.center = (p_traj[frame, 0], p_traj[frame, 1])
        time_text.set_text(f't = {frame * tau:.1f}s\n'
                           f'speed = {speeds[frame]:.2f} m/s')

        speed_line.set_data(time_all[:frame+1], speeds[:frame+1])
        speed_marker.set_data([time_all[frame]], [speeds[frame]])

        if frame > 0:
            k_end = min(frame, H)
            for m in range(4):
                binary_lines[m].set_data(time_u[:k_end], binaries[0, :k_end, m])
        else:
            for m in range(4):
                binary_lines[m].set_data([], [])

        slider.eventson = False
        slider.set_val(frame)
        slider.eventson = True

        fig.canvas.draw_idle()

    def on_slider(val):
        draw_frame(int(val))

    def animate_step(frame_num):
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
        f = max(0, state['frame'] - 1)
        draw_frame(f)

    def on_next(event):
        state['playing'] = False
        btn_play.label.set_text('Play')
        f = min(H, state['frame'] + 1)
        draw_frame(f)

    def on_replay(event):
        state['frame'] = 0
        state['playing'] = True
        btn_play.label.set_text('Pause')
        draw_frame(0)

    slider.on_changed(on_slider)
    btn_play.on_clicked(on_play)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_replay.on_clicked(on_replay)

    timer = fig.canvas.new_timer(interval=200)
    timer.add_callback(animate_step, 0)
    timer.start()

    draw_frame(0)

    if save_gif:
        anim = FuncAnimation(fig, lambda f: draw_frame(f),
                              frames=n_frames, interval=200, repeat=False)
        anim.save(save_gif, writer='pillow', fps=5)
        print(f"GIF saved: {save_gif}")

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

    p_traj, v_traj, u_traj, binaries = solve_single_robot_micp(
        p_init, v_init, p_goal, env.obstacles, H, tau,
        env_bounds, vmax, amax, dmin
    )

    animate_trajectory(env, p_traj, v_traj, u_traj, binaries, tau, dmin)
