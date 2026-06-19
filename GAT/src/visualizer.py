"""
Genel animasyon ve gorsellestirme modulu.
Tum fazlarda kullanilacak tek bir arac.

Kullanim:
  viz = Visualizer(env, tau=0.2, dmin=0.2)
  viz.add_robot_trajectory(p_traj, v_traj, label="Robot 1")
  viz.add_robot_trajectory(p_traj2, v_traj2, label="Robot 2")   # coklu robot
  viz.add_planned_trajectories(plans)                            # MPC planlari
  viz.add_binary_decisions(binaries, obs_idx=0)                  # binary kararlar
  viz.show()                                                     # interaktif animasyon
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Button, Slider


# Robot renkleri
COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple',
          'tab:brown', 'tab:pink', 'tab:cyan']


class Visualizer:
    def __init__(self, env, tau, dmin=0.2):
        """
        Args:
            env: Environment nesnesi
            tau: sampling period (s)
            dmin: robot guvenli yaricap (m)
        """
        self.env = env
        self.tau = tau
        self.dmin = dmin

        # Veri katmanlari — her biri opsiyonel
        self.robot_data = []          # list of {p, v, label, color}
        self.planned_trajs = {}       # robot_idx -> list of planned trajectories
        self.binary_data = []         # list of {binaries, obs_idx, label}

    def add_robot_trajectory(self, p_traj, v_traj, label=None, color=None):
        """Bir robotun gercek trajectory'sini ekle.

        Args:
            p_traj: pozisyon gecmisi, shape (T+1, 2)
            v_traj: hiz gecmisi, shape (T+1, 2)
            label: grafik etiketi (None ise otomatik)
            color: renk (None ise otomatik)
        """
        idx = len(self.robot_data)
        if label is None:
            label = f"Robot {idx + 1}"
        if color is None:
            color = COLORS[idx % len(COLORS)]
        self.robot_data.append({
            'p': np.array(p_traj),
            'v': np.array(v_traj),
            'label': label,
            'color': color,
        })

    def add_planned_trajectories(self, plans, robot_idx=0):
        """MPC planlanan trajectory listesini ekle.

        Args:
            plans: list of p_traj arrays (her MPC adimindaki plan)
            robot_idx: hangi robota ait
        """
        self.planned_trajs[robot_idx] = plans

    def add_binary_decisions(self, binaries, obs_idx=0, label=None):
        """Binary karar verisini ekle.

        Args:
            binaries: shape (n_obs, H, 4) veya (H, 4)
            obs_idx: hangi engel icin gosterilecek
            label: grafik etiketi
        """
        if binaries.ndim == 3:
            b = binaries[obs_idx]
        else:
            b = binaries
        if label is None:
            label = f"Obstacle {obs_idx + 1}"
        self.binary_data.append({'b': b, 'label': label})

    def show(self):
        """Interaktif animasyonu goster."""
        if not self.robot_data:
            raise ValueError("En az bir robot trajectory'si ekleyin.")

        # Toplam frame sayisi = en uzun trajectory
        n_frames = max(len(rd['p']) for rd in self.robot_data)
        time_all = np.arange(n_frames) * self.tau

        # Hangi paneller lazim?
        has_speed = True
        has_plans = bool(self.planned_trajs)
        has_binary = bool(self.binary_data)

        # Panel sayisi: trajectory + speed + (binary varsa)
        n_panels = 2 + (1 if has_binary else 0)

        state = {'frame': 0, 'playing': True}

        fig = plt.figure(figsize=(6 * n_panels, 6.5))
        gs = fig.add_gridspec(2, n_panels, height_ratios=[1, 0.08],
                              hspace=0.35, left=0.05, right=0.97,
                              top=0.92, bottom=0.12)

        axes = [fig.add_subplot(gs[0, i]) for i in range(n_panels)]
        ax_slider = fig.add_subplot(gs[1, :])

        ax_traj = axes[0]
        ax_speed = axes[1]
        ax_bin = axes[2] if has_binary else None

        # ====== Panel 1: Trajectory ======
        px_min, px_max, py_min, py_max = self.env.bounds
        ax_traj.set_xlim(px_min - 0.2, px_max + 0.2)
        ax_traj.set_ylim(py_min - 0.2, py_max + 0.2)
        ax_traj.set_aspect('equal')
        ax_traj.grid(True, alpha=0.3)
        ax_traj.set_xlabel('x (m)')
        ax_traj.set_ylabel('y (m)')
        ax_traj.set_title('Trajectory')

        # Sinirlar
        border = patches.Rectangle(
            (px_min, py_min), px_max - px_min, py_max - py_min,
            linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
        ax_traj.add_patch(border)

        # Engeller
        for obs in self.env.obstacles:
            w = 2 * obs.half_length
            h = 2 * obs.half_width
            rect = patches.Rectangle(
                (-obs.half_length, -obs.half_width), w, h,
                linewidth=1, edgecolor='dimgray', facecolor='gray', alpha=0.7)
            t = (plt.matplotlib.transforms.Affine2D()
                 .rotate(obs.angle)
                 .translate(obs.center[0], obs.center[1])
                 + ax_traj.transData)
            rect.set_transform(t)
            ax_traj.add_patch(rect)

        # Her robot icin: hedef, soluk trajectory, trail, circle
        robot_artists = []
        for i, rd in enumerate(self.robot_data):
            color = rd['color']
            p = rd['p']

            # Hedef (env.robots varsa)
            if i < len(self.env.robots):
                goal = self.env.robots[i].goal
                ax_traj.plot(*goal, '*', color=color, markersize=15,
                             markeredgecolor='black', markeredgewidth=0.5,
                             zorder=5)

            # Soluk tam trajectory
            ax_traj.plot(p[:, 0], p[:, 1], '-', color=color,
                         alpha=0.15, linewidth=1)

            # Trail (animasyonla dolan)
            trail, = ax_traj.plot([], [], '-o', color=color, markersize=2,
                                  linewidth=2, alpha=0.5, label=rd['label'])

            # Robot circle
            circle = patches.Circle((0, 0), self.dmin, fill=False,
                                     edgecolor=color, linewidth=2)
            ax_traj.add_patch(circle)

            # Planlanan trajectory (MPC)
            plan_line = None
            if i in self.planned_trajs:
                plan_line, = ax_traj.plot([], [], '--', color=color,
                                          linewidth=1.5, alpha=0.4)

            robot_artists.append({
                'trail': trail, 'circle': circle,
                'plan_line': plan_line, 'p': p,
            })

        ax_traj.legend(loc='upper right', fontsize=8)

        # Info text
        time_text = ax_traj.text(
            0.02, 0.95, '', transform=ax_traj.transAxes,
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

        # ====== Panel 2: Speed profile ======
        all_speeds = []
        speed_artists = []
        for rd in self.robot_data:
            s = np.linalg.norm(rd['v'], axis=1)
            all_speeds.append(s)

        max_speed = max(s.max() for s in all_speeds)
        ax_speed.set_xlim(0, time_all[-1] + self.tau)
        ax_speed.set_ylim(0, max_speed * 1.15 + 0.05)
        ax_speed.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5,
                         label='vmax')
        ax_speed.set_xlabel('time (s)')
        ax_speed.set_ylabel('speed (m/s)')
        ax_speed.set_title('Speed Profile')
        ax_speed.grid(True, alpha=0.3)

        for i, rd in enumerate(self.robot_data):
            line, = ax_speed.plot([], [], '-o', color=rd['color'],
                                  markersize=2, label=rd['label'])
            marker, = ax_speed.plot([], [], 'o', color=rd['color'],
                                    markersize=7, zorder=5,
                                    markeredgecolor='black',
                                    markeredgewidth=0.5)
            speed_artists.append({'line': line, 'marker': marker})
        ax_speed.legend(loc='upper right', fontsize=8)

        # ====== Panel 3: Binary decisions (opsiyonel) ======
        bin_artists = []
        if ax_bin and self.binary_data:
            dir_labels = ['right', 'top', 'left', 'bottom']
            dir_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red']
            bd = self.binary_data[0]  # ilk binary dataset
            b_data = bd['b']
            n_b_steps = len(b_data)
            time_b = np.arange(n_b_steps) * self.tau

            ax_bin.set_xlim(0, time_all[-1] + self.tau)
            ax_bin.set_ylim(-0.1, 1.3)
            ax_bin.set_xlabel('time (s)')
            ax_bin.set_ylabel('binary value')
            ax_bin.set_title(f'Binary Decisions ({bd["label"]})')
            ax_bin.set_yticks([0, 1])
            ax_bin.grid(True, alpha=0.3)

            for m in range(4):
                line, = ax_bin.plot([], [], '-o', color=dir_colors[m],
                                   markersize=3, label=dir_labels[m], alpha=0.7)
                bin_artists.append(line)
            ax_bin.legend(loc='upper right', fontsize=8)

        # ====== Slider ======
        slider = Slider(ax_slider, 'Step', 0, n_frames - 1, valinit=0,
                        valstep=1, color='tab:blue', alpha=0.5)

        # ====== Butonlar ======
        ax_play = fig.add_axes([0.35, 0.01, 0.08, 0.04])
        ax_prev = fig.add_axes([0.44, 0.01, 0.05, 0.04])
        ax_next = fig.add_axes([0.50, 0.01, 0.05, 0.04])
        ax_replay = fig.add_axes([0.56, 0.01, 0.08, 0.04])

        btn_play = Button(ax_play, 'Pause')
        btn_prev = Button(ax_prev, '<<')
        btn_next = Button(ax_next, '>>')
        btn_replay = Button(ax_replay, 'Replay')

        def draw_frame(frame):
            frame = int(np.clip(frame, 0, n_frames - 1))
            state['frame'] = frame

            info_lines = [f't = {frame * self.tau:.1f}s']

            # Her robot icin guncelle
            for i, ra in enumerate(robot_artists):
                p = ra['p']
                f = min(frame, len(p) - 1)

                ra['trail'].set_data(p[:f+1, 0], p[:f+1, 1])
                ra['circle'].center = (p[f, 0], p[f, 1])

                # Planlanan trajectory
                if ra['plan_line'] is not None and i in self.planned_trajs:
                    plans = self.planned_trajs[i]
                    if f < len(plans):
                        pt = plans[f]
                        ra['plan_line'].set_data(pt[:, 0], pt[:, 1])
                    else:
                        ra['plan_line'].set_data([], [])

                # Speed
                spd = all_speeds[i]
                sf = min(frame, len(spd) - 1)
                t_slice = time_all[:sf+1]
                speed_artists[i]['line'].set_data(t_slice, spd[:sf+1])
                speed_artists[i]['marker'].set_data(
                    [time_all[sf]], [spd[sf]])

                info_lines.append(
                    f'{self.robot_data[i]["label"]}: '
                    f'v={spd[sf]:.2f} m/s')

                # Hedefe mesafe
                if i < len(self.env.robots):
                    dist = np.linalg.norm(p[f] - self.env.robots[i].goal)
                    info_lines[-1] += f', d={dist:.2f}m'

            # Binary decisions
            if bin_artists and self.binary_data:
                b_data = self.binary_data[0]['b']
                k_end = min(frame, len(b_data))
                time_b = np.arange(k_end) * self.tau
                for m in range(4):
                    if k_end > 0:
                        bin_artists[m].set_data(time_b, b_data[:k_end, m])
                    else:
                        bin_artists[m].set_data([], [])

            time_text.set_text('\n'.join(info_lines))

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
            draw_frame(state['frame'] - 1)

        def on_next(event):
            state['playing'] = False
            btn_play.label.set_text('Play')
            draw_frame(state['frame'] + 1)

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


# --- Demo: onceki MPC senaryosunu Visualizer ile goster ---
if __name__ == "__main__":
    from environment import Environment
    from single_robot_micp import solve_single_robot_micp
    from mpc_loop import run_mpc

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
    env.add_robot(position=(-2.5, 0.3), goal=(0.0, 1.5))

    print("Running MPC...")
    p_hist, v_hist, u_hist, plans = run_mpc(
        env, robot_idx=0, H=H, tau=tau,
        vmax=vmax, amax=amax, dmin=dmin,
        max_steps=60, goal_tol=0.15
    )

    viz = Visualizer(env, tau=tau, dmin=dmin)
    viz.add_robot_trajectory(p_hist, v_hist, label="Robot 1")
    viz.add_planned_trajectories(plans, robot_idx=0)
    viz.show()
