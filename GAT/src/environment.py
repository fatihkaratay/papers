"""
Faz 1.1-1.2: 2D ortam + double-integrator robot dinamikleri.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


class Obstacle:
    """Dikdörtgen engel.

    Makaledeki parametreler (Section II-A, Denklem 6):
      - center: (px_o, py_o) — engelin merkezi
      - half_length: L_o — yarı uzunluk
      - half_width: W_o — yarı genişlik
      - angle: alpha_o — rotasyon açısı (radyan)
    """

    def __init__(self, center, half_length, half_width, angle=0.0):
        self.center = np.array(center, dtype=float)  # (px_o, py_o)
        self.half_length = half_length                # L_o
        self.half_width = half_width                  # W_o
        self.angle = angle                            # alpha_o (radyan)

    def __repr__(self):
        return (f"Obstacle(center={self.center}, L={self.half_length}, "
                f"W={self.half_width}, angle={np.degrees(self.angle):.1f}deg)")


class Robot:
    """Robot with double-integrator dynamics.

    Makaledeki parametreler:
      - State: x_i(k) = [p_i(k), v_i(k)]  (Section II-A)
      - Feature: theta_i = [p_i(t), p_goal_i]  (Section II-B)
      - Dynamics (Denklem 1):
          p(k+1) = p(k) + tau*v(k) + 0.5*tau^2*u(k)
          v(k+1) = v(k) + tau*u(k)
    """

    def __init__(self, position, goal, velocity=(0.0, 0.0)):
        self.position = np.array(position, dtype=float)  # p_i = (px, py)
        self.goal = np.array(goal, dtype=float)           # p_goal_i = (px_g, py_g)
        self.velocity = np.array(velocity, dtype=float)   # v_i = (vx, vy)
        # Trajectory history (for plotting)
        self.history = [self.position.copy()]

    def step(self, u, tau):
        """Apply control input u (acceleration) for one time step.

        Denklem (1):
          p(k+1) = p(k) + tau*v(k) + 0.5*tau^2*u(k)
          v(k+1) = v(k) + tau*u(k)

        Args:
            u: (ux, uy) acceleration input (m/s^2)
            tau: sampling time period (s)
        """
        u = np.array(u, dtype=float)
        self.position = self.position + tau * self.velocity + 0.5 * tau**2 * u
        self.velocity = self.velocity + tau * u
        self.history.append(self.position.copy())

    @property
    def state(self):
        """State vector x_i = [px, py, vx, vy]."""
        return np.concatenate([self.position, self.velocity])

    def __repr__(self):
        return (f"Robot(pos={self.position}, vel={self.velocity}, "
                f"goal={self.goal})")


class Environment:
    """2D ortam: sınırlar, engeller ve robotlar.

    Makaledeki parametreler (Section II-A, Denklem 2):
      - bounds: [px_min, px_max, py_min, py_max]
    """

    def __init__(self, bounds):
        """
        bounds: (px_min, px_max, py_min, py_max) — ortam sınırları
        """
        self.bounds = bounds  # (px_min, px_max, py_min, py_max)
        self.obstacles = []
        self.robots = []

    def add_obstacle(self, center, half_length, half_width, angle=0.0):
        obs = Obstacle(center, half_length, half_width, angle)
        self.obstacles.append(obs)
        return obs

    def add_robot(self, position, goal):
        robot = Robot(position, goal)
        self.robots.append(robot)
        return robot

    def plot(self, ax=None):
        """Ortamı çiz: sınırlar, engeller, robotlar ve hedefler."""
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 6))

        px_min, px_max, py_min, py_max = self.bounds

        # Ortam sınırları
        ax.set_xlim(px_min - 0.2, px_max + 0.2)
        ax.set_ylim(py_min - 0.2, py_max + 0.2)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('x (m)')
        ax.set_ylabel('y (m)')

        # Sınır çizgisi
        border = patches.Rectangle(
            (px_min, py_min), px_max - px_min, py_max - py_min,
            linewidth=2, edgecolor='black', facecolor='none', linestyle='--'
        )
        ax.add_patch(border)

        # Engeller — döndürülmüş dikdörtgenler
        for obs in self.obstacles:
            # Dikdörtgenin sol-alt köşesi (döndürme öncesi, merkeze göre)
            width = 2 * obs.half_length
            height = 2 * obs.half_width
            rect = patches.Rectangle(
                (-obs.half_length, -obs.half_width), width, height,
                linewidth=1, edgecolor='dimgray', facecolor='gray', alpha=0.7
            )
            # Döndür ve merkeze taşı
            t = (plt.matplotlib.transforms.Affine2D()
                 .rotate(obs.angle)
                 .translate(obs.center[0], obs.center[1])
                 + ax.transData)
            rect.set_transform(t)
            ax.add_patch(rect)

        # Robotlar ve hedefleri
        colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']
        for idx, robot in enumerate(self.robots):
            color = colors[idx % len(colors)]
            # Robot pozisyonu (dolu daire)
            ax.plot(*robot.position, 'o', color=color, markersize=10,
                    label=f'Robot {idx + 1}')
            # Hedef (yıldız)
            ax.plot(*robot.goal, '*', color=color, markersize=15,
                    markeredgecolor='black', markeredgewidth=0.5)
            # Başlangıç → hedef ok
            ax.annotate('', xy=robot.goal, xytext=robot.position,
                        arrowprops=dict(arrowstyle='->', color=color,
                                        linestyle=':', alpha=0.4))

        ax.legend(loc='upper right')
        ax.set_title('Multi-Robot Navigation Environment')
        return ax


# --- Demo: double-integrator dynamics test ---
if __name__ == "__main__":
    tau = 0.2  # sampling period (s) — makaledeki deger: 200ms
    env = Environment(bounds=(0.0, 4.0, 0.0, 3.0))

    # Tek robot: (0.5, 0.5) -> (3.5, 2.5)
    robot = env.add_robot(position=(0.5, 0.5), goal=(3.5, 2.5))

    # Elle sabit ivme uygula: hedefe dogru 30 adim
    # Basit strateji: ilk 15 adim hizlan, son 15 adim yavasla
    n_steps = 30
    direction = robot.goal - robot.position
    direction = direction / np.linalg.norm(direction)

    for k in range(n_steps):
        if k < n_steps // 2:
            u = 0.3 * direction   # hizlan
        else:
            u = -0.3 * direction  # yavasla
        robot.step(u, tau)

    # Trajectory ciz
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Sol: ortam + trajectory
    env.plot(ax=ax1)
    history = np.array(robot.history)
    ax1.plot(history[:, 0], history[:, 1], '-', color='tab:blue', linewidth=2,
             alpha=0.7, label='trajectory')
    ax1.set_title('Robot Trajectory (double-integrator)')
    ax1.legend()

    # Sag: hiz profili
    # Hizi yeniden hesapla (history'den)
    speeds = [np.linalg.norm(history[i+1] - history[i]) / tau
              for i in range(len(history) - 1)]
    ax2.plot(np.arange(len(speeds)) * tau, speeds, '-o', markersize=3)
    ax2.set_xlabel('time (s)')
    ax2.set_ylabel('speed (m/s)')
    ax2.set_title('Speed Profile')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('../plots/faz1_2_dynamics.png', dpi=150)
    plt.show()
    print("Dynamics demo saved: plots/faz1_2_dynamics.png")
