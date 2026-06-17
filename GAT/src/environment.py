import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class Obstacle:
    def __init__(self, center, half_length, half_width, angle=0.0):
        self.center = np.array(center, dtype=float)  # (px_o, py_o)
        self.half_length = half_length                # L_o
        self.half_width = half_width                  # W_o
        self.angle = angle                            # alpha_o (radyan)

    def __repr__(self):
        return (f"Obstacle(center={self.center}, L={self.half_length}, "
                f"W={self.half_width}, angle={np.degrees(self.angle):.1f}deg)")

class Robot:
    def __init__(self, position, goal):
        self.position = np.array(position, dtype=float)  # (px, py)
        self.goal = np.array(goal, dtype=float)           # (px_g, py_g)

    def __repr__(self):
        return f"Robot(pos={self.position}, goal={self.goal})"

class Environment:
    def __init__(self, bounds):
        self.bounds = bounds # (px_min, px_max, py_min, py_max)
        self.obstacles = []
        self.robots = []
    
    def add_obstacle(self, center, half_length, half_width, angle=0.0):
        obs = Obstacle(center=center, half_length=half_length, half_width=half_width, angle=angle)
        self.obstacles.append(obs)
        return obs
    
    def add_robot(self, position, goal):
        robot = Robot(position, goal)
        self.robots.append(robot)
        return robot
    
    def plot(self, ax=None):
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


if __name__ == "__main__":
    env = Environment(bounds=(-3.0, 0.5, -0.5, 2.0))
    
    # Engeller (dikdörtgenler)
    env.add_obstacle(center=(-1.5, 0.7), half_length=0.3, half_width=0.15)
    env.add_obstacle(center=(-1.0, 1.3), half_length=0.2, half_width=0.2)
    env.add_obstacle(center=(-2.0, 1.0), half_length=0.15, half_width=0.25, angle=np.radians(30))

    # Robotlar: başlangıç → hedef
    env.add_robot(position=(-2.5, 0.2), goal=(-0.3, 1.5))
    env.add_robot(position=(-0.2, 0.3), goal=(-2.5, 1.2))
    env.add_robot(position=(-1.5, 1.8), goal=(-1.0, 0.1))
    
    env.plot()
    plt.tight_layout()
    plt.savefig('phase1_1_environment.png', dpi=150)
    plt.show()
    print("Environment saved: phase1_1_environment.png")