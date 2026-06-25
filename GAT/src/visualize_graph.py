import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from environment import Environment


def build_heterogeneous_graph(env, dprox=5.0):
    nodes = []

    for i, robot in enumerate(env.robots):
        nodes.append({
            'type': 'robot',
            'id': f'R{i+1}',
            'index': i,
            'feature': np.concatenate([robot.position, robot.goal]),
            'pos': robot.position.copy(),
        })

    for o, obs in enumerate(env.obstacles):
        nodes.append({
            'type': 'obstacle',
            'id': f'O{o+1}',
            'index': len(env.robots) + o,
            'feature': np.array([obs.center[0], obs.center[1],
                                  obs.angle, obs.half_length, obs.half_width]),
            'pos': obs.center.copy(),
        })

    NR = len(env.robots)
    NO = len(env.obstacles)

    # Edges
    edges = {'ER': [], 'ERO': [], 'EOR': [], 'EO': []}

    # ER: Robot-Robot
    for i in range(NR):
        for j in range(i + 1, NR):
            dist = np.linalg.norm(env.robots[i].position - env.robots[j].position)
            if dist <= dprox:
                edges['ER'].append((i, j))
                edges['ER'].append((j, i))

    # ERO: Robot -> Obstacle
    for i in range(NR):
        for o in range(NO):
            edges['ERO'].append((i, NR + o))

    # EOR: Obstacle -> Robot
    for i in range(NR):
        for o in range(NO):
            edges['EOR'].append((NR + o, i))

    # EO: Obstacle <-> Obstacle
    for o1 in range(NO):
        for o2 in range(NO):
            if o1 != o2:
                edges['EO'].append((NR + o1, NR + o2))

    return nodes, edges


def visualize_graph(env, nodes, edges):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    env.plot(ax=ax1)
    ax1.set_title('Physical Environment')

    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-1.5, 1.5)
    ax2.set_aspect('equal')
    ax2.set_title('Heterogeneous Graph Structure')
    ax2.axis('off')

    NR = len(env.robots)
    NO = len(env.obstacles)
    N = NR + NO

    node_pos = {}
    for i in range(NR):
        angle = np.pi/2 + np.pi * i / max(NR - 1, 1) if NR > 1 else np.pi/2
        node_pos[i] = np.array([-0.8, 1.0 - 2.0 * i / max(NR - 1, 1)]) if NR > 1 \
            else np.array([-0.8, 0.0])

    for o in range(NO):
        node_pos[NR + o] = np.array([0.8, 1.0 - 2.0 * o / max(NO - 1, 1)]) if NO > 1 \
            else np.array([0.8, 0.0])

    edge_styles = {
        'ER':  {'color': 'tab:blue',   'ls': '-',  'lw': 2.5, 'label': 'ER (Robot↔Robot)'},
        'ERO': {'color': 'tab:orange', 'ls': '-',  'lw': 2.0, 'label': 'ERO (Robot→Obstacle)'},
        'EOR': {'color': 'tab:green',  'ls': '--', 'lw': 1.5, 'label': 'EOR (Obstacle→Robot)'},
        'EO':  {'color': 'tab:purple', 'ls': ':',  'lw': 1.5, 'label': 'EO (Obstacle↔Obstacle)'},
    }

    drawn_labels = set()
    for etype, elist in edges.items():
        style = edge_styles[etype]
        for (src, dst) in elist:
            p1 = node_pos[src]
            p2 = node_pos[dst]
            mid = (p1 + p2) / 2
            label = style['label'] if etype not in drawn_labels else None
            drawn_labels.add(etype)

            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            shrink = 0.15
            length = np.sqrt(dx**2 + dy**2)
            if length > 0:
                ux, uy = dx/length, dy/length
                s = p1 + shrink * np.array([ux, uy])
                e = p2 - shrink * np.array([ux, uy])
            else:
                s, e = p1, p2

            ax2.annotate('', xy=e, xytext=s,
                         arrowprops=dict(arrowstyle='->' if etype in ('ERO', 'EOR') else '-',
                                         color=style['color'],
                                         linestyle=style['ls'],
                                         linewidth=style['lw'],
                                         alpha=0.6))

    for node in nodes:
        pos = node_pos[node['index']]
        if node['type'] == 'robot':
            circle = plt.Circle(pos, 0.12, color='tab:blue', alpha=0.8, zorder=5)
            ax2.add_patch(circle)
            ax2.text(pos[0], pos[1], node['id'], ha='center', va='center',
                     fontsize=10, fontweight='bold', color='white', zorder=6)
        else:
            rect = plt.Rectangle((pos[0]-0.12, pos[1]-0.10), 0.24, 0.20,
                                  color='gray', alpha=0.8, zorder=5)
            ax2.add_patch(rect)
            ax2.text(pos[0], pos[1], node['id'], ha='center', va='center',
                     fontsize=10, fontweight='bold', color='white', zorder=6)

    from matplotlib.lines import Line2D
    legend_elements = []
    for etype, style in edge_styles.items():
        legend_elements.append(
            Line2D([0], [0], color=style['color'], linestyle=style['ls'],
                   linewidth=style['lw'], label=style['label']))
    legend_elements.append(
        Line2D([0], [0], marker='o', color='w', markerfacecolor='tab:blue',
               markersize=12, label='Robot node'))
    legend_elements.append(
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray',
               markersize=12, label='Obstacle node'))
    ax2.legend(handles=legend_elements, loc='lower center',
               fontsize=8, ncol=2, bbox_to_anchor=(0.5, -0.15))

    info = "Node Features:\n"
    for node in nodes:
        f = node['feature']
        if node['type'] == 'robot':
            info += f"  {node['id']}: θ = [p=({f[0]:.1f},{f[1]:.1f}), g=({f[2]:.1f},{f[3]:.1f})]\n"
        else:
            info += f"  {node['id']}: θ = [c=({f[0]:.1f},{f[1]:.1f}), α={np.degrees(f[2]):.0f}°, L={f[3]:.2f}, W={f[4]:.2f}]\n"

    edge_info = "\nEdge Counts:\n"
    for etype, elist in edges.items():
        has_binary = etype in ('ER', 'ERO')
        edge_info += f"  {etype}: {len(elist)} edges {'(binary predictions)' if has_binary else '(info flow only)'}\n"

    ax2.text(0.0, -1.3, info + edge_info, ha='center', va='top',
             fontsize=7.5, family='monospace',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    plt.tight_layout()
    plt.savefig('../plots/phase3_graph_structure.png', dpi=150, bbox_inches='tight')
    plt.show()
    print("Saved: plots/phase3_graph_structure.png")


if __name__ == "__main__":
    env = Environment(bounds=(-3.0, 3.0, -1.0, 3.0))
    env.add_obstacle(center=(-0.5, 1.0), half_length=0.35, half_width=0.2)
    env.add_obstacle(center=(0.8, 1.8), half_length=0.25, half_width=0.2,
                     angle=np.radians(30))
    env.add_robot(position=(-2.5, 0.0), goal=(2.5, 2.5))
    env.add_robot(position=(2.5, 0.0), goal=(-2.5, 2.5))
    env.add_robot(position=(-2.0, 2.5), goal=(2.0, 0.0))

    nodes, edges = build_heterogeneous_graph(env, dprox=10.0)

    print("=== Heterogeneous Graph ===")
    print(f"Nodes: {len(nodes)} ({len(env.robots)} robots + {len(env.obstacles)} obstacles)")
    for etype, elist in edges.items():
        print(f"  {etype}: {len(elist)} edges")

    visualize_graph(env, nodes, edges)
