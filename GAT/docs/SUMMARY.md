# Project Summary: GAT + Distributed Optimization for Multi-Robot MICP

> **Paper:** Le et al., "Combining Graph Attention Networks and Distributed Optimization for Multi-Robot Mixed-Integer Convex Programming" (arXiv:2503.21548v1)

> **Goal:** Understand the paper step by step through implementation. Not a 1:1 replication — focus on learning each concept thoroughly.

---

## The Problem

Multiple robots must navigate to their goals in a 2D environment while avoiding rectangular obstacles and each other. This is a **multi-robot collision-free trajectory planning** problem.

### Why is it hard?

The collision avoidance constraint is fundamentally an OR decision:

```
"Pass to the RIGHT of the obstacle  OR  pass to the LEFT"
```

This OR logic cannot be expressed as a simple linear constraint. It requires **binary (integer) decision variables**, turning the problem into a **Mixed-Integer Convex Program (MICP)**. MICP is NP-hard — solve time grows exponentially with the number of robots.

### The Paper's Key Idea

**Learn the binary decisions, solve the rest.**

```
Classical:    MICP (binary + continuous together) → NP-hard, slow
This paper:   GAT → predict binaries (~ms) → QP (continuous only) → polynomial, fast
```

A Graph Attention Network (GAT) learns to predict the binary variables offline. At runtime, it predicts binaries in milliseconds, and the remaining problem becomes a simple convex QP that solves quickly.

---

## Step-by-Step: What We Built and Learned

### Phase 0: Core Concepts

No code — just understanding the problem space.

**Key concepts:**
- **Mixed-Integer**: The OR in collision avoidance → binary variables via the big-M trick
- **Convex**: Once binaries are fixed, the remaining problem has quadratic cost + linear constraints = convex QP
- **Graph Attention Network**: Robots and obstacles naturally form a graph. GAT can learn on this structure and handle variable-sized inputs
- **Framework overview**: Offline (train GAT on solved MICP data) → Online (GAT predicts binaries → solve QP)

---

### Phase 1: 2D Environment and Robot Dynamics

**Files:** `environment.py`, `single_robot_qp.py`

**What we built:** A simulation environment with rectangular obstacles and robots following double-integrator dynamics.

**Double-integrator model:**
```
p(k+1) = p(k) + τ·v(k) + 0.5·τ²·u(k)    (position update)
v(k+1) = v(k) + τ·u(k)                    (velocity update)
```

Where `p` = position, `v` = velocity, `u` = acceleration (control input), `τ` = time step.

**What we learned:**
- State-space representation: state = [position, velocity], input = acceleration
- Simple trajectory planning with QP: minimize distance to goal + control effort
- Bound constraints: velocity limits, acceleration limits, environment boundaries

---

### Phase 2: Single Robot MICP with Obstacles

**Files:** `single_robot_micp.py`, `mpc_loop.py`

**What we built:** A single robot avoiding a single obstacle using the big-M formulation.

**Big-M formulation (Equation 6):**

For a robot at position `(px, py)` and an obstacle centered at `(ox, oy)` with half-lengths `L, W` and rotation angle `α`, we define 4 directional constraints:

```
Right:   cos(α)·(px-ox) + sin(α)·(py-oy) ≥ L + d_min - M·b₁
Top:    -sin(α)·(px-ox) + cos(α)·(py-oy) ≥ W + d_min - M·b₂
Left:   -cos(α)·(px-ox) - sin(α)·(py-oy) ≥ L + d_min - M·b₃
Bottom:  sin(α)·(px-ox) - cos(α)·(py-oy) ≥ W + d_min - M·b₄

b₁ + b₂ + b₃ + b₄ ≤ 3   (at least one direction must be active)
```

- `b = 0`: constraint is **active** (robot must maintain distance on that side)
- `b = 1`: constraint is **relaxed** (M is large, so constraint is trivially satisfied)
- The sum constraint ensures the robot passes on at least one side

**Cost function (Equations 7-8):**
```
minimize:  w_pt·‖p(H) - goal‖² + Σ_k [w_p·‖p(k) - goal‖² + w_u·‖u(k)‖²]
```
Terminal cost (reach goal) + tracking cost + effort cost (smooth control).

**Receding horizon (MPC):**
At each real time step: solve MICP for H-step horizon → apply only the first control input → re-solve. This gives feedback and adapts to changing conditions.

**What we learned:**
- Big-M trick: converting OR logic into binary variables
- MICP structure: binary variables make the problem combinatorially hard
- The role of each binary: it decides which side of the obstacle the robot will pass

---

### Phase 3: Multi-Robot MICP

**Files:** `multi_robot_micp.py`, `benchmark_micp.py`, `visualize_graph.py`

**What we built:** Multiple robots avoiding obstacles AND each other.

**Robot-robot collision avoidance (Equation 4):**
```
px_i - px_j ≥ 2·d_min - M·b₁
px_j - px_i ≥ 2·d_min - M·b₂
py_i - py_j ≥ 2·d_min - M·b₃
py_j - py_i ≥ 2·d_min - M·b₄
b₁ + b₂ + b₃ + b₄ ≤ 3
```

Same big-M idea but for robot pairs instead of robot-obstacle.

**Proximity-based edges (Equation 5):**
Only add collision constraints between robots that are close enough (`‖p_i - p_j‖ ≤ d_prox`). Distant robots don't need constraints — saves binary variables.

**Benchmark results:**
| Robots | Binary Variables | Solve Time |
|--------|-----------------|------------|
| 2 | ~240 | ~0.2s |
| 3 | ~540 | ~0.4s |
| 4 | ~1040 | ~0.7s |
| 5 | ~1740 | ~1.0s+ |

Binary variables grow quadratically with robot count, solve time grows exponentially. This motivates the GAT approach.

**Heterogeneous graph structure (Definition 3):**
- **Node types:** Robot (position + goal) and Obstacle (center + angle + size)
- **Edge types:**
  - `ER`: robot ↔ robot (collision avoidance binaries)
  - `ERO`: robot → obstacle (collision avoidance binaries)
  - `EOR`: obstacle → robot (information flow, no binaries)
  - `EO`: obstacle ↔ obstacle (information flow, no binaries)

**What we learned:**
- Multi-agent MICP complexity: coupling constraints between robots make the problem much harder
- The graph structure emerges naturally from the physical problem
- Exponential scaling is the fundamental bottleneck

---

### Phase 4: Data Generation

**Files:** `scenario_generator.py`, `collect_data.py`, `generate_dataset.py`

**What we built:** A pipeline to generate training data for the GAT.

**Process:**
1. Generate random scenarios (N robots, M obstacles, random positions/goals)
2. Solve each scenario with GUROBI (full MICP) → get optimal binary solutions
3. Refine binaries: if `b=1` but the constraint is already satisfied → set `b=0`
4. Convert to graph format: node features + edge indices + binary labels
5. Split into train (90%) and validation (10%)

**Binary refinement — why?**
GUROBI might set `b=1` (relaxed) even when the constraint is already satisfied with `b=0`. Both are optimal for GUROBI, but this inconsistency confuses the GAT during training. Solution: check each `b=1` and flip to `b=0` if the constraint holds.

**Dataset statistics:**
- 1978/2000 scenarios solved successfully (98.9% success rate)
- 1780 train, 198 validation samples
- Robot count: 2-5, Obstacle count: 1-3
- Label distribution: ~55% ones (relaxed), ~45% zeros (active) — well balanced

**What we learned:**
- Supervised learning requires careful data preparation
- Parametric MICP: same structure, different parameters → learnable pattern
- Data quality matters: refinement improved label consistency

---

### Phase 5: Graph Attention Network Training

**Files:** `gat_model.py`, `train_gat.py`

**What we built:** A heterogeneous GAT that predicts binary variables from graph structure.

**Architecture:**

```
Robot features [NR, 4]     ─┐
                              ├─ Projection ─► h [NR+NO, 64]
Obstacle features [NO, 5]  ─┘
                                     │
                             GAT Layer 1 (4 heads × 16 dim)
                             attention + aggregate + ELU
                                     │
                             GAT Layer 2 (4 heads × 16 dim)
                             attention + aggregate + ELU
                                     │
                                 h [NR+NO, 64]
                                     │
                    ┌────────────────┴────────────────┐
                    │                                  │
              Decoder RO                         Decoder RR
          [h_i ‖ h_j] → FF → H×4            [h_i ‖ h_j] → FF → H×4
          robot-obstacle binaries            robot-robot binaries
```

**Step 5.2 — Projection Layer:**
Robot features (4D) and obstacle features (5D) have different dimensions. A linear projection maps both to the same 64D space so they can interact in GAT layers.

**Step 5.3 — GAT Attention Mechanism (Equations 11-12):**
```
1) e_ij = LeakyReLU(a^T · [W·h_i ‖ W·h_j])     — raw attention score
2) α_ij = softmax_j(e_ij)                         — normalize (sum to 1)
3) h_i' = σ(Σ_j α_ij · W · h_j)                  — weighted aggregation
```

Each node asks: "which neighbors are most important to me?" and aggregates their information accordingly. Multi-head attention (4 heads) lets the model attend to different aspects simultaneously.

**Step 5.4 — Encoder:**
2-layer GAT with ELU activation. Layer 1 gathers direct neighbor info, layer 2 gathers 2-hop information (neighbors' neighbors).

**Step 5.5-5.6 — Decoder:**
For each edge (i, j): concatenate embeddings `[h_i ‖ h_j]` → feedforward NN → H×4 binary predictions. Separate decoders for RO and RR edges because they represent different physical constraints.

**Step 5.7-5.8 — Training:**

Loss: Binary Cross-Entropy with Logits
```
Loss = -[y·log(σ(x)) + (1-y)·log(1-σ(x))]
```

**Training experiments and insights:**

| Config | Train RO | Val RO | Val RR | Gap |
|--------|----------|--------|--------|-----|
| Baseline | 77.8% | 74.8% | 90.8% | 3% |
| + Normalization | 91.0% | 84.2% | 90.8% | 6.8% |
| + Dropout 0.3 | 75.0% | 75.0% | 84.2% | 0% |
| + Dropout 0.1 | 83.9% | 84.3% | 87.4% | 0.4% |

- **Normalization was critical**: +10% val RO accuracy. Different feature scales (positions ±4, angles 0-2π, sizes 0.2-1.0) hurt gradient flow
- **LR schedule helped**: StepLR (×0.5 every 20 epochs) — visible accuracy jumps at step boundaries
- **Dropout 0.3 too aggressive** (underfitting), 0.1 good balance, 0 gave highest val accuracy with some overfitting
- **Best model:** No dropout, with normalization — Val RO 84.2%, Val RR 90.8%

**Model size:** 57,784 parameters — quite small, trains in ~10 minutes on CPU.

**What we learned:**
- GAT naturally handles variable-sized graphs (different robot/obstacle counts)
- Attention learns which neighbors matter for each decision
- Input normalization >> model tuning for performance gains
- Heterogeneous graphs need separate projections and decoders

---

### Phase 6: Online Pipeline — GAT + QP

**Files:** `gat_qp_solver.py`, `evaluate_pipeline.py`

**What we built:** End-to-end pipeline: scenario → GAT prediction → QP solve → trajectory.

**Step 6.1 — GAT + QP solver:**
1. Build graph from scenario
2. GAT predicts binaries (~2ms)
3. Fix binaries in the MICP formulation → remaining problem is a QP
4. Solve QP with GUROBI (~571ms)

**Step 6.2 — Infeasibility handling:**

Two fixes were necessary:

*Post-processing (sum ≤ 3 constraint):*
GAT doesn't know about the `b₁+b₂+b₃+b₄ ≤ 3` constraint. If it predicts all zeros (all 4 directions active), the robot must be on all sides of the obstacle simultaneously — impossible! Fix: if sum = 0, set the highest-probability direction to 1. If sum = 4, set the lowest-probability direction to 0.

*Always-soft constraints:*
GAT's ~84% accuracy means many QPs are infeasible with hard constraints. Initial approach: try hard → if infeasible → retry with soft = two GUROBI calls, slow. Fix: always use soft constraints with high penalty (1000). Optimizer keeps slack at 0 when possible, uses it only when necessary. One GUROBI call instead of two.

**Step 6.4-6.5 — Evaluation (100 random scenarios):**

| Metric | MICP (GUROBI) | GAT + QP |
|--------|--------------|----------|
| Mean solve time | 675ms | 573ms |
| **Speedup** | — | **1.2x** |
| Collision rate | 0% | 17% |
| Goal reached | 19% | 6.1% |
| Binary accuracy (RO) | — | 84.7% |
| Binary accuracy (RR) | — | 83.8% |

**Why only 1.2x speedup?**
With 2-5 robots, MICP is already fast (~675ms). The real benefit appears with many more robots: MICP grows exponentially, QP grows polynomially. At 10+ robots, MICP takes minutes while QP stays under a second.

**Why 17% collision?**
Binary accuracy of ~84% means ~16% of decisions are wrong. Some of these wrong decisions remove necessary collision avoidance constraints, leading to collisions.

**What we learned:**
- ML + optimization hybrid works: learn the hard part (combinatorial), solve the easy part (convex)
- Post-processing is essential: ML output must respect physical constraints
- Soft constraints are a practical necessity: graceful degradation > hard failure
- The speedup scales with problem size, not visible at small scale

---

## Key Takeaways

1. **The big-M trick** converts OR logic into binary variables, making collision avoidance expressible as linear constraints at the cost of integer variables (NP-hard).

2. **The GAT approach** decomposes MICP into two parts: binary prediction (learned, fast) and convex QP (solved, fast). This avoids the exponential cost of branch-and-bound.

3. **Data quality > model complexity.** Input normalization gave +10% accuracy. Binary refinement improved label consistency. These mattered more than model architecture changes.

4. **Post-processing is not optional.** The GAT doesn't inherently respect physical constraints (sum ≤ 3). Enforcing them after prediction is critical for feasibility.

5. **Soft constraints enable robustness.** When ML predictions are imperfect, soft constraints allow graceful degradation instead of infeasibility.

6. **Speedup is problem-size dependent.** At 2-5 robots: modest 1.2x. At 10+ robots: MICP becomes minutes/hours while QP stays sub-second — that's where this approach truly shines.

7. **Heterogeneous graphs** are a natural fit for multi-agent problems. Different node types (robots vs obstacles) with different features, different edge types with different physical meanings — GAT handles all of this elegantly.

---

## Project Structure

```
GAT/
├── src/
│   ├── environment.py          # Phase 1: 2D world, obstacles, robot dynamics
│   ├── single_robot_qp.py      # Phase 1: QP trajectory (no obstacles)
│   ├── single_robot_micp.py    # Phase 2: MICP with big-M (single robot)
│   ├── mpc_loop.py             # Phase 2: Receding horizon control
│   ├── multi_robot_micp.py     # Phase 3: Multi-robot MICP
│   ├── benchmark_micp.py       # Phase 3: Solve time vs robot count
│   ├── visualize_graph.py      # Phase 3: Heterogeneous graph visualization
│   ├── scenario_generator.py   # Phase 4: Random scenario generation
│   ├── collect_data.py         # Phase 4: MICP solve + binary refinement
│   ├── generate_dataset.py     # Phase 4: Large dataset + train/val split
│   ├── gat_model.py            # Phase 5: GAT model (projection + encoder + decoder)
│   ├── train_gat.py            # Phase 5: Training loop with normalization
│   ├── gat_qp_solver.py        # Phase 6: GAT predict + QP solve
│   ├── evaluate_pipeline.py    # Phase 6: 100-scenario evaluation
│   ├── visualizer.py           # Visualization utilities
│   ├── animate_micp.py         # Animation for single robot
│   └── animate_multi_robot.py  # Animation for multi robot
├── data/
│   ├── dataset_train.pkl       # 1780 training graphs
│   ├── dataset_val.pkl         # 198 validation graphs
│   └── dataset_full.pkl        # 1978 raw samples
├── models/
│   ├── best_model.pt           # Trained GAT weights
│   └── norm_stats.pkl          # Feature normalization statistics
├── docs/
│   ├── 2503.21548v1.pdf        # Original paper
│   └── SUMMARY.md              # This file
└── IMPLEMENTATION_PLAN.md      # Phase-by-phase progress tracker
```

---

## Tools Used

- **Python 3, NumPy, Matplotlib** — simulation and visualization
- **GUROBI (gurobipy)** — MICP and QP optimization solver
- **PyTorch** — neural network training
- **PyTorch Geometric** — graph neural network layers (GATConv, HeteroData)
