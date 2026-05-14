# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

ROS 2 (`ament_python`) package `omni3_control` — kinematics and control for a 3-wheel omnidirectional robot driven by two RoboClaw motor controllers over USB.

## Build & Run

This package lives inside a colcon workspace (typical layout: `~/ros2_ws/src/omni3_control`). Build and source from the workspace root:

```bash
colcon build --packages-select omni3_control --symlink-install
source install/setup.bash
```

Run the nodes (entry points declared in `setup.py`):

```bash
ros2 run omni3_control kinematics_node      # cmd_vel ↔ wheel_speeds + odom
ros2 run omni3_control move_1m_node         # self-contained 1m drive demo
```

Quick standalone kinematics sanity check (no ROS needed):

```bash
python3 omni3_control/kinematics.py
```

Tests use `pytest` (`tests_require=['pytest']`) but no test files exist yet.

## Architecture

The package contains **two independent operating modes** that should not be mixed:

### 1. `kinematics_node.py` — Pure kinematics ROS node
- Subscribes `/cmd_vel` (Twist, world frame) → publishes `/wheel_speeds_qpps` (`Float64MultiArray`, QPPS for 3 wheels).
- Subscribes `/encoder_delta` (`Int32MultiArray`, tick deltas) → publishes `/odom`.
- Does **not** talk to hardware. Expects an external driver to consume `/wheel_speeds_qpps` and produce `/encoder_delta`. Note: `config/roboclaw_params.yaml` targets a `roboclaw_driver` node that **is not in this repo** — it must be supplied separately or `move_1m_node` used instead.

### 2. `move_1m_node.py` — Self-contained closed-loop demo
- Talks directly to two RoboClaws over USB (`/dev/roboclaw_front` @ 0x80, `/dev/roboclaw_rear` @ 0x81 — udev symlinks). Imports `omni3_control.roboclaw` (vendored RoboClaw driver).
- Wheel↔motor mapping is hardcoded: **W1 = RC-A M2 (β=−60°), W2 = RC-A M1 (β=+60°), W3 = RC-B M2 (β=180°)**. `DIR_W*` constants flip per-wheel sign.
- Encoder reading runs in its own thread (`_enc_reader`, ~50 Hz) so the 20 Hz control timer never blocks on serial I/O. Raw encoder reads are converted to **signed int32** before differencing.
- P-controller on world-frame position error → `forward_world()` → QPPS → `SpeedM1`/`SpeedM2`.

### `kinematics.py` — ROS-free math library
- `OmniKinematics` builds Jacobian `J[i] = [−sin β_i, cos β_i, L]` from `OmniParams(wheel_radius, robot_radius, beta)`.
- Convention: body frame `x→right, y→forward`, CCW positive; wheel angles `β = (−60°, +60°, 180°)`. The constraint `r·φ̇ = J · q̇` is the source of truth — both nodes call `forward_world` for FK and `J_inv @ (dphi * r)` for body-frame displacement from tick deltas.
- `det(J) = 3√3·L/2` (never singular for `L>0`).

## Critical constants (cross-file consistency required)

These values appear in multiple files and **must agree** with the physical robot, or odometry and control will silently diverge:

| Constant | Value | Where |
|---|---|---|
| `wheel_radius` | 0.05 m | `kinematics_node.py`, `move_1m_node.py`, `OmniParams` default |
| `robot_radius` | 0.27 m | same |
| `beta` | (−60°, +60°, 180°) | same |
| `COUNTS_PER_REV` | 750 | `kinematics_node.py`, `move_1m_node.py` |
| baud / addresses | 38400, 0x80 / 0x81 | `move_1m_node.py`, `encoder_test.py` |

Note: `config/roboclaw_params.yaml` lists `encoder_ticks_per_rev: 1440` and ports `/dev/ttyACM*` — these are for an **external** `roboclaw_driver` node, not the in-repo code, and are out of sync with the in-repo `COUNTS_PER_REV=750` and `/dev/roboclaw_{front,rear}` symlinks. Verify which driver is actually in use before changing either.

## Conventions

- All in-code comments and log strings are in Turkish — preserve that style when editing.
- Keep `kinematics.py` ROS-free (it is imported by both nodes and runnable standalone for testing).
