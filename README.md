# DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments

<div align="center">

[![SLAM](https://img.shields.io/badge/SLAM-2D_LiDAR-blue.svg)]()
[![LiDAR](https://img.shields.io/badge/Sensor-360°_LiDAR-green.svg)]()
[![Simulator](https://img.shields.io/badge/Simulator-MuJoCo-red.svg)]()
[![Robot](https://img.shields.io/badge/Robot-MMK2-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

*MMK2 2D SLAM Learning Example — Built on [DISCOVERSE](https://github.com/TATP-233/DISCOVERSE) Simulation Platform*

*Contributed by [lesapulxw](https://github.com/lesapulxw)*

</div>

<div align="center">
<h1>🗺️ MMK2 2D SLAM Learning Example</h1>
</div>

A complete 2D SLAM learning routine for the MMK2 robot, featuring LiDAR scan matching, occupancy grid mapping, frontier-based active exploration, and autonomous navigation — all in MuJoCo simulation.

[中文文档](README_zh.md)

## 📦 Installation

```bash
git clone https://github.com/TATP-233/DISCOVERSE.git
cd DISCOVERSE
conda create -n discoverse python=3.10 && conda activate discoverse
pip install -e .
python scripts/setup_submodules.py
```

| Scenario | Install Command | Use Cases |
|----------|-----------------|-----------|
| Basic Simulation | `pip install -e .` | Learning, basic development |
| LiDAR SLAM | `pip install -e ".[lidar,visualization]"` | SLAM, navigation research |
| Imitation Learning | `pip install -e ".[act_full]"` | Robot skill learning |
| High-Fidelity Rendering | `pip install -e ".[gs]"` | Visual simulation, Real2Sim |

## 🗺️ MMK2 2D SLAM Learning Example

A complete 2D SLAM learning routine for the MMK2 robot, built on MuJoCo-LiDAR and DISCOVERSE simulation platform. *(Contributed by [lesapulxw](https://github.com/lesapulxw))*

### Features

- **2D LiDAR SLAM**: 360° single-line LiDAR + log-odds occupancy grid mapping with Bresenham ray casting
- **ICP Scan Matching**: KDTree nearest-neighbor + SVD rigid transform for odometry drift correction
- **Differential Drive Odometry**: Wheel kinematics with motion noise model
- **Active SLAM**: BFS frontier detection + A* path planning for autonomous exploration
- **Map-then-Navigate**: Two-phase workflow — manual mapping → click-to-navigate
- **ROS2 Bridge**: Publishes `/scan`, `/map`, `/planned_path`, TF tree (`map→odom→base_link→laser`)
- **Real-time GUI**: Matplotlib 2×2 debug panel (occupancy grid, lidar polar, trajectory, status)

### Quick Start

```bash
cd examples/MMK2_SLAM

# Keyboard-controlled SLAM (interactive)
python run_keyboard_slam.py --no-ros

# Autonomous frontier exploration
python run_active_slam.py --no-ros

# Map first, then click-to-navigate
python run_map_then_nav.py --no-ros
```

| Script | Description |
|--------|-------------|
| `run_keyboard_slam.py` | WASD manual control, real-time map building |
| `run_active_slam.py` | Autonomous exploration with frontier detection |
| `run_map_then_nav.py` | Two-phase: keyboard mapping → click-to-navigate |

See [examples/MMK2_SLAM/README.md](examples/MMK2_SLAM/README.md) for full documentation.

## ❔ Troubleshooting

For installation and runtime issues, please refer to our **[Troubleshooting Guide](discoverse/doc/troubleshooting.md)**.

## ⚖️ License

DISCOVERSE is released under the [MIT License](LICENSE).

## 📜 Citation

```bibtex
@article{jia2025discoverse,
    title={DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments},
    author={Yufei Jia and Guangyu Wang and Yuhang Dong and Junzhe Wu and Yupei Zeng and Haonan Lin and Zifan Wang and Haizhou Ge and Weibin Gu and Chuxuan Li and Ziming Wang and Yunjie Cheng and Wei Sui and Ruqi Huang and Guyue Zhou},
    journal={arXiv preprint arXiv:2507.21981},
    year={2025},
    url={https://arxiv.org/abs/2507.21981}
}
```
