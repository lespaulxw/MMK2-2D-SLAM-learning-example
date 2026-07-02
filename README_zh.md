# DISCOVERSE: 复杂高保真环境中的高效机器人仿真

<div align="center">

[![SLAM](https://img.shields.io/badge/SLAM-2D_LiDAR-blue.svg)]()
[![LiDAR](https://img.shields.io/badge/Sensor-360°_LiDAR-green.svg)]()
[![Simulator](https://img.shields.io/badge/Simulator-MuJoCo-red.svg)]()
[![Robot](https://img.shields.io/badge/Robot-MMK2-orange.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)

*MMK2 2D SLAM 学习例程 — 基于 [DISCOVERSE](https://github.com/TATP-233/DISCOVERSE) 仿真平台*

*由 [lesapulxw](https://github.com/lesapulxw) 开发贡献*

</div>

<div align="center">
<h1>🗺️ MMK2 2D SLAM 学习例程</h1>
</div>

基于 MuJoCo 仿真的 MMK2 机器人完整 2D SLAM 学习例程，涵盖激光雷达扫描匹配、占据栅格地图构建、前沿自主探索与自主导航。

## 📦 安装

```bash
git clone https://github.com/TATP-233/DISCOVERSE.git
cd DISCOVERSE
conda create -n discoverse python=3.10 && conda activate discoverse
pip install -e .
python scripts/setup_submodules.py
```

| 场景 | 安装命令 | 适用 |
|------|----------|------|
| 基础仿真 | `pip install -e .` | 学习、基础开发 |
| 激光雷达 SLAM | `pip install -e ".[lidar,visualization]"` | SLAM、导航研究 |
| 机械臂模仿学习 | `pip install -e ".[act_full]"` | 机器人技能学习 |
| 高保真渲染 | `pip install -e ".[gs]"` | 视觉仿真、Real2Sim |

## 🗺️ MMK2 2D SLAM 学习例程

基于 MuJoCo-LiDAR 和 DISCOVERSE 仿真平台的完整 2D SLAM 学习例程。*（由 [lesapulxw](https://github.com/lesapulxw) 贡献）*

### 功能特性

- **2D 激光雷达 SLAM**：360° 单线 LiDAR + log-odds 占据栅格地图（Bresenham 光线投射）
- **ICP 扫描匹配**：KDTree 最近邻 + SVD 刚体变换，校正里程计漂移
- **差速驱动里程计**：轮式运动学 + 运动噪声模型
- **主动 SLAM**：BFS 前沿检测 + A* 路径规划，实现自主探索建图
- **先建图后导航**：两阶段流程 — 手动建图 → 点击地图自主导航
- **ROS2 桥接**：发布 `/scan`、`/map`、`/planned_path`，TF 树 `map→odom→base_link→laser`
- **实时可视化 GUI**：Matplotlib 2×2 调试面板（占据栅格、激光极坐标、轨迹、状态）

### 快速开始

```bash
cd examples/MMK2_SLAM

# 键盘控制手动建图
python run_keyboard_slam.py --no-ros

# 自主前沿探索建图
python run_active_slam.py --no-ros

# 先建图后自主导航
python run_map_then_nav.py --no-ros
```

| 脚本 | 说明 |
|------|------|
| `run_keyboard_slam.py` | WASD 键盘控制，实时观察建图过程 |
| `run_active_slam.py` | 自主检测前沿并探索未知区域，全自动建图 |
| `run_map_then_nav.py` | 两阶段：键盘建图 → 点击地图设置目标自主导航 |

完整文档请参见 [examples/MMK2_SLAM/README.md](examples/MMK2_SLAM/README.md)。

## ❔ 故障排除

有关安装和运行时问题，请参考 **[故障排除指南](discoverse/doc/troubleshooting.md)**。

## ⚖️ 许可证

DISCOVERSE 在 [MIT 许可证](LICENSE) 下发布。

## 📜 引用

```bibtex
@article{jia2025discoverse,
    title={DISCOVERSE: Efficient Robot Simulation in Complex High-Fidelity Environments},
    author={Yufei Jia and Guangyu Wang and Yuhang Dong and Junzhe Wu and Yupei Zeng and Haonan Lin and Zifan Wang and Haizhou Ge and Weibin Gu and Chuxuan Li and Ziming Wang and Yunjie Cheng and Wei Sui and Ruqi Huang and Guyue Zhou},
    journal={arXiv preprint arXiv:2507.21981},
    year={2025},
    url={https://arxiv.org/abs/2507.21981}
}
```
