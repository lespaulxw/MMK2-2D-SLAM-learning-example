# MMK2 2D SLAM 学习例程

基于 DISCOVERSE 平台的 MMK2 机器人和 MuJoCo-LiDAR 模块，构建完整的 2D SLAM 学习环境。

## 功能特点

- **三种运行模式**:
  - **键盘遥控 SLAM** (`run_keyboard_slam.py`)：WASD 手动控制，实时观察建图过程
  - **主动 SLAM** (`run_active_slam.py`)：前沿探索 + A* 路径规划，全自动自主建图
  - **先建图后导航** (`run_map_then_nav.py`)：两阶段——键盘建图 → 点击地图自主导航
- **完整 SLAM 流水线**：轮式里程计 → ICP 扫描匹配 → 占据栅格更新 → 前沿探索
- **双重可视化**: Matplotlib 调试 GUI + ROS2 RViz2
- **ROS2 可选**: 支持 `--no-ros` 纯 Python 模式

## 技术栈

| 组件 | 技术 |
|------|------|
| 物理仿真 | MuJoCo 3.10.0 |
| 激光雷达 | MuJoCo-LiDAR 子模块（CPU/Taichi 后端） |
| 扫描匹配 | ICP（KDTree + SVD） |
| 地图表示 | Log-odds 占据栅格 + Bresenham 光线投射 |
| 路径规划 | A* 算法 |
| 前沿探索 | BFS 聚类 + 最近前沿选择 |
| 可视化 | Matplotlib（非阻塞模式）/ RViz2 |
| 通信 | ROS2 Humble（可选） |

## 目录结构与模块职责

```
MMK2_SLAM/
├── config/                     # 配置管理
│   └── slam_config.py          # SLAM 全部参数（继承 MMK2Cfg）
├── core/                       # 核心算法模块
│   ├── slam_estimator.py       # SLAM 估计器（中枢，协调所有子模块）
│   ├── odometry.py             # 差速驱动轮式里程计
│   ├── scan_matching.py        # ICP 扫描匹配（KDTree + SVD）
│   ├── occupancy_grid.py       # 占据栅格地图（log-odds + Bresenham）
│   └── frontier_exploration.py # 前沿检测 + A* 路径规划
├── robot/                      # 机器人封装
│   ├── mmk2_slam_robot.py      # MMK2 机器人（LiDAR 集成 + 位姿获取 + 差速控制）
│   └── motion_controller.py    # 运动控制器（航点跟踪）
├── ros/                        # ROS2 通信层
│   └── ros2_bridge.py          # 发布 LaserScan / Map / Path / TF（支持 --no-ros）
├── gui/                        # 调试可视化
│   └── slam_viewer_gui.py      # Matplotlib 2×2 实时调试面板
├── scenes/                     # MuJoCo 场景 XML（全部基础几何体，无需 mesh）
│   ├── slam_room_mmk2.xml      # 主场景（8m×6m 房间 + 障碍物 + MMK2）
│   └── *.xml                   # MMK2 模型文件
├── rviz/
│   └── slam_2d.rviz            # RViz2 预设配置
├── run_keyboard_slam.py        # 入口：键盘遥控 SLAM
├── run_active_slam.py          # 入口：主动探索 SLAM
├── run_map_then_nav.py         # 入口：先建图后导航
└── README.md
```

## 核心算法流程

```
轮式里程计 ──→ 初始位姿估计
     │
     ▼
LiDAR 扫描 ──→ ICP 帧间匹配 ──→ 位姿校正
     │              │
     ▼              ▼
占据栅格地图 ←── 校正后位姿
     │
     ▼
前沿检测 ──→ A* 路径规划 ──→ 运动控制
```

### 各模块简介

| 模块 | 核心算法 | 说明 |
|------|----------|------|
| **轮式里程计** | 差速运动学 + 圆弧运动模型 | 从左右轮位移计算位姿变化，仿真模式下噪声为 0 |
| **ICP 扫描匹配** | KDTree 最近邻 + SVD 刚体变换 | 优先 scan-to-map（地图丰富时），回退 frame-to-frame（地图稀疏时），得分 > 0.5 才应用校正 |
| **占据栅格** | Log-odds 表示 + Bresenham 光线投射 | 分辨率 0.05m/格，空闲更新 -1.0，占据更新 +2.0，裁剪范围 [-100, 100] |
| **前沿探索** | BFS 连通检测 + A* 路径规划 | 前沿 = 有未知邻居的空闲格，选最近前沿簇质心为目标，8 邻域扩展 |

### 模块依赖关系

```
run_*.py (入口脚本)
    ├──> config/slam_config.py (SLAMConfig)
    ├──> robot/mmk2_slam_robot.py ──> discoverse.mmk2_base + mujoco_lidar
    ├──> core/slam_estimator.py ──> odometry + scan_matching + occupancy_grid + frontier
    ├──> ros/ros2_bridge.py [可选]
    └──> gui/slam_viewer_gui.py [可选]
```

## 安装依赖

### 1. DISCOVERSE 环境

```bash
conda activate discoverse
```

### 2. MuJoCo-LiDAR 子模块

```bash
cd DISCOVERSE
git submodule update --init submodules/MuJoCo-LiDAR
```

### 3. Python 依赖

```bash
pip install scipy matplotlib
```

### 4. ROS2 (可选, 用于 RViz 可视化)

```bash
sudo apt install ros-humble-desktop
source /opt/ros/humble/setup.bash
```

## 运行方式

### 键盘遥控 SLAM (交互式学习)

```bash
cd examples/MMK2_SLAM

# 纯 Python 模式 (无需 ROS2)
python run_keyboard_slam.py --no-ros

# ROS2 + RViz 模式
python run_keyboard_slam.py
# 另一终端:
rviz2 -d rviz/slam_2d.rviz
```

### 主动 SLAM (前沿探索)

```bash
python run_active_slam.py --no-ros
```

### 先建图后导航 (两阶段)

```bash
python run_map_then_nav.py --no-ros
# 阶段 1: WASD 键盘建图 → 按 N 切换
# 阶段 2: 点击地图面板设置目标 → 自主导航
```

## 键盘控制

| 按键 | 功能 |
|------|------|
| W / S | 前进 / 后退 |
| A / D | 左转 / 右转 |
| Shift | 按住加速 |
| N | 切换导航模式（主动 SLAM） |
| H | 显示帮助 |
| R | 重置状态 |
| ESC | 切换自由视角 |
| [ / ] | 切换相机 |

## ROS2 话题与 TF 树

### TF 树

```
map → odom → base_link → laser
```

| 变换 | 来源 | 说明 |
|------|------|------|
| `map → odom` | SLAM ICP 校正 | 校正量 = SLAM 估计位姿 - 里程计位姿 |
| `odom → base_link` | 轮式里程计 | 差速运动学积分 |
| `base_link → laser` | 固定安装偏移 | `pos=(0.09, 0, 0.215)` |

### ROS2 话题

| 话题 | 消息类型 | 说明 |
|------|----------|------|
| `/scan` | `sensor_msgs/LaserScan` | 2D 激光扫描（360 射线） |
| `/map` | `nav_msgs/OccupancyGrid` | 占据栅格地图 (0-100, -1=未知) |
| `/planned_path` | `nav_msgs/Path` | 主动 SLAM 规划路径 |
| `/robot_trajectory` | `nav_msgs/Path` | SLAM 估计的位姿轨迹 |

## 关键设计决策

1. **LaserScan 而非 PointCloud2**: `get_distances()` 返回 1D 距离数组，天然映射到 `LaserScan.ranges`，兼容 slam_toolbox
2. **实时位姿获取**: 使用 `mj_data.body("agv_link").xpos` 而非 `mj_model.body().pos`（机器人有 free joint 会移动）
3. **ICP 双模式**: scan-to-map（地图丰富）+ frame-to-frame（地图稀疏自动回退），计算量小适合实时
4. **Matplotlib GUI**: `plt.ion()` 非阻塞模式，可嵌入 GLFW 主循环
5. **ROS2 可选**: 所有 ROS2 代码 try/except，`--no-ros` 下纯 Python 运行

## 调试 GUI 面板

Matplotlib 2×2 布局:

| 占据栅格地图 | 激光扫描极坐标 |
|:---:|:---:|
| **轨迹俯视图** | **状态信息** |

- **占据栅格地图**: 红色=占据, 绿色=空闲, 机器人位置(蓝点) + 轨迹(青线) + 路径(黄虚线) + 目标(红星)
- **激光扫描**: 实时雷达扫描数据（机器人坐标系）
- **轨迹俯视图**: SLAM 估计轨迹(青线) vs 里程计轨迹(紫虚线)，可对比漂移
- **状态信息**: 扫描帧数、ICP 得分、地图覆盖率、机器人位姿等

## 学习路径

1. **键盘遥控 SLAM**: 手动控制机器人，观察地图如何随移动逐步构建
2. **主动 SLAM**: 观察机器人如何自主检测前沿并探索未知区域
3. **先建图后导航**: 建图完成后点击地图设置目标，体验完整导航流程
4. **参数调优**: 修改 `config/slam_config.py` 中的参数，观察对 SLAM 效果的影响：
   - `map_resolution`: 地图分辨率（越小越精细但计算量越大）
   - `scan_match_min_score`: ICP 匹配阈值（影响校正频率）
   - `odom_noise_alpha*`: 里程计噪声参数（影响里程计精度）
   - `frontier_min_size`: 前沿最小尺寸（影响探索粒度）

## 常见问题

### LiDAR 初始化失败

确保 MuJoCo-LiDAR 子模块已初始化:
```bash
git submodule update --init submodules/MuJoCo-LiDAR
```

### ROS2 不可用

使用 `--no-ros` 参数运行纯 Python 模式，仍可使用 Matplotlib GUI 调试。

### Matplotlib GUI 不显示

确保使用 `TkAgg` 后端（已默认设置）。如遇问题，尝试:
```bash
export MPLBACKEND=TkAgg
```

### 地图覆盖不全

- 增大 `map_width` 和 `map_height`（配置文件中）
- 调整 `map_origin` 使地图覆盖机器人活动范围
- 在主动 SLAM 中降低 `max_coverage` 参数

