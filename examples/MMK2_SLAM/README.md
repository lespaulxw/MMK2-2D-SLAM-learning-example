# MMK2 2D SLAM 学习例程

基于 DISCOVERSE 平台的 MMK2 机器人和 MuJoCo-LiDAR 模块，构建完整的 2D SLAM 学习环境。

## 功能特点

- **2D 激光雷达 SLAM**: 360° 单线 LiDAR + 占据栅格地图构建
- **轮式里程计**: 差速驱动运动学 + 运动噪声模型
- **ICP 扫描匹配**: KDTree 最近邻 + SVD 刚体变换，校正里程计漂移
- **前沿探索**: BFS 前沿检测 + A* 路径规划，实现自主探索
- **三种运行模式**: 键盘遥控建图、主动探索建图、先建图后导航
- **双重可视化**: Matplotlib 调试 GUI + ROS2 RViz2
- **ROS2 可选**: 支持 `--no-ros` 纯 Python 模式

## 目录结构

```
MMK2_SLAM/
├── config/
│   └── slam_config.py            # SLAM 参数配置 (继承 MMK2Cfg)
├── core/
│   ├── occupancy_grid.py         # 占据栅格地图 (log-odds + Bresenham)
│   ├── scan_matching.py          # ICP 扫描匹配 (KDTree + SVD)
│   ├── odometry.py               # 差速驱动轮式里程计
│   ├── frontier_exploration.py   # 前沿检测 + A* 路径规划
│   └── slam_estimator.py         # SLAM 估计器 (整合以上模块)
├── robot/
│   ├── mmk2_slam_robot.py        # MMK2 机器人 (继承 MMK2Base + LiDAR)
│   └── motion_controller.py      # 运动控制器 (航点跟踪)
├── ros/
│   └── ros2_bridge.py            # ROS2 桥接 (LaserScan/Map/Path/TF)
├── gui/
│   └── slam_viewer_gui.py        # Matplotlib 2×2 调试面板
├── scenes/
│   ├── slam_room_mmk2.xml        # SLAM 场景 (8m×6m 房间)
│   └── *.xml                     # MMK2 模型文件
├── rviz/
│   └── slam_2d.rviz              # RViz2 配置
├── run_keyboard_slam.py          # 键盘遥控 SLAM
├── run_active_slam.py            # 主动 SLAM (前沿探索)
├── run_map_then_nav.py           # 先建图后导航 (两阶段)
└── README.md
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
| H | 显示帮助 |
| R | 重置状态 |
| ESC | 切换自由视角 |
| [ / ] | 切换相机 |

## 架构说明

### SLAM 流程

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

### 关键设计决策

1. **LaserScan 而非 PointCloud2**: `get_distances()` 返回 1D 距离数组，天然映射到 `LaserScan.ranges`，兼容 slam_toolbox
2. **实时位姿获取**: 使用 `mj_data.body("agv_link").xpos` 而非 `mj_model.body().pos`（机器人有 free joint 会移动）
3. **ICP 帧间匹配**: 匹配当前帧 vs 上一帧（非全局地图），计算量小，适合实时学习
4. **Matplotlib GUI**: `plt.ion()` 非阻塞模式，可嵌入 GLFW 主循环
5. **ROS2 可选**: 所有 ROS2 代码 try/except，`--no-ros` 下纯 Python 运行

### TF 树

```
map → odom → base_link → laser
```

- `map → odom`: SLAM 校正量（ICP 校正后的位姿差）
- `odom → base_link`: 里程计位姿
- `base_link → laser`: 激光雷达安装偏移

### ROS2 话题

| 话题 | 消息类型 | 说明 |
|------|----------|------|
| `/scan` | `sensor_msgs/LaserScan` | 激光扫描 |
| `/map` | `nav_msgs/OccupancyGrid` | 占据栅格地图 |
| `/planned_path` | `nav_msgs/Path` | 规划路径 |
| `/robot_trajectory` | `nav_msgs/Path` | 机器人轨迹 |
| `/mujoco_scene` | `visualization_msgs/MarkerArray` | MuJoCo 场景 |

## 学习路径

1. **键盘遥控 SLAM**: 手动控制机器人，观察地图如何随移动逐步构建
2. **主动 SLAM**: 观察机器人如何自主检测前沿并探索未知区域
3. **先建图后导航**: 建图完成后点击地图设置目标，体验完整导航流程
4. **参数调优**: 修改 `config/slam_config.py` 中的参数，观察对 SLAM 效果的影响：
   - `map_resolution`: 地图分辨率（越小越精细但计算量越大）
   - `scan_match_min_score`: ICP 匹配阈值（影响校正频率）
   - `odom_noise_alpha*`: 里程计噪声参数（影响里程计精度）
   - `frontier_min_size`: 前沿最小尺寸（影响探索粒度）

## 调试 GUI 面板

Matplotlib 2×2 布局:

| 占据栅格地图 | 激光扫描极坐标 |
|:---:|:---:|
| **轨迹俯视图** | **状态信息** |

- **占据栅格地图**: 红色=占据, 绿色=空闲, 机器人位置(蓝点) + 轨迹(青线) + 路径(黄虚线) + 目标(红星)
- **激光扫描**: 实时雷达扫描数据（机器人坐标系）
- **轨迹俯视图**: SLAM 估计轨迹(青线) vs 里程计轨迹(紫虚线)，可对比漂移
- **状态信息**: 扫描帧数、ICP 得分、地图覆盖率、机器人位姿等

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
