# MMK2 SLAM 学习例程 — 项目架构文档

## 1. 项目概述

本项目是基于 [DISCOVERSE](https://github.com/discoverse) 仿真平台的 **2D SLAM 学习例程**，使用 MMK2 机器人模型在 MuJoCo 物理仿真环境中实现完整的 SLAM 建图流程。

### 核心特性

- **三种运行模式**：
  - **键盘遥控 SLAM** (`run_keyboard_slam.py`)：WASD 手动控制，实时观察建图过程
  - **主动 SLAM** (`run_active_slam.py`)：前沿探索 + A* 路径规划，全自动自主建图
  - **先建图后导航** (`run_map_then_nav.py`)：两阶段——键盘建图 → 点击地图自主导航

- **完整 SLAM 流水线**：轮式里程计 → ICP 扫描匹配（scan-to-map）→ 占据栅格更新 → 前沿探索
- **Matplotlib 调试 GUI**：4 面板实时显示地图、扫描、轨迹、状态
- **可选 ROS2 桥接**：发布 `/scan`、`/map`、`/planned_path` 等话题，支持 RViz2 可视化

### 技术栈

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

---

## 2. 目录结构

```
MMK2_SLAM/
├── config/                     # 配置文件
│   ├── __init__.py
│   └── slam_config.py          # SLAM 全部参数（地图、里程计、ICP、探索、轨迹）
│
├── core/                       # 核心算法模块
│   ├── __init__.py
│   ├── slam_estimator.py       # SLAM 估计器（整合所有子模块的主流程）
│   ├── odometry.py             # 差速驱动轮式里程计
│   ├── scan_matching.py        # ICP 扫描匹配（KDTree + SVD）
│   ├── occupancy_grid.py       # 占据栅格地图（log-odds + Bresenham）
│   ├── frontier_exploration.py # 前沿检测 + A* 路径规划
│   └── trajectory.py           # 预定义轨迹生成器
│
├── robot/                      # 机器人封装
│   ├── __init__.py
│   ├── mmk2_slam_robot.py      # MMK2 机器人（LiDAR 集成 + 位姿获取 + 差速控制）
│   └── motion_controller.py    # 运动控制器（PID 跟踪预定义轨迹）
│
├── ros/                        # ROS2 桥接
│   ├── __init__.py
│   └── ros2_bridge.py          # 发布 LaserScan / Map / Path 等 ROS2 话题
│
├── gui/                        # 调试可视化
│   ├── __init__.py
│   └── slam_viewer_gui.py      # Matplotlib 2×2 实时调试面板
│
├── scenes/                     # MuJoCo 场景 XML
│   ├── slam_room_mmk2.xml      # 主场景（8m×6m 房间 + 障碍物 + MMK2 机器人）
│   ├── mmk2_slim.xml           # 简化 MMK2 模型（基础几何体，无 mesh 依赖）
│   ├── mmk2_dependencies_slim.xml  # MMK2 关节/执行器定义
│   ├── head.xml                # 头部模型
│   ├── arm_left.xml            # 左臂模型
│   └── arm_right.xml           # 右臂模型
│
├── rviz/                       # RViz2 配置
│   └── slam_2d.rviz            # RViz2 预设视图（LaserScan + Map + TF）
│
├── run_keyboard_slam.py        # 入口：键盘遥控 SLAM
├── run_active_slam.py          # 入口：主动探索 SLAM
├── run_map_then_nav.py         # 入口：先建图后导航
├── PROJECT_ARCHITECTURE.md     # 本文档
├── TROUBLESHOOTING.md          # 问题排查记录
└── SLAM_LEARNING_GUIDE.md      # SLAM 学习指南
```

### 各目录职责详解

| 目录 | 职责 | 关键文件 |
|------|------|----------|
| `config/` | 集中管理所有参数 | `slam_config.py` 继承 `MMK2Cfg`，定义地图分辨率、ICP 阈值、探索参数等 |
| `core/` | SLAM 算法核心 | `slam_estimator.py` 是中枢，协调里程计→ICP→地图→探索的完整流程 |
| `robot/` | 硬件抽象层 | `mmk2_slam_robot.py` 封装 MuJoCo-LiDAR 集成、MuJoCo 真值位姿获取、差速控制 |
| `ros/` | ROS2 通信层 | `ros2_bridge.py` 将内部数据转为 ROS2 消息，支持 `--no-ros` 纯 Python 模式 |
| `gui/` | 调试可视化 | `slam_viewer_gui.py` 用 `plt.ion()` 非阻塞模式嵌入 GLFW 主循环 |
| `scenes/` | 仿真场景 | 全部使用基础几何体（box/sphere/cylinder），无需 mesh 文件 |
| `rviz/` | RViz2 预设 | 预配置 Fixed Frame、LaserScan、Map、Path 显示 |

---

## 3. 核心模块架构

### 3.1 SLAM 估计器流程

```
┌─────────────────────────────────────────────────────────────┐
│                    SLAMEstimator.process_scan()              │
│                                                             │
│  ┌──────────┐    ┌──────────────┐    ┌──────────────────┐  │
│  │ 轮式里程计 │───>│ ICP 扫描匹配  │───>│ 占据栅格地图更新  │  │
│  │ (odometry)│    │ (scan-to-map) │    │ (occupancy_grid) │  │
│  └──────────┘    └──────────────┘    └──────────────────┘  │
│       │                  │                     │            │
│       │            ┌─────┴─────┐              │            │
│       │            │ 关键帧机制  │              │            │
│       │            └───────────┘              │            │
│       │                                       │            │
│       └───────────── 里程计同步 <──────────────┘            │
│                                                             │
│  ┌──────────────────┐                                       │
│  │ 前沿探索 (可选)    │ <── 仅主动 SLAM 模式使用              │
│  │ (frontier_explore)│                                      │
│  └──────────────────┘                                       │
└─────────────────────────────────────────────────────────────┘
```

**处理流程**：

1. **距离数组 → 2D 点云**：`ranges_to_points()` 将 `(ranges, angles)` 转为 `(N, 2)` 笛卡尔点云
2. **Scan-to-map ICP**（优先）：从地图提取机器人周围 5m 半径的占据点，将当前扫描变换到世界坐标系后与地图匹配
3. **Frame-to-frame ICP**（回退）：地图不够丰富时（< 20 占据点），匹配当前帧 vs 上一关键帧
4. **位姿校正**：ICP 得分 > 0.5 时应用校正，同步里程计
5. **地图更新**：用校正后位姿执行 Bresenham 光线投射更新占据栅格
6. **关键帧**：仅在移动 > 0.15m 或旋转 > 0.15rad 后更新参考扫描

### 3.2 差速驱动运动学模型

```python
# 差速运动学 (odometry.py)
d_center = (d_left + d_right) / 2.0    # 中心位移
d_theta  = (d_right - d_left) / L      # 角位移 (L = 轮距)

# 圆弧运动模型
if |d_theta| < ε:                       # 近似直线
    x += d_center * cos(θ)
    y += d_center * sin(θ)
else:                                   # 圆弧运动
    r = d_center / d_theta
    x += r * (sin(θ + dθ) - sin(θ))
    y += -r * (cos(θ + dθ) - cos(θ))
θ += d_theta
```

**关键参数**：
- 轮半径 `wheel_radius = 0.0838 m`
- 轮距 `wheel_distance = 0.3265 m`（实际 slim 模型驱动轮间距）
- 仿真模式下里程计噪声为 0（MuJoCo 提供完美轮子数据）

**控制 → 速度映射**：
```python
v_left  = (linear_vel - angular_vel * L / 2) / wheel_radius
v_right = (linear_vel + angular_vel * L / 2) / wheel_radius
```

### 3.3 ICP 配准算法

```
输入: source_points (N, 2), target_points (M, 2), init_pose [dx, dy, dθ]
输出: refined_pose [dx, dy, dθ], score ∈ [0, 1]

1. 构建 target 的 KDTree
2. 用 init_pose 变换 source → transformed
3. 迭代 (最多 50 次):
   a. KDTree 查询每个 transformed 点的最近邻
   b. 过滤距离 > 0.3m 的错误对应点
   c. 计算对应点对的质心
   d. 去质心 → SVD 分解 H = src^T · tgt → R = Vt^T · U^T
   e. 处理反射 (det(R) < 0 时翻转 Vt 最后一行)
   f. 求平移 t = centroid_tgt - R · centroid_src
   g. 累积变换到 current_pose
   h. 重新变换所有 source 点
   i. 检查收敛: |prev_error - mean_error| < 1e-4
4. 计算得分: score = 1 - mean_dist / max_correspondence_dist
5. 安全检查: 变化超过阈值 (0.5m / 0.3rad) 则拒绝匹配
```

**两种匹配模式**：

| 模式 | Source | Target | Init Pose | 适用场景 |
|------|--------|--------|-----------|----------|
| Scan-to-map | 当前扫描 (世界系) | 地图占据点 (世界系) | [0,0,0] | 地图有 ≥20 占据点 |
| Frame-to-frame | 当前扫描 (机器人系) | 上一关键帧 (机器人系) | 里程计位移 | 地图稀疏时回退 |

### 3.4 占据栅格地图

**Log-odds 表示法**：

$$l(x) = \log \frac{p(x)}{1 - p(x)}$$

- 初始值 0（未知）
- 空闲更新: `l += -1.0`（每帧射线穿过）
- 占据更新: `l -= (-1.0) + 2.0 = +1.0`（端点净增量）
- 裁剪范围: `[-100, 100]`
- 概率恢复: `p = 1 / (1 + exp(-l))`

**Bresenham 光线投射**：

对每条激光射线，用 Bresenham 直线算法遍历从机器人到端点的栅格路径：
- 路径上的栅格: `l += log_odd_free`（标记为空闲）
- 端点栅格: `l -= log_odd_free`（撤销空闲标记）+ `l += log_odd_occupied`（标记为占据）

**地图参数**：
- 分辨率: 0.05m/格
- 尺寸: 200×200 格 = 10m×10m
- 原点: (-5.0, -5.0)（地图中心 = 世界原点）

### 3.5 前沿检测与 A* 路径规划

**前沿检测** (`find_frontiers`)：
- 定义：空闲栅格中至少有一个**未知**邻居的栅格
- 算法：遍历所有空闲栅格，BFS 搜索连通区域，同时检测前沿点
- 过滤：前沿簇大小 < 5 格的丢弃

**目标选择** (`select_target`)：
- 计算每个前沿簇的质心（世界坐标）
- 选择距机器人最近的质心作为目标

**A* 路径规划** (`plan_path`)：
- 8 邻域扩展（对角线代价 √2）
- 代价设定：空闲格 = 1.0，未知格 = 5.0，占据格 = 不可通行
- 启发式：欧氏距离
- 目标被占据时自动搜索最近空闲格

---

## 4. ROS2 话题与 TF 树

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

| 话题 | 消息类型 | 频率 | 说明 |
|------|----------|------|------|
| `/scan` | `sensor_msgs/LaserScan` | 10 Hz | 2D 激光扫描（360 射线） |
| `/map` | `nav_msgs/OccupancyGrid` | 5 Hz | 占据栅格地图 (0-100, -1=未知) |
| `/map_metadata` | `nav_msgs/MapMetaData` | 1 Hz | 地图元数据（分辨率、尺寸、原点） |
| `/planned_path` | `nav_msgs/Path` | 事件触发 | 主动 SLAM 规划路径 |
| `/robot_trajectory` | `nav_msgs/Path` | 5 Hz | SLAM 估计的位姿轨迹 |

---

## 5. 依赖关系

### 外部依赖

```
DISCOVERSE 平台
├── discoverse.envs.simulator.SimulatorBase   # 仿真基类（MuJoCo 渲染、步进、键盘）
├── discoverse.robots_env.mmk2_base.MMK2Base  # MMK2 机器人（关节映射、传感器）
└── discoverse.robots_env.mmk2_base.MMK2Cfg   # 基础配置

MuJoCo-LiDAR 子模块
├── mujoco_lidar.lidar_wrapper.MjLidarWrapper  # LiDAR 封装（trace_rays, get_distances）
├── mujoco_lidar.scan_gen.create_lidar_single_line  # 360° 单线扫描模式
└── mujoco_lidar.core_cpu.mjlidar_cpu          # CPU 后端（mj_multiRay 调用）

MuJoCo 3.10.0
├── mj_multiRay()     # 多射线追踪（需 2D 列向量）
├── mj_step()         # 物理步进
├── mjv_defaultFreeCamera()  # 自由相机
└── MjvOption()       # 渲染选项（geomgroup 默认 [1,1,1,0,0,0]）
```

### 模块依赖图

```
run_keyboard_slam.py / run_active_slam.py / run_map_then_nav.py
    │
    ├──> config/slam_config.py (SLAMConfig)
    │
    ├──> robot/mmk2_slam_robot.py (MMK2SlamRobot)
    │        ├──> discoverse.robots_env.mmk2_base (MMK2Base)
    │        └──> mujoco_lidar (MjLidarWrapper)
    │
    ├──> core/slam_estimator.py (SLAMEstimator)
    │        ├──> core/odometry.py (DifferentialDriveOdometry)
    │        ├──> core/scan_matching.py (ICPMatcher)
    │        ├──> core/occupancy_grid.py (OccupancyGrid)
    │        └──> core/frontier_exploration.py (FrontierExplorer)
    │
    ├──> ros/ros2_bridge.py (SLAMROS2Bridge)  [可选]
    │
    └──> gui/slam_viewer_gui.py (SLAMViewerGUI)  [可选]
```
