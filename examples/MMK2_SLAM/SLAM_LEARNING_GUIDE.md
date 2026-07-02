# MMK2 SLAM 学习例程 — SLAM 学习指南

## 1. SLAM 基础概念

### 什么是 SLAM？

**SLAM**（Simultaneous Localization and Mapping，同时定位与地图构建）是机器人学的核心问题之一：

> 一个机器人被放置在一个**未知环境**中，它需要同时完成两件事：
> 1. **建图**（Mapping）：构建环境的地图
> 2. **定位**（Localization）：确定自己在地图中的位置

这两个问题互为鸡生蛋——建图需要知道机器人位置，定位需要知道地图。SLAM 算法就是解决这个"鸡生蛋"问题的方法。

### 为什么需要 SLAM？

| 场景 | 没有 SLAM | 有 SLAM |
|------|-----------|---------|
| 扫地机器人 | 随机碰撞，漏扫区域多 | 系统覆盖，高效清洁 |
| 自动驾驶 | 依赖 GPS + 高精地图 | 无 GPS 也能自主导航 |
| 仓储 AGV | 需要铺设磁条/二维码 | 自由规划路径 |
| 无人机 | 无法在室内飞行 | 室内自主避障导航 |

### SLAM 的数学本质：贝叶斯滤波

SLAM 的本质是**贝叶斯滤波**——在不确定性中递推估计状态。

**状态**：机器人位姿 \(x_t\) 和地图 \(m\)

**观测**：控制输入 \(u_t\)（轮子转速）和传感器测量 \(z_t\)（激光扫描）

**目标**：估计后验概率 \(p(x_t, m | z_{1:t}, u_{1:t})\)

贝叶斯滤波的两步递推：

**预测步**（用运动模型预测新状态）：
\[
p(x_t | x_{t-1}) = f(x_{t-1}, u_t) + \text{噪声}
\]

**更新步**（用传感器观测修正预测）：
\[
p(x_t | z_t) \propto p(z_t | x_t) \cdot p(x_t)
\]

本项目中的实现：
- **预测** = 轮式里程计（差速运动学积分）
- **更新** = ICP 扫描匹配（最小化扫描间差异）

---

## 2. 本项目涉及的 SLAM 算法详解

### 2.1 轮式里程计 (`core/odometry.py`)

#### 差速运动学模型

差速驱动机器人有两个独立驱动的轮子，通过左右轮速度差实现转向：

```
         L (轮距)
    ←─────────────→
   ┌───┐           ┌───┐
   │ L │           │ R │   ← 左右驱动轮
   └───┘           └───┘
         ↑
       机器人中心
```

**运动学方程**：

\[
v_{\text{center}} = \frac{v_L + v_R}{2}, \quad \omega = \frac{v_R - v_L}{L}
\]

其中 \(L\) 是轮距，\(v_L, v_R\) 是左右轮线速度。

**离散化更新**（圆弧运动模型）：

\[
\Delta s = \frac{\Delta s_L + \Delta s_R}{2}, \quad \Delta\theta = \frac{\Delta s_R - \Delta s_L}{L}
\]

当 \(\Delta\theta \approx 0\) 时（近似直线）：
\[
x_{t+1} = x_t + \Delta s \cdot \cos\theta_t, \quad y_{t+1} = y_t + \Delta s \cdot \sin\theta_t
\]

否则（圆弧运动）：
\[
x_{t+1} = x_t + \frac{\Delta s}{\Delta\theta} (\sin(\theta_t + \Delta\theta) - \sin\theta_t)
\]
\[
y_{t+1} = y_t - \frac{\Delta s}{\Delta\theta} (\cos(\theta_t + \Delta\theta) - \cos\theta_t)
\]

#### 协方差传播与噪声模型

真实里程计存在误差来源：轮子打滑、地面不平、编码器量化误差。

本项目使用 Probabilistic Robotics (Thrun et al.) 的运动噪声模型：

```python
# 噪声标准差与运动量成正比
σ_θ = α₁|Δθ| + α₂|Δs|      # 旋转噪声
σ_s = α₃|Δs| + α₄|Δθ|      # 平移噪声
```

本项目在仿真模式下将所有 α 设为 0（MuJoCo 提供完美数据），但保留了完整的噪声框架供学习参考。

#### 学习要点

- 里程计是**递推估计**，误差会**累积**（drift）
- 长时间运行后里程计位姿会偏离真实位姿
- 需要外部观测（如 ICP、GPS）来修正累积误差
- GUI 的轨迹面板可以对比 SLAM 轨迹 vs 里程计轨迹，直观看到漂移

---

### 2.2 ICP 扫描匹配 (`core/scan_matching.py`)

#### 原理

ICP（Iterative Closest Point）的目标是找到刚体变换 \(T = (R, t)\)，使得：

\[
\min_{R, t} \sum_{i} \| R \cdot s_i + t - t_i \|^2
\]

其中 \(s_i\) 是 source 点，\(t_i\) 是对应的 target 点。

#### 算法步骤

```
输入: source 点集 S, target 点集 T, 初始猜测 T₀
重复:
  1. 用当前变换 T 将 S 变换到 T 的坐标系
  2. 为每个变换后的 s_i 在 T 中找最近邻 (KDTree)
  3. 过滤距离过远的错误对应点
  4. 计算对应点对的质心 μ_s, μ_t
  5. 去质心: s' = s - μ_s, t' = t - μ_t
  6. SVD 分解: H = Σ s'·t'^T = UΣV^T → R = V^T U^T
  7. 求平移: t = μ_t - R · μ_s
  8. 累积变换: T ← (R, t) ∘ T
  9. 检查收敛: |误差变化| < ε
输出: 最优变换 T*, 匹配得分
```

#### KDTree 加速

朴素最近邻搜索是 O(N·M)，KDTree 将其加速到 O(N·log M)：

```python
from scipy.spatial import cKDTree
target_tree = cKDTree(target_points)  # 构建 KDTree
distances, indices = target_tree.query(transformed, k=1)  # 查询最近邻
```

#### SVD 求解刚体变换

关键步骤是求解最优旋转矩阵 R：

```python
# 交叉协方差矩阵
H = src_centered.T @ tgt_centered  # 2×2 矩阵

# SVD 分解
U, S, Vt = np.linalg.svd(H)

# 最优旋转
R = Vt.T @ U.T

# 处理反射 (det(R) < 0 时)
if np.linalg.det(R) < 0:
    Vt[-1, :] *= -1
    R = Vt.T @ U.T
```

#### 收敛条件

- **误差变化**：`|prev_error - mean_error| < tolerance`（默认 1e-4）
- **最大迭代**：50 次（防止死循环）
- **得分阈值**：`score > 0.5` 才接受匹配结果
- **变化范围**：平移 > 0.5m 或旋转 > 0.3rad 视为错误匹配

#### Scan-to-map vs Frame-to-frame

| 特性 | Frame-to-frame | Scan-to-map |
|------|----------------|-------------|
| 参考 | 上一帧扫描 | 累积地图 |
| 漂移 | 会累积 | 不累积 |
| 计算量 | 小 | 中等 |
| 鲁棒性 | 低（两帧可能差异大） | 高（地图是多帧平均） |
| 适用 | 地图稀疏时回退 | 地图有一定覆盖后 |

本项目优先使用 scan-to-map，地图不够丰富时回退到 frame-to-frame。

---

### 2.3 占据栅格地图 (`core/occupancy_grid.py`)

#### Log-odds 表示法

直接存储概率 \(p \in [0, 1]\) 的问题：多个观测的联合更新需要大量乘法。

Log-odds 变换：
\[
l = \log\frac{p}{1-p}, \quad p = \frac{1}{1+e^{-l}}
\]

优势：多个观测的更新变为**加法**：
\[
l_{\text{new}} = l_{\text{old}} + \log\frac{p(z|x)}{1-p(z|x)}
\]

本项目参数：
- 空闲更新量: `log_odd_free = -1.0`
- 占据更新量: `log_odd_occupied = 2.0`
- 裁剪范围: `[-100, 100]`（防止数值溢出）

#### Bresenham 光线投射

对每条激光射线，需要标记从机器人到命中点的路径为空闲、命中点为占据。

Bresenham 算法用整数运算高效遍历直线上的栅格：

```python
def _bresenham_update(self, x0, y0, x1, y1):
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx - dy

    while True:
        self.log_odds[y, x] += log_odd_free  # 路径标记空闲
        if (x, y) == (x1, y1):
            self.log_odds[y, x] -= log_odd_free  # 撤销空闲
            self.log_odds[y, x] += log_odd_occupied  # 标记占据
            break
        # Bresenham 步进
        e2 = 2 * err
        if e2 > -dy: err -= dy; x += sx
        if e2 < dx:  err += dx; y += sy
```

#### 逆传感器模型

每个栅格的更新量取决于它相对于激光射线的几何关系：
- **射线穿过的栅格**：空闲（`l += -1.0`）
- **射线端点的栅格**：占据（`l += +1.0`，净增量 = -1 + 1 + 2 - 1 = 1）
- **射线范围外的栅格**：不更新

多次扫描后，同一位置的占据栅格会累积到很高的 log-odds 值（如 +10），空闲栅格会累积到很低的值（如 -10），地图越来越确定。

---

### 2.4 前沿探索 (`core/frontier_exploration.py`)

#### 前沿检测

**前沿**定义：空闲栅格中至少有一个**未知**邻居的栅格。

这些栅格位于"已知"和"未知"区域的边界，是探索最有价值的目标。

```
■ ■ ■ ■ ■ ■
■ · · ? ? ■    ■ = 占据/墙壁
■ · R · ? ■    · = 空闲
■ · · · ? ■    R = 机器人
■ ■ ■ ■ ■ ■    ? = 未知 (前沿候选)
```

算法：BFS 遍历所有空闲栅格，检查每个空闲栅格的 8 邻居是否有未知栅格。

#### 前沿聚类

相邻的前沿点归为同一簇（BFS 连通分量），过滤掉太小的簇（< 5 格）。

#### 目标选择

计算每个前沿簇的质心，选择距机器人最近的质心作为目标。

#### A* 路径规划

在栅格地图上使用 A* 算法规划从当前位置到目标的路径：

- **代价函数**：空闲格 = 1.0，未知格 = 5.0（可以穿越但不优先），占据格 = 不可通行
- **启发式**：欧氏距离（admissible + consistent → 最优解）
- **8 邻域**：对角线代价 √2

---

## 3. 扩展学习方向

### 3.1 EKF-SLAM（扩展卡尔曼滤波 SLAM）

**原理**：将机器人位姿和地图特征点位置组成联合状态向量，用 EKF 递推估计。

**状态向量**：\([x_r, y_r, \theta_r, x_{m1}, y_{m1}, x_{m2}, y_{m2}, ...]^T\)

**优点**：理论优雅，有不确定性估计（协方差矩阵）

**缺点**：协方差矩阵 O(n²) 空间复杂度，特征点提取困难

**适用场景**：小规模环境、特征点明确（如室内角点）

---

### 3.2 FastSLAM（粒子滤波 SLAM）

**原理**：用一组粒子（采样）近似后验分布，每个粒子维护一份独立的地图。

**核心思想**：Rao-Blackwellized 分解——粒子采样机器人轨迹，每个粒子独立维护栅格地图。

**优点**：能处理多模态分布（如走廊对称性歧义）

**缺点**：粒子退化（需要大量粒子），高维空间效率低

**适用场景**：非线性强、非高斯噪声环境

---

### 3.3 Graph-SLAM / Factor Graph（图优化 SLAM）

**原理**：将 SLAM 建模为图优化问题——节点是位姿，边是约束（里程计、回环检测）。

**图结构**：
```
x₀ ──odom──→ x₁ ──odom──→ x₂
│              │              │
scan           scan           scan
│              │              │
m₀            m₁            m₂
              ↑______________↓
              loop closure (回环)
```

**优化目标**：最小化所有约束的加权误差平方和

**优点**：全局一致性（回环检测消除累积漂移），稀疏矩阵高效求解

**适用场景**：大规模环境、需要全局一致地图

**代表系统**：g2o, GTSAM, Ceres Solver

---

### 3.4 Cartographer（Google）

**原理**：子图匹配 + 回环检测的 2D/3D SLAM 系统。

**核心创新**：
1. **局部子图**：将连续扫描帧插入局部子图（submap），每个子图是独立的占据栅格
2. **分支定界匹配**：用多分辨率栅格加速全局匹配，检测回环
3. **后端优化**：SPA（Sparse Pose Adjustment）全局优化

**优点**：实时性好，2D/3D 统一框架，回环检测鲁棒

**适用场景**：室内移动机器人、大规模建筑

---

### 3.5 ORB-SLAM / VINS（视觉 SLAM）

**ORB-SLAM**：基于特征点的视觉 SLAM
- 提取 ORB 特征点 → 三角化 → PnP 定位 → 回环检测
- 三个并行线程：跟踪、局部建图、回环检测

**VINS（Visual-Inertial Navigation System）**：视觉惯性融合
- 融合相机图像 + IMU 数据
- 紧耦合优化，互补传感器特性

**适用场景**：无人机、AR/VR、无激光雷达的平台

---

### 3.6 LIO-SAM / FAST-LIO（激光惯性融合）

**LIO-SAM**：激光惯性里程计 + 因子图优化
- 融合 LiDAR 点云 + IMU 数据
- 因子图后端，支持回环检测

**FAST-LIO**：基于迭代卡尔曼滤波的紧耦合 LiDAR-惯性里程计
- 直接法（不需要特征提取）
- 增量式 ikd-Tree 维护局部地图

**适用场景**：自动驾驶、室外大场景、高速运动

---

## 4. 学习路径建议

### 阶段 1：感性认识（本项目三个 Demo）

```
1. 运行 run_keyboard_slam.py
   → 手动控制机器人，观察地图如何逐步构建
   → 注意：转弯时地图是否对齐？直线行驶时里程计是否漂移？

2. 运行 run_active_slam.py
   → 观察机器人如何自主发现前沿并前往探索
   → 注意：前沿检测 + A* 规划如何协作

3. 运行 run_map_then_nav.py
   → 先键盘建图，然后切换到导航模式
   → 点击地图设置目标，观察机器人自主导航
```

### 阶段 2：理解算法（阅读核心代码）

```
1. core/odometry.py
   → 理解差速运动学模型
   → 实验：增大噪声参数，观察里程计轨迹 vs SLAM 轨迹的差异

2. core/scan_matching.py
   → 理解 ICP 的 SVD 求解过程
   → 实验：修改 max_correspondence_dist，观察匹配成功率变化

3. core/occupancy_grid.py
   → 理解 log-odds 更新和 Bresenham 光线投射
   → 实验：修改 log_odd_free/log_odd_occupied，观察地图"确定度"变化

4. core/frontier_exploration.py
   → 理解前沿检测和 A* 规划
   → 实验：修改 frontier_min_size，观察探索行为变化
```

### 阶段 3：参数调优实验

修改 `config/slam_config.py` 中的参数，观察对建图效果的影响：

| 参数 | 调大 | 调小 |
|------|------|------|
| `map_resolution` | 地图更粗糙，速度快 | 地图更精细，速度慢 |
| `scan_match_min_score` | 更保守，少校正 | 更激进，多校正（但可能引入错误） |
| `scan_match_correspondence_dist` | 容忍更大偏差 | 更严格匹配 |
| `scan_match_keyframe_dist` | 更少关键帧，更快 | 更多关键帧，更精确 |
| `odom_noise_alpha*` | 里程计更嘈杂（模拟真实传感器） | 里程计更精确 |
| `map_log_odd_occupied` | 障碍物更快确认 | 需要更多观测确认 |
| `map_log_odd_free` | 空闲区域更快确认 | 需要更多观测确认 |
| `frontier_min_size` | 前沿更大，探索更粗粒度 | 前沿更小，探索更精细 |

### 阶段 4：深入学习（扩展方向）

```
1. 图优化 SLAM
   → 学习 g2o / GTSAM 框架
   → 实现简单的 2D 位姿图优化

2. 粒子滤波 SLAM
   → 实现 FastSLAM 1.0
   → 理解粒子退化和重采样

3. 视觉 SLAM
   → 学习 ORB-SLAM3 源码
   → 理解特征提取、三角化、PnP

4. 激光惯性融合
   → 学习 FAST-LIO 源码
   → 理解 IMU 预积分和紧耦合
```

---

## 5. 参考文献

1. Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
2. Yamauchi, B. (1997). "A Frontier-Based Approach for Autonomous Exploration."
3. Hart, P. E., Nilsson, N. J., & Raphael, B. (1968). "A Formal Basis for the Heuristic Determination of Minimum Cost Paths."
4. Zhang, J., & Singh, S. (2014). "LOAM: Lidar Odometry and Mapping in Real-time."
5. Qin, T., Li, P., & Shen, S. (2018). "VINS-Mono: A Robust and Versatile Monocular Visual-Inertial State Estimator."
6. Hess, W., et al. (2016). "Real-Time Loop Closure in 2D LIDAR SLAM." (Cartographer)
