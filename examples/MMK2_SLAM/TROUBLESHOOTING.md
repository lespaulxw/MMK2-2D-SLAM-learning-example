# MMK2 SLAM 学习例程 — 问题排查文档

本文档记录了项目开发过程中遇到的所有问题、根因分析和修复方法。

---

## 目录

1. [MuJoCo 3.10.0 `mj_multiRay()` API 兼容性](#1-mujoco-3100-mj_multiray-api-兼容性)
2. [GLFW 键盘回调注册方式](#2-glfw-键盘回调注册方式)
3. [Matplotlib 中文字体缺失 Glyph Warning](#3-matplotlib-中文字体缺失-glyph-warning)
4. [占据栅格全绿无红色障碍物](#4-占据栅格全绿无红色障碍物)
5. [机器人不可见](#5-机器人不可见)
6. [相机角度异常](#6-相机角度异常)
7. [机器人无法转向](#7-机器人无法转向)
8. [建图漂移](#8-建图漂移)
9. [探索目标点频繁切换](#9-探索目标点频繁切换)

---

## 1. MuJoCo 3.10.0 `mj_multiRay()` API 兼容性

### 错误现象

```
TypeError: mj_multiRay(): incompatible function arguments.
The following argument types are supported:
    1. vec: numpy.ndarray[float64[3, 1]]
    2. dist: numpy.ndarray[float64[m, 1]]
    3. geomid: numpy.ndarray[int32[m, 1]]
```

传入 1D 数组 `(N,)` 时报类型不匹配，改为 2D 列向量 `(N, 1)` 后仍然报错。

### 根因分析

MuJoCo 3.10.0 的 pybind11 绑定对 `mj_multiRay()` 的参数类型有严格要求：

1. **`vec` 参数**：必须是 2D 列向量 `[m, 1]` 的 flat 形式，即 `[3*m, 1]`
2. **`dist` 和 `geomid`**：必须是 `[m, 1]` 形状的 2D 数组
3. **`normal` 参数**：类型签名为 `| None`，但 pybind11 要求**显式传递** `None`
4. **`flg_static` 参数**：必须是 `bool` 类型，不能是 `int`
5. **数组内存布局**：pybind11 不接受 numpy view（如 `.reshape(-1, 1)`），必须是连续内存的独立数组

### 修复方法

完全重写 `mjlidar_cpu.py` 中的射线追踪逻辑：

```python
# 正确: 直接创建 2D 列向量 (连续内存)
self._dist = np.full((_nray, 1), self.cutoff_dist, dtype=np.float64)
_geomid = np.zeros((_nray, 1), dtype=np.int32)

# 错误: view 不被 pybind11 接受
# self._dist = np.full(_nray, self.cutoff_dist).reshape(-1, 1)

mujoco.mj_multiRay(
    m=self.mj_model, d=self.mj_data, pnt=pnt,
    vec=world_vecs_flat,
    geomgroup=self.geomgroup, flg_static=True,     # bool, 非 int
    bodyexclude=self.bodyexclude, geomid=_geomid,
    dist=self._dist, normal=None,                   # 必须显式传 None
    nray=_nray, cutoff=self.cutoff_dist,
)
```

`get_distances()` 返回 `.flatten()` 保持下游 1D 兼容。

### 涉及文件

- `submodules/MuJoCo-LiDAR/mujoco_lidar/core_cpu/mjlidar_cpu.py`

---

## 2. GLFW 键盘回调注册方式

### 错误现象

```
AttributeError: 'LP__GLFWwindow' object has no attribute 'set_key_callback'
```

### 根因分析

GLFW 的 Python 绑定中，`GLFWwindow` 是一个 C 指针类型（`LP__GLFWwindow`），不支持直接调用方法。回调注册必须通过模块级函数 `glfw.set_key_callback(window, callback)` 完成，而非 `window.set_key_callback(callback)`。

### 修复方法

```python
# 错误
robot.window.set_key_callback(on_key)

# 正确
glfw.set_key_callback(robot.window, on_key)
```

### 涉及文件

- `run_keyboard_slam.py`

---

## 3. Matplotlib 中文字体缺失 Glyph Warning

### 错误现象

数百条警告刷屏：

```
UserWarning: Glyph 19977 (CJK 字符 '三') missing from font(s) DejaVu Sans.
```

同时报错：

```
ValueError: Attempting to set identical low and high xlims makes transformation singular
```

### 根因分析

**问题 1**：Matplotlib 默认字体 DejaVu Sans 不包含 CJK（中日韩）字符。GUI 中使用了中文标签，每个中文字符都触发一条 Glyph missing 警告。

**问题 2**：`set_xlim(extent[0], extent[2])` 中 extent 索引错误，当 extent 为 `[0, 0, 0, 0]` 时（地图尚未更新），`xlim` 的 low 和 high 相同导致奇异变换。

### 修复方法

1. **所有 GUI 标签改为英文**，避免中文字体依赖
2. **添加 CJK 字体检测**：启动时检测可用 CJK 字体，若无则提示安装
3. **抑制 Glyph 警告**：`warnings.filterwarnings('ignore', message='Glyph .* missing')`
4. **修复 extent 索引**：`set_xlim(extent[0], extent[1])`，`set_ylim(extent[2], extent[3])`

### 涉及文件

- `gui/slam_viewer_gui.py`

---

## 4. 占据栅格全绿无红色障碍物

### 错误现象

地图上所有区域都显示为绿色（空闲），看不到任何红色（占据）区域。

### 根因分析

`_bresenham_update()` 中端点占据标记的符号错误：

```python
# 当射线命中障碍物时，端点先被标记为空闲 (log_odd_free = -1.0)
self.log_odds[y, x] += self.log_odd_free  # -1.0

# 然后需要"撤销"空闲标记并标记为占据
# 错误: += self.log_odd_free → -1.0 + (-1.0) = -2.0 (更空闲了!)
# 正确: -= self.log_odd_free → -1.0 - (-1.0) = 0.0 (撤销空闲)
# 然后: += log_odd_occupied → 0.0 + 2.0 = 2.0 (标记占据)
```

错误代码中端点最终值 = -1 + (-1) + 2 = 0（中性），导致障碍物永远无法被标记为占据。

### 修复方法

```python
# 修复前
self.log_odds[y, x] += self.log_odd_free  # BUG: 又加了一次 free

# 修复后
self.log_odds[y, x] -= self.log_odd_free  # 撤销空闲标记
```

修复后端点值 = -1 - (-1) + 2 = 2.0（正确占据）。

### 涉及文件

- `core/occupancy_grid.py`（`_bresenham_update` 方法，第 157 行）

---

## 5. 机器人不可见

### 错误现象

能看到房间（墙壁、地面、障碍物），但完全看不到机器人。

### 根因分析

MuJoCo 的 `MjvOption()` 默认 `geomgroup = [1, 1, 1, 0, 0, 0]`，即 **group 3/4/5 默认不可见**。

所有机器人 XML 文件中的 geom 都设置了 `group="4"`：

```xml
<geom type="box" rgba="1 0 0 1" group="4" .../>
```

而房间墙体 geom 没有 group 属性（默认 group=0，可见），所以能看到房间但看不到机器人。

### 修复方法

将所有机器人 XML 文件中的 `group="4"` 改为 `group="0"`：

```bash
# 影响范围
scenes/mmk2_slim.xml  — 7 处 (底盘、车轮、升降机构)
scenes/head.xml       — 1 处 (头部)
scenes/arm_left.xml   — 7 处 (左臂各连杆)
scenes/arm_right.xml  — 7 处 (右臂各连杆)
```

### 涉及文件

- `scenes/mmk2_slim.xml`
- `scenes/head.xml`
- `scenes/arm_left.xml`
- `scenes/arm_right.xml`

---

## 6. 相机角度异常

### 错误现象

机器人模型显示异常（严重扭曲/翻转），或完全看不到。

### 根因分析

场景 XML 使用了 `<compiler angle="radian"/>`，这意味着**所有角度属性都是弧度**，包括相机的 `euler` 属性。

错误地使用了角度值：

```xml
<!-- 错误: 50 弧度 ≈ 2865° (完全无意义的旋转) -->
<camera name="overview" pos="0 -8 5" euler="50 0 0" fovy="45"/>

<!-- 错误: 90 弧度 ≈ 5157° -->
<camera name="top_down" pos="0 0 10" euler="90 0 0" fovy="45"/>
```

### 修复方法

```xml
<!-- 正确: 1.0 弧度 ≈ 57° (俯视倾斜角度) -->
<camera name="overview" pos="0 -8 5" euler="1.0 0 0" fovy="45"/>

<!-- 正确: 0 弧度 = 正上方俯视 -->
<camera name="top_down" pos="0 0 10" euler="0 0 0" fovy="45"/>
```

**经验法则**：在 `<compiler angle="radian">` 下，`euler="0 0 0"` 的相机看向 -Z 方向（正下方）。

### 涉及文件

- `scenes/slam_room_mmk2.xml`

---

## 7. 机器人无法转向

### 错误现象

WASD 控制中 W/S 可以前进后退，但 A/D 键无法使机器人转向。

### 根因分析

**问题 1**：主循环中存在冗余的 `robot.updateControlFromKeyboard(robot.key_states)` 调用，该方法直接设置 `ctrl[0:2]`，但随后被 `robot.step(action)` 中的 `updateControl(action)` 覆盖。

**问题 2**：MMK2 的轮子执行器是 `motor` 类型（扭矩控制），不是 `velocity` 类型。`ctrl=1.0` 只产生 ±1.127 Nm 的扭矩，不足以克服 4 轮摩擦力使机器人转向。

### 修复方法

1. **移除冗余调用**：删除主循环中的 `robot.updateControlFromKeyboard(robot.key_states)`
2. **增大角速度**：`angular_vel` 从 1.0 提高到 2.0（扭矩 ±2.255 Nm，翻倍）

```python
# 修复前
angular_vel = 1.0 * speed_factor

# 修复后
angular_vel = 2.0 * speed_factor
```

### 涉及文件

- `run_keyboard_slam.py`（主循环）
- `robot/mmk2_slam_robot.py`（`updateControlFromKeyboard` 方法）

---

## 8. 建图漂移

### 错误现象

移动过程中，已建立好的障碍物在地图上重复出现、位置不稳定，地图整体呈现“重影”效果。

### 根因分析

漂移由三个独立问题叠加导致：

#### 问题 1：轮距参数错误（贡献 ~70% 漂移）

```
配置值: wheel_distance = 0.189 m
实际值: wheel_distance = 0.3265 m (lft_wheel y=+0.16325, rgt_wheel y=-0.16325)
```

`MMK2Base` 基类中 `wheel_distance = 0.189` 对应的是完整 MMK2 模型的前轮间距，但 slim 模型的驱动轮（`lft_wheel_joint`/`rgt_wheel_joint`）实际间距为 0.3265m。

**影响**：里程计高估旋转 **72.8%**（`0.3265 / 0.189 = 1.73x`），每次转弯都产生严重角度偏差。

#### 问题 2：人工里程计噪声（贡献 ~20% 漂移）

```python
# 仿真中 MuJoCo 提供完美轮子数据，但代码额外加了 5% 高斯噪声
odom_noise_alpha1 = 0.05  # 旋转->旋转噪声
odom_noise_alpha2 = 0.05  # 旋转->平移噪声
```

#### 问题 3：帧间 ICP 累积误差（贡献 ~10% 漂移）

原始方案匹配当前帧 vs 上一帧（frame-to-frame），每次匹配的小误差不断累积。

### 修复方法

**修复 1**：修正轮距参数

```python
# config/slam_config.py
odom_wheel_distance = 0.3265  # was 0.189

# robot/mmk2_slam_robot.py
class MMK2SlamRobot(MMK2Base):
    wheel_distance = 0.3265  # 覆盖基类的 0.189
```

**修复 2**：噪声归零

```python
# config/slam_config.py
odom_noise_alpha1 = 0.0
odom_noise_alpha2 = 0.0
odom_noise_alpha3 = 0.0
odom_noise_alpha4 = 0.0
```

```python
# core/odometry.py
def _add_motion_noise(self, d_left, d_right):
    if (self.noise_alpha1 == 0 and self.noise_alpha2 == 0 and
            self.noise_alpha3 == 0 and self.noise_alpha4 == 0):
        return d_left, d_right  # 仿真模式: 直接返回
```

**修复 3**：Scan-to-map ICP + 关键帧机制

```python
# core/slam_estimator.py — process_scan() 核心改动
# 优先: scan-to-map (当前扫描 vs 地图占据点)
map_points = self.grid.get_occupied_points_around(odom_pose, radius=5.0)
if len(map_points) >= 20:
    scan_world = transform_points(current_points, odom_pose)
    icp_pose, icp_score = self.icp.match(scan_world, map_points, [0,0,0])

# 回退: frame-to-frame (地图稀疏时)
# 关键帧: 仅在移动 > 0.15m 或旋转 > 0.15rad 后更新参考扫描
```

**修复 4**：ICP 参数优化

```python
# config/slam_config.py
scan_match_min_score = 0.5              # was 0.3 (拒绝低质量匹配)
scan_match_correspondence_dist = 0.3    # was 1.0 (防止错误对应点)
```

### 涉及文件

- `config/slam_config.py`（轮距、噪声、ICP 参数）
- `robot/mmk2_slam_robot.py`（轮距覆盖）
- `core/odometry.py`（噪声跳过逻辑、typo 修复）
- `core/scan_matching.py`（对应距离从配置读取）
- `core/occupancy_grid.py`（新增 `get_occupied_points_around`）
- `core/slam_estimator.py`（scan-to-map ICP + 关键帧）

---

## 9. 探索目标点频繁切换

### 错误现象

机器人在探索过程中频繁更换目标点，即使尚未到达当前目标。日志显示：

```
Frame 840: target (0.53, 2.68) → (0.58, 2.68) → (0.97, 2.68) → (1.47, 2.68)
Frame 960: target (4.33, 1.58) → (4.38, 1.58)
```

目标点沿墙滑动，机器人航向不断摆动（106°→128°→77°→23°），表现为原地打转或反复掉头。

### 根因分析

三个独立问题叠加：

#### 问题 1：重规划检查只量了 1 帧位移（根因）

`stuck_prev_pos` 每帧都更新（line 445），但重规划检查每 30 帧才执行一次：

```python
# 旧代码: stuck_prev_pos 每帧更新
stuck_prev_pos = robot_pose[:2].copy()  # ← 每帧覆盖

# 30 帧后的检查:
recent_move = robot_pose - stuck_prev_pos  # 只量了最后 1 帧的位移 (~0.3mm)
# 0.3mm < 10mm 阈值 → 永远判定为“卡住”
```

机器人实际每秒移动 ~0.3m，但因为只量 1 帧，`recent_move` 始终 < 10mm，
导致每秒都误触发重规划，无条件切换到新前沿目标。

#### 问题 2：LiDAR 减速时误触发重规划（1帧延迟）

重规划条件检查使用上一帧的 `lidar_safe['linear_scale']`：

```python
# 旧代码
lidar_was_slowing = lidar_safe['linear_scale'] < 0.5  # 上一帧数据
```

当机器人刚进入墙边减速区时：
- 上一帧：`lidar_safe = 1.0`（未减速）→ `lidar_was_slowing = False`
- 当前帧：LiDAR 开始减速，机器人速度极低（~0.5mm/帧）
- `stuck_prev_pos` 每帧更新，但 `explore_interval` 检查时 `recent_move < 0.01m`
- 结果：误判为卡住，触发重规划

#### 问题 3：无目标持久化

每次重规划都无条件切换到新目标，即使新旧目标几乎一样：

```python
# 旧代码: 只要重规划就切换
current_target = target  # 无条件替换
current_path = path
```

前沿检测随地图更新不断变化，同一前沿簇的质心可能偏移几个格子，导致目标点沿墙滑动。

#### 问题 4：卡住检测被 LiDAR 减速干扰

LiDAR 减速时速度极低（0.5mm/帧 < 5mm 阈值），被误判为卡住，触发恢复模式（原地旋转）。

### 修复方法

**修复 1**：重规划检查用独立位置追踪器（只在检查时更新，不是每帧）

```python
# 新代码: replan_check_pos 只在 explore_interval 检查时更新
replan_check_pos = None  # 初始化

# 探索决策中:
if replan_check_pos is not None:
    recent_move = np.linalg.norm(robot_pose[:2] - replan_check_pos)
    # 现在量的是 30 帧 (1秒) 的位移, 而非 1 帧
    # 正常移动: ~0.3m >> 0.01m 阈值 → 不会误判

# 检查后更新:
replan_check_pos = robot_pose[:2].copy()
```

**修复 2**：重规划前检查当前帧 LiDAR 状态（非上一帧）

```python
# 新代码: 实时检查当前 LiDAR 状态
ranges_chk, angles_chk = robot.get_lidar_scan()
lidar_now = lidar_safety_check(ranges_chk, angles_chk, config, motion_dir=motion_dir)
lidar_is_slowing = lidar_now['linear_scale'] < 0.5

if recent_move < 0.01 and not lidar_is_slowing:
    should_replan = True  # 真正卡住且未被 LiDAR 减速
elif lidar_is_slowing:
    pass  # LiDAR 正在减速，保持当前目标
```

**修复 3**：目标持久化（新目标与旧目标距离 < 0.5m 时保持当前目标）

```python
# 阈值 0.5m: 过滤地图更新导致的前沿质心偏移 (<0.5m),
# 但不阻止切换到不同前沿簇 (通常 >2m)
if (current_target is not None and
        np.linalg.norm(np.array(target) - np.array(current_target)) < 0.5):
    replan_target_kept += 1  # 保持当前目标，不切换
else:
    current_target = target  # 新目标足够远，切换
    current_path = path
    controller.set_path(path)
```

**修复 4**：卡住检测排除 LiDAR 减速帧

```python
lidar_slowing = lidar_safe['linear_scale'] < 0.5
if moved < 0.005 and not lidar_slowing and has_motion_intent:
    stuck_counter += 1  # 只在非减速状态下计数
else:
    stuck_counter = 0   # LiDAR 减速时重置
```

**修复 5**：添加调试日志

每次重规划输出原因、目标变化距离、LiDAR 状态：

```
[Explore] Target→(4.38, 1.58) path:126pts mem:8 reason:no_target Δ=3.52m
[Explore] Skip replan: LiDAR slowing (0.35), keeping target (4.38, 1.58) [blocked×3]
[Explore] Replan stats: no_target=5 stuck=2 blocked_lidar=12 kept=8
```

### 涉及文件

- `run_active_slam.py`（重规划逻辑、调试计数器、目标持久化）
- `config/slam_config.py`（`path_inflation_radius=0.80`, `path_min_clearance=0.40`）

---

## 问题排查清单

遇到新问题时，按以下顺序检查：

- [ ] MuJoCo 版本兼容性（`mj_multiRay` 参数格式）
- [ ] GLFW 回调注册方式（模块级函数 vs 对象方法）
- [ ] XML `<compiler angle="radian">` 对角度属性的影响
- [ ] geom `group` 属性与 `MjvOption.geomgroup` 的对应关系
- [ ] 轮距/轮半径参数与实际模型的一致性
- [ ] 里程计噪声是否在仿真中不必要地引入
- [ ] ICP 参数是否合理（对应距离、得分阈值）
- [ ] 占据栅格 log-odds 更新符号的正确性
- [ ] 重规划是否被 LiDAR 减速误触发（检查 `lidar_safe` 延迟）
- [ ] 目标点切换是否有距离阈值保护（目标持久化）
