"""
MMK2 SLAM 学习例程配置
=====================
继承 MMK2Cfg，添加 2D SLAM 所需的全部参数。
"""
import numpy as np
from discoverse.robots_env.mmk2_base import MMK2Cfg


class SLAMConfig(MMK2Cfg):
    """2D SLAM 学习例程配置"""

    # === 场景 ===
    mjcf_file_path = "slam_room_mmk2.xml"
    timestep       = 0.0025
    decimation     = 4
    sync           = True
    headless       = False
    render_set     = {
        "fps"    : 30,
        "width"  : 1280,
        "height" : 720
    }
    use_gaussian_renderer = False

    # === 机器人初始位置 (与场景 XML keyframe 一致) ===
    init_state     = {
        "base_position"    : [-3.0, -2.0, 0.0],
        "base_orientation" : [1.0, 0.0, 0.0, 0.0],
        "slide_qpos"       : [0.0],
        "head_qpos"        : [0.0, 0.0],
        "lft_arm_qpos"     : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "lft_gripper_qpos" : [0.0],
        "rgt_arm_qpos"     : [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        "rgt_gripper_qpos" : [0.0],
    }

    # === LiDAR ===
    lidar_enabled = True
    lidar_site_name = "laser"                # 与 mmk2_slim.xml 中的 site 名称一致
    lidar_backend = "cpu"                    # 'cpu', 'taichi'
    lidar_cutoff_dist = 10.0                 # 最大探测距离 (m)
    lidar_horizontal_resolution = 360        # 2D 射线数
    lidar_horizontal_fov = 2 * np.pi         # 360 度
    lidar_publish_rate = 10                  # Hz

    # === 占据栅格地图 ===
    map_resolution = 0.05                    # 每格 5cm
    map_width = 300                          # 格数 (15m)
    map_height = 220                         # 格数 (11m)
    map_origin = [-7.5, -5.5]               # 地图原点在世界坐标系中的位置 (m)
    map_log_odd_lo = -100                   # log-odds 下限
    map_log_odd_hi = 100                    # log-odds 上限
    map_log_odd_free = -1.0                 # 空闲更新量
    map_log_odd_occupied = 2.0              # 占据更新量

    # === 里程计 ===
    odom_wheel_radius = 0.0838
    odom_wheel_distance = 0.3265
    odom_noise_alpha1 = 0.0                 # 旋转->旋转噪声 (仿真中 MuJoCo 提供完美数据)
    odom_noise_alpha2 = 0.0                 # 旋转->平移噪声
    odom_noise_alpha3 = 0.0                 # 平移->平移噪声
    odom_noise_alpha4 = 0.0                 # 平移->旋转噪声

    # === 扫描匹配 (ICP) ===
    scan_match_max_iter = 50                # ICP 最大迭代
    scan_match_tolerance = 1e-4             # 收敛阈值
    scan_match_max_translation = 0.5        # 最大允许平移 (m)
    scan_match_max_rotation = 0.3           # 最大允许旋转 (rad)
    scan_match_min_score = 0.5              # 最低匹配得分阈值 (提高以拒绝低质量匹配)
    scan_match_correspondence_dist = 0.3    # 最大对应点距离 (m)
    scan_match_map_radius = 5.0             # scan-to-map 匹配搜索半径 (m)
    scan_match_keyframe_dist = 0.05          # 关键帧最小位移 (m) — 密集关键帧提升稀疏区匹配
    scan_match_keyframe_rot = 0.05          # 关键帧最小旋转 (rad)

    # === 路径规划安全 ===
    path_inflation_radius = 0.80            # 路径膨胀半径 (m) — 路径远离障碍物的影响范围
    path_inflation_weight = 6.0             # 膨胀代价权重 — 越大越远离障碍物
    robot_radius = 0.22                     # 机器人等效半径 (m) — 底盘半宽 0.20m + 余量
    path_min_clearance = 0.40               # 最小通行距离 (m) — 机器人半径 + 安全余量

    # === LiDAR 避障 ===
    lidar_collision_dist = 0.5              # LiDAR 碰撞预警距离 (m)
    lidar_slowdown_dist = 1.0               # LiDAR 减速距离 (m)
    lidar_front_arc = np.deg2rad(60)        # 前方检测扇区半角 (rad)

    # === 前沿探索 ===
    frontier_min_size = 5                   # 最小前沿格数
    frontier_reach_threshold = 0.5          # 到达目标阈值 (m) — 增大防止贴墙
    exploration_step_time = 1.0             # 探索决策间隔 (s) — 更频繁重规划
    frontier_safety_margin = 0.45           # 前沿安全边距 (m) — 距障碍物太近的前沿格被过滤
    frontier_visit_penalty = 2.5            # 已访问前沿惩罚 (m) — 等效距离增加
    frontier_forget_dist = 1.5              # 遗忘距离 (m) — 距离多近算“访问过”
    frontier_memory_max = 8                 # 访问记忆容量 — 记住最近访问的前沿数

    # === 航点跟踪 ===
    trajectory_reach_threshold = 0.2        # 航点到达阈值 (m)

    # === 运动控制 ===
    max_linear_vel = 0.3                    # m/s — 降低速度减少漂移
    max_angular_vel = 0.6                   # rad/s
    kp_angular = 1.5                        # 角度比例增益
    kp_linear = 1.0                         # 距离比例增益

    # === 运行模式 ===
    use_ros2 = True                         # 是否启用 ROS2 发布
    use_gui = True                          # 是否启用调试 GUI
    gui_update_rate = 5                     # GUI 更新频率 Hz
