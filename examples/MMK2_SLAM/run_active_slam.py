#!/usr/bin/env python3
"""
主动 SLAM — 前沿探索自主建图
==============================
机器人自主检测地图上的未知-空闲边界 (前沿),
规划路径前往未探索区域, 实现全自动 SLAM 建图。

特点:
- 前沿检测 + BFS 聚类 + 最近前沿选择
- A* 路径规划 (避障)
- 实时 LiDAR 扫描 + 占据栅格地图构建
- ICP 扫描匹配校正里程计误差
- Matplotlib 调试 GUI + ROS2 RViz 可视化

运行方式:
  # 纯 Python 模式
  python run_active_slam.py --no-ros

  # ROS2 + RViz 模式
  python run_active_slam.py
  # 另一终端: rviz2 -d rviz/slam_2d.rviz
"""
import os
import sys
import argparse
import numpy as np
import glfw

_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

_project_root = os.path.dirname(_current_dir)
_lidar_path = os.path.join(_project_root, "..", "submodules", "MuJoCo-LiDAR")
_lidar_path = os.path.normpath(_lidar_path)
if _lidar_path not in sys.path:
    sys.path.insert(0, _lidar_path)

from config.slam_config import SLAMConfig
from robot.mmk2_slam_robot import MMK2SlamRobot
from robot.motion_controller import MotionController
from core.slam_estimator import SLAMEstimator
from ros.ros2_bridge import SLAMROS2Bridge
from gui.slam_viewer_gui import SLAMViewerGUI


def print_banner():
    print("\n" + "=" * 60)
    print("  MMK2 主动 SLAM — 前沿探索自主建图")
    print("=" * 60)
    print("  机器人将自主检测前沿并探索未知区域")
    print("  ESC: 自由视角  R: 重置  N: 切换导航模式")
    print("  导航模式: 在 GUI 地图上点击设置目标点")
    print("=" * 60)


def lidar_safety_check(ranges, angles, config, motion_dir=1.0):
    """LiDAR 方向感知避障: 分层检测

    分层策略:
    - 窄扇区 (±30°):  减速/停车 — 只有正前方障碍物才影响速度
    - 中扇区 (±60°):  碰撞停车 — 侧前方太近才停车
    - 360°:           极紧急碰撞 (0.3m) — 任何方向都要停

    motion_dir > 0 (前进): 检测前方扇区
    motion_dir < 0 (后退): 检测后方扇区
    motion_dir = 0 (原地): 360° 检测

    Args:
        ranges: (N,) 距离数组
        angles: (N,) 角度数组 (机器人坐标系)
        config: SLAMConfig
        motion_dir: 运动方向, 1.0=前进, -1.0=后退, 0=原地

    Returns:
        dict: {
            'linear_scale': 0.0~1.0  线速度缩放
            'angular_vel':  float    反应式转向角速度 (rad/s)
            'min_dist': float        全局最近障碍物距离
            'min_angle': float       全局最近障碍物角度 (rad)
        }
    """
    result = {
        'linear_scale': 1.0,
        'angular_vel': 0.0,
        'min_dist': config.lidar_cutoff_dist,
        'min_angle': 0.0,
    }

    valid = np.isfinite(ranges) & (ranges > 0.1) & (ranges < config.lidar_cutoff_dist)
    if np.sum(valid) == 0:
        return result

    r = ranges[valid]
    a = angles[valid]

    # === 1. 全局最近障碍物 (仅用于状态显示) ===
    min_idx = np.argmin(r)
    result['min_dist'] = r[min_idx]
    result['min_angle'] = a[min_idx]

    # === 2. 按运动方向确定基准角度 ===
    if motion_dir > 0.1:  # 前进: 基准 0° (正前方)
        center_angle = 0.0
    elif motion_dir < -0.1:  # 后退: 基准 180° (正后方)
        center_angle = np.pi
    else:  # 原地旋转: 不减速
        center_angle = None

    if center_angle is None:
        # 原地旋转: 不减速, 只做极紧急停车
        if result['min_dist'] < config.lidar_collision_dist * 0.5:
            result['linear_scale'] = 0.0
            obs_angle = result['min_angle']
            result['angular_vel'] = -np.sign(obs_angle) * 0.5
        return result

    # 计算每个障碍物相对于运动方向的角度差 (归一化到 [-π, π])
    rel_angle = a - center_angle
    rel_angle = (rel_angle + np.pi) % (2 * np.pi) - np.pi

    # === 3. 窄扇区减速 (±30°): 正前方障碍物 ===
    narrow_arc = np.deg2rad(30)
    narrow_mask = np.abs(rel_angle) < narrow_arc
    if np.sum(narrow_mask) > 0:
        narrow_min_dist = np.min(r[narrow_mask])
        if narrow_min_dist < config.lidar_slowdown_dist:
            if narrow_min_dist < config.lidar_collision_dist:
                result['linear_scale'] = 0.0
            else:
                t = (narrow_min_dist - config.lidar_collision_dist) / \
                    (config.lidar_slowdown_dist - config.lidar_collision_dist)
                result['linear_scale'] = max(0.05, t)

    # === 4. 中扇区碰撞停车 (±60°): 侧前方太近 ===
    if result['linear_scale'] > 0.0:  # 还没被窄扇区停车
        medium_arc = np.deg2rad(60)
        medium_mask = np.abs(rel_angle) < medium_arc
        if np.sum(medium_mask) > 0:
            medium_min_dist = np.min(r[medium_mask])
            if medium_min_dist < config.lidar_collision_dist * 0.7:
                result['linear_scale'] = 0.0

    # === 5. 360° 极紧急碰撞 (0.3m) ===
    hard_collision = config.lidar_collision_dist * 0.5
    if result['min_dist'] < hard_collision:
        result['linear_scale'] = 0.0
        obs_angle = result['min_angle']
        result['angular_vel'] = -np.sign(obs_angle) * 0.5
        if abs(obs_angle) < 0.1:
            result['angular_vel'] = 0.5

    return result


def main():
    parser = argparse.ArgumentParser(description="MMK2 主动 SLAM (前沿探索)")
    parser.add_argument('--no-ros', action='store_true', help='不启动 ROS2')
    parser.add_argument('--no-gui', action='store_true', help='不启动调试 GUI')
    parser.add_argument('--lidar-backend', type=str, default='cpu',
                        choices=['cpu', 'taichi'], help='LiDAR 后端')
    parser.add_argument('--max-coverage', type=float, default=0.8,
                        help='最大地图覆盖率, 达到后停止 (0-1)')
    args = parser.parse_args()

    print_banner()

    # === 配置 ===
    config = SLAMConfig()
    config.mjcf_file_path = os.path.join(_current_dir, "scenes", "slam_room_mmk2.xml")
    config.lidar_backend = args.lidar_backend
    config.use_ros2 = not args.no_ros
    config.use_gui = not args.no_gui

    # === 初始化机器人 ===
    print("\n[1/4] 初始化 MMK2 机器人...")
    robot = MMK2SlamRobot(config)
    robot.reset()
    # 设置自由相机初始视角 (对准机器人)
    robot.free_camera.lookat = np.array([-3.0, -2.0, 0.5])
    robot.free_camera.distance = 8.0
    robot.free_camera.elevation = -35
    robot.free_camera.azimuth = 180
    print("[OK] 机器人初始化成功")

    # === 初始化 SLAM 估计器 ===
    print("\n[2/4] 初始化 SLAM 估计器...")
    slam = SLAMEstimator(config)
    init_pose = robot.get_robot_pose_2d()
    slam.set_pose(init_pose)
    print(f"[OK] SLAM 估计器初始化成功")
    print(f"     机器人初始位姿: x={init_pose[0]:.3f}, y={init_pose[1]:.3f}, "
          f"theta={np.degrees(init_pose[2]):.1f}°")
    print(f"     SLAM 初始位姿:  x={slam.pose[0]:.3f}, y={slam.pose[1]:.3f}")
    print(f"     里程计初始位姿: x={slam.odom.pose[0]:.3f}, y={slam.odom.pose[1]:.3f}")
    print(f"     地图原点: {config.map_origin}, 格数: {config.map_width}x{config.map_height}")

    # === 运动控制器 ===
    controller = MotionController(config)

    # === ROS2 + GUI ===
    print(f"\n[3/4] ROS2 桥接...")
    ros_bridge = SLAMROS2Bridge(config)

    gui = None
    if config.use_gui:
        print("\n[4/4] 初始化调试 GUI...")
        gui = SLAMViewerGUI(config)

    # === 主动探索状态 ===
    current_target = None
    current_path = None
    exploration_timer = 0.0
    exploration_done = False
    lidar_safe = {'linear_scale': 1.0, 'angular_vel': 0.0,
                  'min_dist': config.lidar_cutoff_dist, 'min_angle': 0.0}
    motion_dir = 1.0  # 初始运动方向 (前进)
    # 卡住检测: 记录上次位置, 长时间未移动则触发恢复
    stuck_prev_pos = None
    stuck_counter = 0  # 连续卡住帧数
    stuck_threshold = int(4.0 * 30)  # 4秒未移动 → 触发恢复
    stuck_recovery_mode = False  # 是否在恢复模式
    # 导航模式: 按 N 切换, 在 GUI 地图上点击设置目标点
    nav_mode = False  # False=探索模式, True=导航模式
    map_frozen = False  # 导航模式下冻结地图更新
    # 重规划调试计数器
    replan_no_target = 0
    replan_stuck = 0
    replan_blocked_by_lidar = 0
    replan_target_kept = 0  # 新目标与旧目标相近, 保持当前目标
    # 重规划专用位置追踪 (只在 explore_interval 检查时更新, 不是每帧)
    replan_check_pos = None

    # === 主循环 ===
    print("\n" + "-" * 60)
    print("主动 SLAM 仿真循环已启动")
    print("-" * 60)

    frame_count = 0
    slam_interval = max(1, int(30 / config.lidar_publish_rate))
    explore_interval = int(config.exploration_step_time * 30)  # 帧数
    wheel_radius = robot.wheel_radius
    wheel_distance = robot.wheel_distance

    try:
        while robot.running:
            # === 0. 键盘检测: N 切换导航模式 ===
            if robot.running and glfw.get_key(robot.window, glfw.KEY_N) == glfw.PRESS:
                nav_mode = not nav_mode
                map_frozen = nav_mode
                if nav_mode:
                    print("\n[Mode] >>> NAVIGATION MODE <<< 在 GUI 地图上点击设置目标点")
                    print("       按 N 返回探索模式")
                    # 停止探索, 清除当前目标
                    current_target = None
                    current_path = None
                    if gui is not None:
                        gui.set_navigation_mode(True)
                else:
                    print("\n[Mode] >>> EXPLORATION MODE <<< 恢复自主探索")
                    if gui is not None:
                        gui.set_navigation_mode(False)
                # 等待按键释放避免重复触发
                while glfw.get_key(robot.window, glfw.KEY_N) != glfw.RELEASE:
                    pass

            # === 导航模式: 处理 GUI 点击目标 ===
            if nav_mode and gui is not None and gui.has_new_goal():
                goal = gui.get_nav_goal()
                gui.clear_new_goal()
                print(f"[Nav] Goal: ({goal[0]:.2f}, {goal[1]:.2f})")
                # 用 A* 规划路径
                nav_path = slam.explorer.plan_path(
                    [slam.pose[0], slam.pose[1]], goal, slam.grid
                )
                if nav_path is not None and len(nav_path) > 0:
                    current_target = goal
                    current_path = nav_path
                    controller.set_path(nav_path)
                    print(f"[Nav] Path planned: {len(nav_path)}pts")
                else:
                    # 路径规划失败, 直接朝目标移动
                    current_target = goal
                    current_path = None
                    print(f"[Nav] No path found, moving directly to goal")

            # === 1. 获取当前位姿 + 探索决策 (仅在探索模式下) ===
            robot_pose = slam.get_pose()
            if not nav_mode:
                should_replan = False
                replan_reason = ""

                if current_target is None:
                    # 原因 1: 无目标 (初始 / 到达 / 被清除)
                    should_replan = True
                    replan_no_target += 1
                    replan_reason = "no_target"
                elif frame_count % explore_interval == 0:
                    # 原因 2: 定期检查 — 机器人是否真正卡住?
                    # 使用 replan_check_pos (只在检查时更新), 不是 stuck_prev_pos (每帧更新)
                    if replan_check_pos is not None:
                        dx = robot_pose[0] - replan_check_pos[0]
                        dy = robot_pose[1] - replan_check_pos[1]
                        recent_move = np.sqrt(dx*dx + dy*dy)
                        # 检查当前帧 LiDAR 状态 (不用上一帧, 避免延迟误判)
                        ranges_chk, angles_chk = robot.get_lidar_scan()
                        lidar_now = lidar_safety_check(
                            ranges_chk, angles_chk, config,
                            motion_dir=motion_dir
                        )
                        lidar_is_slowing = lidar_now['linear_scale'] < 0.5

                        if recent_move < 0.01 and not lidar_is_slowing:
                            should_replan = True
                            replan_stuck += 1
                            replan_reason = (f"stuck(move={recent_move:.4f}m"
                                             f",lidar={lidar_now['linear_scale']:.2f})")
                        elif lidar_is_slowing:
                            replan_blocked_by_lidar += 1
                            if frame_count % 300 == 0:
                                print(f"[Explore] Skip replan: LiDAR slowing"
                                      f" ({lidar_now['linear_scale']:.2f}),"
                                      f" keeping target ({current_target[0]:.2f},"
                                      f" {current_target[1]:.2f})"
                                      f" [blocked×{replan_blocked_by_lidar}]")
                    # 更新重规划检查位置 (只在 explore_interval 时, 不是每帧!)
                    replan_check_pos = robot_pose[:2].copy()

                if should_replan:
                    target, path = slam.get_exploration_target()
                    if target is not None:
                        # 目标持久化: 新目标与当前目标太近时保持当前目标
                        # 阈值 0.5m: 过滤地图更新导致的前沿质心偏移 (<0.5m),
                        # 但不阻止切换到不同前沿簇 (通常 >2m)
                        if (current_target is not None and
                                np.linalg.norm(
                                    np.array(target) - np.array(current_target)
                                ) < 0.5):
                            replan_target_kept += 1
                            if replan_reason.startswith("stuck"):
                                # 真正卡住但目标没变 → 可能需要其他恢复策略
                                if replan_target_kept % 3 == 0:
                                    print(f"[Explore] Stuck but target unchanged"
                                          f" ({target[0]:.2f}, {target[1]:.2f}),"
                                          f" keeping path")
                        else:
                            old_t = current_target
                            current_target = target
                            current_path = path
                            if path is not None and len(path) > 0:
                                controller.set_path(path)
                            n_mem = len(slam.explorer.frontier_memory)
                            dist = (np.linalg.norm(
                                np.array(target) - np.array(old_t)
                            ) if old_t is not None else 0)
                            print(f"[Explore] Target→({target[0]:.2f}, {target[1]:.2f})"
                                  f" path:{len(path) if path else 0}pts"
                                  f" mem:{n_mem} reason:{replan_reason}"
                                  f" Δ={dist:.2f}m")
                    else:
                        if slam.get_coverage() > args.max_coverage * 0.5:
                            print("[Explore] No more frontiers, exploration complete!")
                            exploration_done = True
                        else:
                            print("[Explore] No frontiers found, rotating...")
                            current_target = None
                            current_path = None

                # 定期输出重规划统计
                if frame_count > 0 and frame_count % 600 == 0:
                    print(f"[Explore] Replan stats: no_target={replan_no_target}"
                          f" stuck={replan_stuck} blocked_lidar={replan_blocked_by_lidar}"
                          f" kept={replan_target_kept}")

            # 2. 运动控制 (带 ICP 漂移安全 + LiDAR 方向感知避障)
            # robot_pose 已在探索决策前获取
            # 漂移安全: 连续未校正帧数过多时减速/停止
            no_corr = slam.no_correction_count
            speed_scale = 1.0
            if no_corr > 50:       # >5秒无校正 → 原地旋转建图
                speed_scale = 0.0
            elif no_corr > 30:     # >3秒无校正 → 极慢移动
                speed_scale = 0.3
            elif no_corr > 15:     # >1.5秒无校正 → 减速
                speed_scale = 0.6

            # 先计算计划速度 (不应用安全缩放), 确定运动方向
            if exploration_done or stuck_recovery_mode:
                if stuck_recovery_mode:
                    # 恢复模式: 原地旋转
                    linear_vel, angular_vel = 0.0, 0.5
                    motion_dir = 0.0
                else:
                    linear_vel, angular_vel = 0.0, 0.0
                    motion_dir = 0.0
            elif current_path is not None and len(current_path) > 0:
                linear_vel, angular_vel, completed = controller.follow_path(
                    robot_pose, current_path
                )
                motion_dir = 1.0 if linear_vel >= 0 else -1.0
                if completed:
                    current_target = None
            elif current_target is not None:
                linear_vel, angular_vel, reached = controller.move_to_target(
                    robot_pose, current_target
                )
                motion_dir = 1.0 if linear_vel >= 0 else -1.0
                if reached:
                    current_target = None
            else:
                linear_vel = 0.0
                angular_vel = 0.3
                motion_dir = 0.0

            # LiDAR 方向感知避障: 只在运动方向检测障碍物
            ranges_check, angles_check = robot.get_lidar_scan()
            lidar_safe = lidar_safety_check(ranges_check, angles_check, config,
                                            motion_dir=motion_dir)
            speed_scale = min(speed_scale, lidar_safe['linear_scale'])

            # 应用安全缩放
            linear_vel *= speed_scale
            # 角速度不缩放 (保证转向能力)

            # 反应式避障转向: 叠加到路径规划的角速度上
            if lidar_safe['angular_vel'] != 0.0:
                angular_vel += lidar_safe['angular_vel']
                linear_vel = 0.0  # 避障时强制停车

            # === 卡住检测 ===
            # 只在机器人有运动意图时检测卡住 (LiDAR 减速时不算卡住)
            has_motion_intent = abs(linear_vel) > 0.01 or abs(angular_vel) > 0.1
            lidar_slowing = lidar_safe['linear_scale'] < 0.5  # LiDAR 正在减速

            if stuck_prev_pos is not None and not exploration_done:
                dx = robot_pose[0] - stuck_prev_pos[0]
                dy = robot_pose[1] - stuck_prev_pos[1]
                moved = np.sqrt(dx*dx + dy*dy)
                # 只在非减速状态下检测卡住
                if moved < 0.005 and not lidar_slowing and has_motion_intent:
                    stuck_counter += 1
                else:
                    stuck_counter = 0
                    if stuck_recovery_mode and moved > 0.01:
                        # 恢复成功: 机器人开始移动了
                        stuck_recovery_mode = False
                        print("[Stuck] Recovery success, resuming exploration")
            else:
                stuck_counter = 0
            stuck_prev_pos = robot_pose[:2].copy()

            if stuck_counter > stuck_threshold and not stuck_recovery_mode:
                print(f"[Stuck] Robot stuck for {stuck_counter/30:.1f}s, "
                      f"entering recovery mode")
                stuck_recovery_mode = True
                stuck_counter = 0
                # 清除当前目标, 重置访问记忆
                current_target = None
                current_path = None
                slam.explorer.frontier_memory.clear()

            # 3. 构建动作向量并步进仿真
            action = robot.init_joint_ctrl.copy()
            v_left = (linear_vel - angular_vel * wheel_distance / 2) / wheel_radius
            v_right = (linear_vel + angular_vel * wheel_distance / 2) / wheel_radius
            action[0] = np.clip(v_left, -10.0, 10.0)
            action[1] = np.clip(v_right, -10.0, 10.0)
            robot.step(action)

            # 4. SLAM 处理
            if frame_count % slam_interval == 0:
                ranges, angles = robot.get_lidar_scan()
                if len(ranges) > 0:
                    wl, wr = robot.get_wheel_positions()
                    odom_pose = slam.update_odometry(wl, wr)
                    if map_frozen:
                        # 导航模式: 只定位不建图 (保持地图不变)
                        slam.process_scan_localization_only(
                            odom_pose, ranges, angles, config.lidar_cutoff_dist
                        )
                    else:
                        slam.process_scan(odom_pose, ranges, angles,
                                         config.lidar_cutoff_dist)

            # 5. ROS2 发布
            if ros_bridge.enabled:
                ros_bridge.publish_all(robot, slam, current_path, current_target)
                ros_bridge.spin_once()

            # 6. GUI 更新
            if gui is not None:
                robot_pose = slam.get_pose()
                ranges, angles = robot.get_lidar_scan()
                gui.update(slam, robot_pose, ranges, angles,
                          target=current_target, planned_path=current_path,
                          max_range=config.lidar_cutoff_dist)

            # 7. 状态打印
            if frame_count % 120 == 0 and frame_count > 0:
                pose = slam.get_pose()
                stats = slam.get_stats()
                drift_warn = ""
                if slam.no_correction_count > 50:
                    drift_warn = " | !!DRIFT!! STOPPED"
                elif slam.no_correction_count > 30:
                    drift_warn = " | !DRIFT! SLOW 0.3x"
                elif slam.no_correction_count > 15:
                    drift_warn = " | DRIFT SLOW 0.6x"
                lidar_warn = ""
                min_d = lidar_safe['min_dist']
                min_a = np.degrees(lidar_safe['min_angle'])
                if min_d < config.lidar_collision_dist * 0.6:
                    lidar_warn = f" | !!COLLISION!! {min_d:.2f}m@{min_a:.0f}deg"
                elif min_d < config.lidar_slowdown_dist:
                    lidar_warn = f" | SLOW {min_d:.2f}m@{min_a:.0f}deg"
                stuck_warn = ""
                if stuck_recovery_mode:
                    stuck_warn = " | !!STUCK!! RECOVERING"
                elif stuck_counter > stuck_threshold * 0.5:
                    stuck_warn = f" | stuck? {stuck_counter/30:.0f}s"
                mode_str = "NAV" if nav_mode else "EXPLORE"
                if map_frozen:
                    mode_str += "(frozen)"
                print(f"Frame {frame_count:5d} [{mode_str}] | "
                      f"Pose({pose[0]:+.2f}, {pose[1]:+.2f}, "
                      f"{np.degrees(pose[2]):+.1f}deg) | "
                      f"Coverage={stats['coverage']*100:.1f}% | "
                      f"ICP={stats['avg_icp_score']:.3f}"
                      f"{drift_warn}{lidar_warn}{stuck_warn}")

            # 检查覆盖率目标
            if slam.get_coverage() > args.max_coverage and frame_count > 100:
                print(f"\n[INFO] 达到目标覆盖率 {args.max_coverage*100:.0f}%, 停止探索")
                exploration_done = True
                if frame_count % 300 == 0:
                    break

            frame_count += 1

    except KeyboardInterrupt:
        print("\n\n[INFO] 用户中断 (Ctrl+C)")

    finally:
        print("\n" + "-" * 60)
        print("正在清理资源...")

        if gui is not None:
            gui.close()
            print("[OK] GUI 已关闭")

        ros_bridge.shutdown()
        print("[OK] ROS2 已关闭")

        print("\n" + "=" * 60)
        stats = slam.get_stats()
        print(f"总扫描帧数: {stats['scan_count']}")
        print(f"ICP 校正次数: {stats['corrections_applied']}")
        print(f"平均 ICP 得分: {stats['avg_icp_score']:.3f}")
        print(f"地图覆盖率: {stats['coverage']*100:.1f}%")
        print(f"轨迹长度: {stats['trajectory_length']} 点")
        print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
