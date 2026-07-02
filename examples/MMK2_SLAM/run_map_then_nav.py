#!/usr/bin/env python3
"""
先建图后导航 — Map then Navigate
====================================
两阶段 SLAM 系统:
  阶段 1 (MAPPING):  键盘遥控建图, WASD 控制机器人
  阶段 2 (NAVIGATION): 冻结地图, 点击 Matplotlib 地图面板设置目标,
                        机器人自主 A* 路径规划 + 跟踪导航

工作流程:
  1. 启动后进入 MAPPING 模式, WASD 遥控建图
  2. 按 N 切换到 NAVIGATION 模式 (地图冻结 + 保存)
  3. 在 Matplotlib 地图面板上点击设置导航目标
  4. 机器人自动 A* 规划路径并跟踪前往
  5. 按 M 返回 MAPPING 模式继续建图
  6. 地图在切换到 NAV 模式时自动保存

运行方式:
  python run_map_then_nav.py --no-ros
"""
import os
import sys
import argparse
import numpy as np
import glfw

# 路径设置
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
from core.slam_estimator import SLAMEstimator
from core.frontier_exploration import FrontierExplorer
from robot.motion_controller import MotionController
from ros.ros2_bridge import SLAMROS2Bridge
from gui.slam_viewer_gui import SLAMViewerGUI

# 运行模式
MODE_MAPPING = "MAPPING"
MODE_NAVIGATION = "NAVIGATION"


def print_banner():
    print("\n" + "=" * 60)
    print("    MMK2 先建图后导航 — Map then Navigate")
    print("=" * 60)
    print("  === MAPPING 模式 ===")
    print("  W/S: 前进/后退  A/D: 左转/右转  Shift: 加速")
    print("  N: 切换到 NAVIGATION 模式")
    print("  === NAVIGATION 模式 ===")
    print("  点击地图面板: 设置导航目标")
    print("  Space: 停止  M: 返回 MAPPING 模式")
    print("  === \u901a\u7528 ===")
    print("  ESC: \u81ea\u7531\u89c6\u89d2  R: \u91cd\u7f6e")
    print("=" * 60)


def setup_keyboard(robot, key_flags):
    """设置 GLFW 键盘回调

    Args:
        robot: MMK2SlamRobot 实例
        key_flags: dict, 用于传递特殊按键事件给主循环
                   key_flags['mode'] 存储当前模式 (由主循环更新)
    """
    def on_key(window, key, scancode, action, mods):
        current_mode = key_flags.get('mode', MODE_MAPPING)

        if action == glfw.PRESS or action == glfw.REPEAT:
            # WASD 仅在 MAPPING 模式下控制机器人
            if current_mode == MODE_MAPPING:
                if key == glfw.KEY_W:
                    robot.key_states['w'] = True
                elif key == glfw.KEY_S:
                    robot.key_states['s'] = True
                elif key == glfw.KEY_A:
                    robot.key_states['a'] = True
                elif key == glfw.KEY_D:
                    robot.key_states['d'] = True
                elif key == glfw.KEY_LEFT_SHIFT or key == glfw.KEY_RIGHT_SHIFT:
                    robot.key_states['shift'] = True
            # 特殊按键: 始终响应 (传递给主循环)
            if key == glfw.KEY_N:
                key_flags['switch_nav'] = True
            elif key == glfw.KEY_M:
                key_flags['switch_map'] = True
            elif key == glfw.KEY_SPACE:
                key_flags['stop'] = True
        elif action == glfw.RELEASE:
            if key == glfw.KEY_W:
                robot.key_states['w'] = False
            elif key == glfw.KEY_S:
                robot.key_states['s'] = False
            elif key == glfw.KEY_A:
                robot.key_states['a'] = False
            elif key == glfw.KEY_D:
                robot.key_states['d'] = False
            elif key == glfw.KEY_LEFT_SHIFT or key == glfw.KEY_RIGHT_SHIFT:
                robot.key_states['shift'] = False
        # 基类回调 (相机/帮助/重置等)
        robot.on_key(window, key, scancode, action, mods)

    glfw.set_key_callback(robot.window, on_key)


def save_map(slam, path=None):
    """保存地图到文件"""
    if path is None:
        path = os.path.join(_current_dir, "saved_map.npy")
    map_data = {
        'log_odds': slam.grid.log_odds.copy(),
        'resolution': slam.grid.resolution,
        'origin': slam.grid.origin.copy(),
        'width': slam.grid.width,
        'height': slam.grid.height,
    }
    np.save(path, map_data, allow_pickle=True)
    print(f"[MAP] 地图已保存到: {path}")
    return path


def main():
    parser = argparse.ArgumentParser(description="MMK2 先建图后导航")
    parser.add_argument('--no-ros', action='store_true', help='不启动 ROS2')
    parser.add_argument('--no-gui', action='store_true', help='不启动调试 GUI')
    parser.add_argument('--lidar-backend', type=str, default='cpu',
                        choices=['cpu', 'taichi'], help='LiDAR 后端')
    args = parser.parse_args()

    print_banner()

    # === 配置 ===
    config = SLAMConfig()
    config.mjcf_file_path = os.path.join(_current_dir, "scenes", "slam_room_mmk2.xml")
    config.lidar_backend = args.lidar_backend
    config.use_ros2 = not args.no_ros
    config.use_gui = not args.no_gui

    # === 初始化机器人 ===
    print("\n[1/5] 初始化 MMK2 机器人...")
    robot = MMK2SlamRobot(config)
    robot.reset()
    robot.free_camera.lookat = np.array([-3.0, -2.0, 0.5])
    robot.free_camera.distance = 8.0
    robot.free_camera.elevation = -35
    robot.free_camera.azimuth = 180

    key_flags = {}  # 特殊按键标志
    setup_keyboard(robot, key_flags)
    robot.printHelp()
    print("[OK] 机器人初始化成功")

    # === 初始化 SLAM ===
    print("\n[2/5] 初始化 SLAM 估计器...")
    slam = SLAMEstimator(config)
    init_pose = robot.get_robot_pose_2d()
    slam.set_pose(init_pose)
    print("[OK] SLAM 估计器初始化成功")

    # === 初始化 ROS2 ===
    print(f"\n[3/5] ROS2 桥接...")
    ros_bridge = SLAMROS2Bridge(config)

    # === 初始化 GUI ===
    gui = None
    if config.use_gui:
        print("\n[4/5] 初始化调试 GUI...")
        gui = SLAMViewerGUI(config)

    # === 初始化运动控制器 + 探索器 ===
    print("\n[5/5] 初始化运动控制器...")
    motion_ctrl = MotionController(config)
    explorer = FrontierExplorer(config)
    print("[OK] 运动控制器初始化成功")

    # === 状态变量 ===
    current_mode = MODE_MAPPING
    key_flags['mode'] = current_mode  # 同步给键盘回调
    frame_count = 0
    slam_interval = max(1, int(30 / config.lidar_publish_rate))
    wheel_radius = robot.wheel_radius
    wheel_distance = robot.wheel_distance

    # 导航状态
    nav_path = None
    nav_goal = None
    nav_completed = False

    # === 主循环 ===
    print("\n" + "-" * 60)
    print("仿真循环已启动 — 当前模式: MAPPING (键盘建图)")
    print("按 N 切换到 NAVIGATION 模式")
    print("-" * 60)

    try:
        while robot.running:
            # ============================================================
            #  模式切换检测
            # ============================================================
            if key_flags.pop('switch_nav', False) and current_mode == MODE_MAPPING:
                current_mode = MODE_NAVIGATION
                key_flags['mode'] = current_mode  # 同步给键盘回调
                print("\n" + "=" * 60)
                print("  >>> 切换到 NAVIGATION 模式 <<<")
                print("  地图已冻结 — 点击 Matplotlib 地图面板设置导航目标")
                print("  Space: 停止  M: 返回 MAPPING 模式")
                print("=" * 60 + "\n")
                if gui:
                    gui.set_navigation_mode(True)
                motion_ctrl.reset()
                motion_ctrl.stop(robot)
                nav_path = None
                nav_goal = None
                nav_completed = False
                # 自动保存地图
                save_map(slam)

            elif key_flags.pop('switch_map', False) and current_mode == MODE_NAVIGATION:
                current_mode = MODE_MAPPING
                key_flags['mode'] = current_mode  # 同步给键盘回调
                print("\n" + "=" * 60)
                print("  >>> 切换回 MAPPING 模式 <<<")
                print("  地图已解冻 — WASD 继续建图")
                print("=" * 60 + "\n")
                if gui:
                    gui.set_navigation_mode(False)
                motion_ctrl.reset()
                motion_ctrl.stop(robot)
                nav_path = None
                nav_goal = None

            # 停止命令
            if key_flags.pop('stop', False) and current_mode == MODE_NAVIGATION:
                motion_ctrl.stop(robot)
                nav_path = None
                nav_goal = None
                nav_completed = False
                if gui:
                    gui.clear_new_goal()
                print("[NAV] 已停止")

            # ============================================================
            #  MAPPING 模式: 键盘控制 + SLAM 建图
            # ============================================================
            if current_mode == MODE_MAPPING:
                # 键盘 → 速度
                linear_vel, angular_vel = 0.0, 0.0
                speed_factor = 3.0 if robot.key_states.get('shift', False) else 1.0
                if robot.key_states.get('w', False):
                    linear_vel = 0.5 * speed_factor
                elif robot.key_states.get('s', False):
                    linear_vel = -0.5 * speed_factor
                if robot.key_states.get('a', False):
                    angular_vel = 2.0 * speed_factor
                elif robot.key_states.get('d', False):
                    angular_vel = -2.0 * speed_factor

                # 速度 → 轮子 → 仿真步进
                action = robot.init_joint_ctrl.copy()
                v_left = (linear_vel - angular_vel * wheel_distance / 2) / wheel_radius
                v_right = (linear_vel + angular_vel * wheel_distance / 2) / wheel_radius
                action[0] = np.clip(v_left, -10.0, 10.0)
                action[1] = np.clip(v_right, -10.0, 10.0)
                robot.step(action)

                # SLAM 处理 (建图)
                if frame_count % slam_interval == 0:
                    ranges, angles = robot.get_lidar_scan()
                    if len(ranges) > 0:
                        wl, wr = robot.get_wheel_positions()
                        odom_pose = slam.update_odometry(wl, wr)
                        slam.process_scan(odom_pose, ranges, angles,
                                         config.lidar_cutoff_dist)

            # ============================================================
            #  NAVIGATION 模式: 自主导航 + 纯定位
            # ============================================================
            elif current_mode == MODE_NAVIGATION:
                # 检查 GUI 点击目标
                if gui and gui.has_new_goal():
                    nav_goal = gui.get_nav_goal()
                    gui.clear_new_goal()
                    nav_completed = False

                    robot_pose = slam.get_pose()
                    print(f"[NAV] 目标: ({nav_goal[0]:.2f}, {nav_goal[1]:.2f})")

                    # A* 路径规划
                    nav_path = explorer.plan_path(
                        [robot_pose[0], robot_pose[1]],
                        nav_goal,
                        slam.grid
                    )
                    if nav_path and len(nav_path) > 0:
                        motion_ctrl.set_path(nav_path)
                        print(f"[NAV] 路径规划成功: {len(nav_path)} 个航点")
                    else:
                        print("[NAV] 路径规划失败! 请尝试其他目标")
                        nav_path = None

                # 路径跟踪
                if nav_path and len(nav_path) > 0:
                    robot_pose = slam.get_pose()
                    linear_vel, angular_vel, completed = motion_ctrl.follow_path(
                        robot_pose, nav_path
                    )

                    if completed:
                        nav_completed = True
                        nav_path = None
                        motion_ctrl.stop(robot)
                        print("[NAV] 已到达目标!")
                        linear_vel, angular_vel = 0.0, 0.0
                else:
                    # 空闲: 停止
                    linear_vel, angular_vel = 0.0, 0.0
                    if not nav_completed:
                        motion_ctrl.stop(robot)

                # 速度 → 轮子 → 仿真步进
                action = robot.init_joint_ctrl.copy()
                v_left = (linear_vel - angular_vel * wheel_distance / 2) / wheel_radius
                v_right = (linear_vel + angular_vel * wheel_distance / 2) / wheel_radius
                action[0] = np.clip(v_left, -10.0, 10.0)
                action[1] = np.clip(v_right, -10.0, 10.0)
                robot.step(action)

                # 纯定位 (不更新地图)
                if frame_count % slam_interval == 0:
                    ranges, angles = robot.get_lidar_scan()
                    if len(ranges) > 0:
                        wl, wr = robot.get_wheel_positions()
                        odom_pose = slam.update_odometry(wl, wr)
                        slam.process_scan_localization_only(
                            odom_pose, ranges, angles, config.lidar_cutoff_dist
                        )

            # ============================================================
            #  公共处理 (两种模式共享)
            # ============================================================

            # ROS2 发布
            if ros_bridge.enabled:
                ros_bridge.publish_all(robot, slam)
                ros_bridge.spin_once()

            # GUI 更新
            if gui is not None:
                robot_pose = slam.get_pose()
                ranges, angles = robot.get_lidar_scan()
                gui.update(
                    slam, robot_pose, ranges, angles,
                    target=nav_goal,
                    planned_path=nav_path,
                    max_range=config.lidar_cutoff_dist
                )

            # 状态打印
            if frame_count % 120 == 0 and frame_count > 0:
                pose = slam.get_pose()
                stats = slam.get_stats()
                mode_tag = "MAP" if current_mode == MODE_MAPPING else "NAV"
                print(f"[{mode_tag}] Frame {frame_count:5d} | "
                      f"Pose({pose[0]:+.2f}, {pose[1]:+.2f}, "
                      f"{np.degrees(pose[2]):+.1f}deg) | "
                      f"Coverage={stats['coverage']*100:.1f}% | "
                      f"ICP={stats['avg_icp_score']:.3f}")

            frame_count += 1

    except KeyboardInterrupt:
        print("\n\n[INFO] 用户中断 (Ctrl+C)")

    finally:
        print("\n" + "-" * 60)
        print("正在清理资源...")

        # 保存最终地图
        if slam.get_stats()['coverage'] > 0.01:
            save_map(slam)

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
        print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
