#!/usr/bin/env python3
"""
键盘遥控 SLAM — 交互式学习
============================
使用 WASD 键盘控制 MMK2 机器人在场景中移动,
实时进行 2D SLAM 建图。

特点:
- 键盘控制, 自由探索
- 实时 LiDAR 扫描 + 占据栅格地图构建
- ICP 扫描匹配校正里程计误差
- Matplotlib 调试 GUI 实时显示
- 可选 ROS2 发布 (RViz 可视化)

运行方式:
  # 纯 Python 模式 (无需 ROS2)
  python run_keyboard_slam.py --no-ros

  # ROS2 + RViz 模式
  python run_keyboard_slam.py
  # 另一终端: rviz2 -d rviz/slam_2d.rviz

键盘控制:
  W/S : 前进/后退
  A/D : 左转/右转
  Shift: 加速
  H   : 显示帮助
  ESC : 切换自由视角
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
from ros.ros2_bridge import SLAMROS2Bridge
from gui.slam_viewer_gui import SLAMViewerGUI


def print_banner():
    print("\n" + "=" * 60)
    print("    MMK2 键盘遥控 SLAM — 交互式学习")
    print("=" * 60)
    print("  W/S: 前进/后退  A/D: 左转/右转  Shift: 加速")
    print("  H: 帮助  ESC: 自由视角  R: 重置")
    print("=" * 60)


def setup_keyboard(robot):
    """设置 GLFW 键盘回调"""
    def on_key(window, key, scancode, action, mods):
        if action == glfw.PRESS or action == glfw.REPEAT:
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
        # 调用基类回调 (相机/帮助/重置等)
        robot.on_key(window, key, scancode, action, mods)

    glfw.set_key_callback(robot.window, on_key)


def main():
    parser = argparse.ArgumentParser(description="MMK2 键盘遥控 SLAM")
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
    print("\n[1/4] 初始化 MMK2 机器人...")
    robot = MMK2SlamRobot(config)
    robot.reset()
    # 设置自由相机初始视角 (对准机器人)
    robot.free_camera.lookat = np.array([-3.0, -2.0, 0.5])
    robot.free_camera.distance = 8.0
    robot.free_camera.elevation = -35
    robot.free_camera.azimuth = 180
    setup_keyboard(robot)
    robot.printHelp()
    print("[OK] 机器人初始化成功")

    # === 初始化 SLAM 估计器 ===
    print("\n[2/4] 初始化 SLAM 估计器...")
    slam = SLAMEstimator(config)
    # 用机器人初始位姿初始化
    init_pose = robot.get_robot_pose_2d()
    slam.set_pose(init_pose)
    print("[OK] SLAM 估计器初始化成功")

    # === 初始化 ROS2 桥接 (可选) ===
    print(f"\n[3/4] ROS2 桥接...")
    ros_bridge = SLAMROS2Bridge(config)

    # === 初始化 GUI (可选) ===
    gui = None
    if config.use_gui:
        print("\n[4/4] 初始化调试 GUI...")
        gui = SLAMViewerGUI(config)

    # === 主循环 ===
    print("\n" + "-" * 60)
    print("仿真循环已启动 (键盘控制)")
    print("-" * 60)

    frame_count = 0
    slam_interval = max(1, int(30 / config.lidar_publish_rate))
    wheel_radius = robot.wheel_radius
    wheel_distance = robot.wheel_distance

    try:
        while robot.running:
            # 1. 键盘控制 → 速度命令
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

            # 2. 构建动作向量并步进仿真
            action = robot.init_joint_ctrl.copy()
            v_left = (linear_vel - angular_vel * wheel_distance / 2) / wheel_radius
            v_right = (linear_vel + angular_vel * wheel_distance / 2) / wheel_radius
            action[0] = np.clip(v_left, -10.0, 10.0)
            action[1] = np.clip(v_right, -10.0, 10.0)
            robot.step(action)

            # 3. SLAM 处理 (按频率)
            if frame_count % slam_interval == 0:
                ranges, angles = robot.get_lidar_scan()
                if len(ranges) > 0:
                    wl, wr = robot.get_wheel_positions()
                    odom_pose = slam.update_odometry(wl, wr)
                    slam.process_scan(odom_pose, ranges, angles,
                                     config.lidar_cutoff_dist)

            # 4. ROS2 发布
            if ros_bridge.enabled:
                ros_bridge.publish_all(robot, slam)
                ros_bridge.spin_once()

            # 5. GUI 更新
            if gui is not None:
                robot_pose = slam.get_pose()
                ranges, angles = robot.get_lidar_scan()
                gui.update(slam, robot_pose, ranges, angles,
                          max_range=config.lidar_cutoff_dist)

            # 6. 状态打印
            if frame_count % 120 == 0 and frame_count > 0:
                pose = slam.get_pose()
                stats = slam.get_stats()
                print(f"Frame {frame_count:5d} | "
                      f"Pose({pose[0]:+.2f}, {pose[1]:+.2f}, "
                      f"{np.degrees(pose[2]):+.1f}°) | "
                      f"Coverage={stats['coverage']*100:.1f}% | "
                      f"ICP avg={stats['avg_icp_score']:.3f}")

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
        print("=" * 60 + "\n")

    return 0


if __name__ == "__main__":
    exit(main())
