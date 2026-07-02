"""
运动控制器
==========
提供差速驱动机器人的运动控制接口：
1. 航点跟踪 (move_to_target)
2. 路径跟踪 (follow_path)
3. 速度命令执行
"""
import numpy as np
from config.slam_config import SLAMConfig


def normalize_angle(angle):
    """将角度归一化到 [-pi, pi]"""
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


class MotionController:
    """运动控制器

    提供基于比例控制的航点跟踪和路径跟踪功能。
    """

    def __init__(self, config: SLAMConfig):
        self.config = config
        self.max_linear_vel = config.max_linear_vel
        self.max_angular_vel = config.max_angular_vel
        self.kp_angular = config.kp_angular
        self.kp_linear = config.kp_linear
        self.reach_threshold = config.trajectory_reach_threshold

        # 路径跟踪状态
        self.current_path = None
        self.current_waypoint_idx = 0

    def move_to_target(self, robot_pose, target):
        """控制机器人朝目标点移动

        Args:
            robot_pose: [x, y, theta] 机器人当前位姿
            target: [tx, ty] 目标位置

        Returns:
            linear_vel: 线速度 (m/s)
            angular_vel: 角速度 (rad/s)
            reached: 是否到达目标
        """
        dx = target[0] - robot_pose[0]
        dy = target[1] - robot_pose[1]
        distance = np.sqrt(dx**2 + dy**2)

        # 计算目标角度
        target_angle = np.arctan2(dy, dx)
        angle_error = normalize_angle(target_angle - robot_pose[2])

        # 角速度控制 (比例控制)
        angular_vel = np.clip(
            self.kp_angular * angle_error,
            -self.max_angular_vel,
            self.max_angular_vel
        )

        # 线速度控制 (角度误差大时减速)
        linear_vel = np.clip(
            self.kp_linear * distance * np.cos(angle_error),
            0,
            self.max_linear_vel
        )

        # 角度误差过大时只旋转不前进
        if abs(angle_error) > 0.5:
            linear_vel = 0.0

        # 检查是否到达
        reached = distance < self.reach_threshold

        return linear_vel, angular_vel, reached

    def follow_path(self, robot_pose, path):
        """路径跟踪 (逐航点)

        Args:
            robot_pose: [x, y, theta]
            path: list of [x, y] 世界坐标点

        Returns:
            linear_vel: 线速度
            angular_vel: 角速度
            completed: 路径是否完成
        """
        if path is None or len(path) == 0:
            return 0.0, 0.0, True

        # 检查是否完成所有航点
        if self.current_waypoint_idx >= len(path):
            return 0.0, 0.0, True

        # 获取当前目标航点
        target = path[self.current_waypoint_idx]

        # 朝目标移动
        linear_vel, angular_vel, reached = self.move_to_target(robot_pose, target)

        # 到达当前航点, 推进到下一个
        if reached:
            self.current_waypoint_idx += 1
            if self.current_waypoint_idx >= len(path):
                return 0.0, 0.0, True

        return linear_vel, angular_vel, False

    def set_path(self, path):
        """设置新路径并重置跟踪状态"""
        self.current_path = path
        self.current_waypoint_idx = 0

    def execute_velocity_command(self, robot, linear_vel, angular_vel):
        """执行速度命令

        Args:
            robot: MMK2SlamRobot 实例
            linear_vel: 线速度 (m/s)
            angular_vel: 角速度 (rad/s)
        """
        robot.apply_diff_drive(linear_vel, angular_vel)

    def stop(self, robot):
        """停止机器人"""
        robot.apply_diff_drive(0.0, 0.0)

    def reset(self):
        """重置控制器状态"""
        self.current_path = None
        self.current_waypoint_idx = 0
