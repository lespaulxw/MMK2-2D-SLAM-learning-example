"""
轮式里程计
==========
差速驱动轮式里程计，根据左右轮位置增量计算机器人位姿。
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


class DifferentialDriveOdometry:
    """差速驱动轮式里程计

    根据左右轮位置（角度）增量，利用差速驱动运动学模型
    计算机器人在世界坐标系中的位姿。

    属性:
        pose: [x, y, theta] 当前估计位姿
        wheel_radius: 轮半径
        wheel_distance: 轮间距
    """

    def __init__(self, config: SLAMConfig):
        self.wheel_radius = config.odom_wheel_radius
        self.wheel_distance = config.odom_wheel_distance
        self.noise_alpha1 = config.odom_noise_alpha1
        self.noise_alpha2 = config.odom_noise_alpha2
        self.noise_alpha3 = config.odom_noise_alpha3
        self.noise_alpha4 = config.odom_noise_alpha4

        # 位姿初始化
        self.pose = np.array([0.0, 0.0, 0.0])

        # 上一时刻左右轮位置
        self.last_wheel_left = None
        self.last_wheel_right = None

        # 累积里程计轨迹
        self.trajectory = []

    def update(self, wheel_left_pos, wheel_right_pos):
        """根据左右轮位置增量更新位姿

        Args:
            wheel_left_pos: 左轮当前位置 (rad)
            wheel_right_pos: 右轮当前位置 (rad)

        Returns:
            pose: [x, y, theta] 更新后的位姿
        """
        if self.last_wheel_left is None:
            self.last_wheel_left = wheel_left_pos
            self.last_wheel_right = wheel_right_pos
            return self.pose.copy()

        # 计算轮子转过的角度
        d_left = self.wheel_radius * (wheel_left_pos - self.last_wheel_left)
        d_right = self.wheel_radius * (wheel_right_pos - self.last_wheel_right)

        # 添加运动噪声 (模拟真实里程计误差)
        d_left, d_right = self._add_motion_noise(d_left, d_right)

        # 差速驱动运动学
        d_center = (d_left + d_right) / 2.0
        d_theta = (d_right - d_left) / self.wheel_distance

        # 更新位姿 (圆弧运动模型)
        theta = self.pose[2]
        if abs(d_theta) < 1e-6:
            # 近似直线运动
            self.pose[0] += d_center * np.cos(theta)
            self.pose[1] += d_center * np.sin(theta)
        else:
            # 圆弧运动
            r = d_center / d_theta
            self.pose[0] += r * (np.sin(theta + d_theta) - np.sin(theta))
            self.pose[1] += r * (np.cos(theta) - np.cos(theta + d_theta))

        self.pose[2] = normalize_angle(theta + d_theta)

        # 记录轨迹
        self.trajectory.append(self.pose.copy())

        # 更新上一时刻轮位置
        self.last_wheel_left = wheel_left_pos
        self.last_wheel_right = wheel_right_pos

        return self.pose.copy()

    def update_from_velocity(self, linear_vel, angular_vel, dt):
        """从线速度/角速度更新位姿

        Args:
            linear_vel: 线速度 (m/s)
            angular_vel: 角速度 (rad/s)
            dt: 时间间隔 (s)

        Returns:
            pose: [x, y, theta]
        """
        d_center = linear_vel * dt
        d_theta = angular_vel * dt

        theta = self.pose[2]
        if abs(d_theta) < 1e-6:
            self.pose[0] += d_center * np.cos(theta)
            self.pose[1] += d_center * np.sin(theta)
        else:
            r = d_center / d_theta
            self.pose[0] += r * (np.sin(theta + d_theta) - np.sin(theta))
            self.pose[1] += -r * (np.cos(theta + d_theta) - np.cos(theta))

        self.pose[2] = normalize_angle(theta + d_theta)
        self.trajectory.append(self.pose.copy())

        return self.pose.copy()

    def _add_motion_noise(self, d_left, d_right):
        """添加运动噪声

        使用 Odometry Motion Model (Probabilistic Robotics, Thrun et al.)
        仿真模式下噪声参数全为 0 时直接返回原始值。
        """
        # 仿真模式: 噪声参数全为 0 时直接返回
        if (self.noise_alpha1 == 0 and self.noise_alpha2 == 0 and
                self.noise_alpha3 == 0 and self.noise_alpha4 == 0):
            return d_left, d_right

        d_center = (d_left + d_right) / 2.0
        d_theta = (d_right - d_left) / self.wheel_distance

        # 噪声参数
        alpha1 = self.noise_alpha1
        alpha2 = self.noise_alpha2
        alpha3 = self.noise_alpha3
        alpha4 = self.noise_alpha4

        # 添加高斯噪声
        noise_d_theta = alpha1 * abs(d_theta) + alpha2 * abs(d_center)
        noise_d_center = alpha3 * abs(d_center) + alpha4 * abs(d_theta)

        d_theta_noisy = d_theta + np.random.normal(0, noise_d_theta)
        d_center_noisy = d_center + np.random.normal(0, noise_d_center)

        # 从噪声后的运动学参数反推轮子位移
        d_left_noisy = d_center_noisy - d_theta_noisy * self.wheel_distance / 2
        d_right_noisy = d_center_noisy + d_theta_noisy * self.wheel_distance / 2

        return d_left_noisy, d_right_noisy

    def get_pose(self):
        """返回当前位姿 [x, y, theta]"""
        return self.pose.copy()

    def set_pose(self, pose):
        """设置位姿 (用于扫描匹配校正后重置)"""
        self.pose = np.array(pose, dtype=np.float64)

    def get_trajectory(self):
        """返回轨迹历史"""
        return np.array(self.trajectory)

    def reset(self):
        """重置里程计"""
        self.pose = np.array([0.0, 0.0, 0.0])
        self.last_wheel_left = None
        self.last_wheel_right = None
        self.trajectory = []
