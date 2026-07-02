"""
MMK2 SLAM 机器人 - 基于 DISCOVERSE 成熟模块
==========================================
继承 discoverse.robots_env.mmk2_base.MMK2Base，添加：
1. 激光雷达传感器集成 (MuJoCo-LiDAR)
2. 2D 位姿获取接口 (使用 mj_data 实时数据)
3. 差速驱动控制接口
4. 轮式里程计数据获取
"""
import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation

# 添加 MuJoCo-LiDAR 子模块到搜索路径
_project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_lidar_path = os.path.join(_project_root, "submodules", "MuJoCo-LiDAR")
if _lidar_path not in sys.path:
    sys.path.insert(0, _lidar_path)

# 导入 DISCOVERSE 核心模块
from discoverse.robots_env.mmk2_base import MMK2Base
from config.slam_config import SLAMConfig


class MMK2SlamRobot(MMK2Base):
    """MMK2 SLAM 机器人

    在 MMK2Base 基础上添加：
    1. 激光雷达传感器集成
    2. 键盘遥控支持
    3. 机器人位姿获取接口
    4. 差速驱动控制
    """

    # 覆盖基类: slim 模型驱动轮实际距离 (lft: y=+0.16325, rgt: y=-0.16325)
    wheel_distance = 0.3265

    # 机器人初始起点 (与场景 XML keyframe 一致)
    init_position = [-3.0, -2.0, 0.0]

    def __init__(self, config: SLAMConfig):
        self.config = config
        super().__init__(config)

        # 覆盖初始位置 (基类默认 (0,0,0), 这里设为场景 keyframe 位置)
        self.init_joint_pose[0] = self.init_position[0]  # x
        self.init_joint_pose[1] = self.init_position[1]  # y
        self.init_joint_pose[2] = self.init_position[2]  # z

        # 初始化激光雷达
        if self.config.lidar_enabled:
            self._init_lidar()
        else:
            self.lidar_wrapper = None

        # 键盘控制状态
        self.key_states = {
            'w': False, 's': False, 'a': False, 'd': False, 'shift': False
        }

    def _init_lidar(self):
        """初始化 MuJoCo-LiDAR 传感器"""
        try:
            from mujoco_lidar.lidar_wrapper import MjLidarWrapper
            from mujoco_lidar.scan_gen import create_lidar_single_line

            # 生成激光雷达扫描模式（单线，360度）
            self.rays_theta, self.rays_phi = create_lidar_single_line(
                horizontal_resolution=self.config.lidar_horizontal_resolution,
                horizontal_fov=self.config.lidar_horizontal_fov
            )

            print(f"[MMK2SlamRobot] LiDAR rays: {len(self.rays_theta)} points")

            # 创建 LiDAR wrapper
            self.lidar_wrapper = MjLidarWrapper(
                mj_model=self.mj_model,
                site_name=self.config.lidar_site_name,
                backend=self.config.lidar_backend,
                cutoff_dist=self.config.lidar_cutoff_dist,
                args={
                    'bodyexclude': self.mj_model.body("agv_link").id
                }
            )

            # Warm start
            self.lidar_wrapper.trace_rays(self.mj_data, self.rays_theta, self.rays_phi)
            self.lidar_wrapper.get_hit_points()

            print(f"[MMK2SlamRobot] LiDAR initialized (backend={self.config.lidar_backend}, "
                  f"site={self.config.lidar_site_name})")

        except ImportError as e:
            print(f"[WARNING] Failed to initialize LiDAR: {e}")
            print("  Please ensure MuJoCo-LiDAR submodule is initialized:")
            print("  python scripts/setup_submodules.py --module lidar")
            self.lidar_wrapper = None

    def get_lidar_scan(self):
        """获取 2D 激光扫描

        Returns:
            ranges: (N,) 距离数组, inf 表示未命中
            angles: (N,) 射线角度 (rad), 相对于激光雷达前方
        """
        if self.lidar_wrapper is None:
            return np.array([]), np.array([])

        # 执行光线追踪
        self.lidar_wrapper.trace_rays(self.mj_data, self.rays_theta, self.rays_phi)

        # 获取距离数组
        distances = self.lidar_wrapper.get_distances()

        # 将超出范围的设为 inf
        ranges = distances.copy()
        ranges[ranges >= self.config.lidar_cutoff_dist] = np.inf

        # 角度数组就是 rays_theta
        angles = self.rays_theta.copy()

        return ranges, angles

    def get_lidar_points_2d(self):
        """获取 2D 点云 (世界坐标系, 只取 x, y)

        Returns:
            points_2d: (N, 2) numpy array
        """
        if self.lidar_wrapper is None:
            return np.array([]).reshape(0, 2)

        self.lidar_wrapper.trace_rays(self.mj_data, self.rays_theta, self.rays_phi)
        points_3d = self.lidar_wrapper.get_hit_points()

        if len(points_3d) == 0:
            return np.array([]).reshape(0, 2)

        return points_3d[:, :2]

    def get_robot_pose_2d(self):
        """获取机器人在 2D 平面上的位姿

        Returns:
            pose: [x, y, theta] 数组
                x, y: 位置 (米)
                theta: 航向角 (弧度)
        """
        # 使用 mj_data 获取实时位姿 (机器人有 free joint 会移动)
        agv_body = self.mj_data.body("agv_link")
        pos = agv_body.xpos.copy()
        quat_wxyz = agv_body.xquat.copy()

        # 转换为欧拉角 (MuJoCo quat: wxyz -> scipy quat: xyzw)
        rot = Rotation.from_quat(quat_wxyz[[1, 2, 3, 0]])
        euler = rot.as_euler('zyx')  # [yaw, pitch, roll]

        return np.array([pos[0], pos[1], euler[0]])

    def get_laser_pose_2d(self):
        """获取激光雷达传感器在 2D 平面上的位姿

        Returns:
            pose: [x, y, theta] 数组
        """
        laser_site = self.mj_data.site("laser")
        pos = laser_site.xpos.copy()
        mat = laser_site.xmat.reshape(3, 3)
        quat = Rotation.from_matrix(mat).as_quat()
        euler = Rotation.from_quat(quat).as_euler('zyx')

        return np.array([pos[0], pos[1], euler[0]])

    def get_wheel_positions(self):
        """获取左右轮位置 (rad)

        Returns:
            (left_pos, right_pos)
        """
        return float(self.sensor_wheel_qpos[0]), float(self.sensor_wheel_qpos[1])

    def apply_diff_drive(self, linear_vel, angular_vel):
        """差速驱动控制

        Args:
            linear_vel: 线速度 (m/s)
            angular_vel: 角速度 (rad/s)
        """
        wheel_radius = self.wheel_radius
        wheel_distance = self.wheel_distance

        # 差速驱动运动学
        v_left = (linear_vel - angular_vel * wheel_distance / 2) / wheel_radius
        v_right = (linear_vel + angular_vel * wheel_distance / 2) / wheel_radius

        # 设置控制命令 (前两个是左右轮速度)
        self.mj_data.ctrl[0] = np.clip(v_left, -10.0, 10.0)
        self.mj_data.ctrl[1] = np.clip(v_right, -10.0, 10.0)

    def updateControlFromKeyboard(self, key_states):
        """根据键盘输入更新控制命令"""
        speed_factor = 3.0 if key_states.get('shift', False) else 1.0
        linear_speed = 0.5 * speed_factor
        angular_speed = 2.0 * speed_factor

        linear_vel = 0.0
        angular_vel = 0.0

        if key_states.get('w', False):
            linear_vel = linear_speed
        elif key_states.get('s', False):
            linear_vel = -linear_speed

        if key_states.get('a', False):
            angular_vel = angular_speed
        elif key_states.get('d', False):
            angular_vel = -angular_speed

        self.apply_diff_drive(linear_vel, angular_vel)

    def resetState(self):
        """重置机器人状态"""
        super().resetState()

    def printHelp(self):
        """打印帮助信息"""
        print("\n" + "=" * 60)
        print("       MMK2 SLAM 学习例程 - 键盘控制")
        print("=" * 60)
        print("\n=== 键盘控制 ===")
        print("W / S : 前进 / 后退")
        print("A / D : 左转 / 右转")
        print("Shift : 按住加速")
        print("ESC   : 切换到自由视角 (鼠标拖动旋转/平移/缩放)")
        print("[ / ] : 切换相机 (overview / top_down)")
        print("H     : 显示此帮助")
        print("\n=== RViz2 可视化 ===")
        print("1. TF          - 查看坐标系变换")
        print("2. LaserScan   - 话题: /scan")
        print("3. Map         - 话题: /map")
        print("4. Path        - 话题: /planned_path")
        print("5. Fixed Frame: map")
        print("=" * 60 + "\n")
