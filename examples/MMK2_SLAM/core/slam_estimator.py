"""
SLAM 估计器
===========
整合轮式里程计、ICP 扫描匹配、占据栅格地图和前沿探索，
实现完整的 2D SLAM 估计流程。

工作流程:
1. 轮式里程计提供初始位姿估计
2. ICP 扫描匹配校正位姿 (帧间匹配)
3. 用校正后的位姿更新占据栅格地图
4. 前沿探索模块提供自主探索目标
"""
import numpy as np
from config.slam_config import SLAMConfig
from core.occupancy_grid import OccupancyGrid
from core.scan_matching import ICPMatcher, transform_points, normalize_angle
from core.odometry import DifferentialDriveOdometry
from core.frontier_exploration import FrontierExplorer


def ranges_to_points(ranges, angles, max_range):
    """将激光扫描的 (ranges, angles) 转换为 2D 点云

    Args:
        ranges: (N,) 距离数组
        angles: (N,) 角度数组
        max_range: 最大有效距离

    Returns:
        points: (M, 2) 2D 点云 (机器人坐标系)
    """
    valid = np.isfinite(ranges) & (ranges > 0.1) & (ranges < max_range)
    r = ranges[valid]
    a = angles[valid]
    x = r * np.cos(a)
    y = r * np.sin(a)
    return np.column_stack([x, y])


class SLAMEstimator:
    """2D SLAM 估计器

    整合里程计、扫描匹配、占据栅格地图和前沿探索。

    属性:
        odom: DifferentialDriveOdometry 里程计
        grid: OccupancyGrid 占据栅格地图
        icp: ICPMatcher 扫描匹配器
        explorer: FrontierExplorer 前沿探索器
        pose: [x, y, theta] 当前估计位姿
        trajectory: 位姿轨迹历史
        prev_scan_points: 上一帧扫描点云 (用于 ICP 帧间匹配)
    """

    def __init__(self, config: SLAMConfig):
        self.config = config

        # 子模块
        self.odom = DifferentialDriveOdometry(config)
        self.grid = OccupancyGrid(config)
        self.icp = ICPMatcher(config)
        self.explorer = FrontierExplorer(config)

        # 状态
        self.pose = np.array([0.0, 0.0, 0.0])
        self.trajectory = []
        self.prev_scan_points = None
        self.prev_pose = None

        # 统计信息
        self.scan_count = 0
        self.icp_scores = []
        self.corrections_applied = 0
        self.no_correction_count = 0  # 连续未校正帧数 (漂移安全指标)

    def update_odometry(self, wheel_left, wheel_right):
        """更新轮式里程计

        Args:
            wheel_left: 左轮位置 (rad)
            wheel_right: 右轮位置 (rad)

        Returns:
            odom_pose: [x, y, theta] 里程计位姿
        """
        return self.odom.update(wheel_left, wheel_right)

    def update_odometry_from_velocity(self, linear_vel, angular_vel, dt):
        """从速度更新里程计"""
        return self.odom.update_from_velocity(linear_vel, angular_vel, dt)

    def process_scan(self, odom_pose, ranges, angles, max_range):
        """处理一帧激光扫描 — SLAM 核心

        流程:
        1. 距离数组 → 2D 点云 (机器人系)
        2. 优先 scan-to-map ICP (当前扫描 vs 地图占据点)
           回退到 frame-to-frame ICP (当前帧 vs 上一关键帧)
        3. 得分 > 阈值 → 用 ICP 结果校正位姿; 否则用里程计
        4. 用校正后的位姿更新占据栅格地图
        5. 关键帧机制: 仅在机器人移动足够距离后更新参考扫描

        Args:
            odom_pose: [x, y, theta] 里程计位姿
            ranges: (N,) 激光距离数组
            angles: (N,) 激光角度数组
            max_range: 最大有效距离 (m)

        Returns:
            dict: {
                'pose': [x, y, theta] 校正后位姿,
                'icp_score': float 匹配得分,
                'corrected': bool 是否使用 ICP 校正,
                'coverage': float 地图覆盖率,
                'n_points': int 有效点数
            }
        """
        # 1. 转换为点云
        current_points = ranges_to_points(ranges, angles, max_range)
        n_points = len(current_points)

        # 2. ICP 匹配
        icp_score = 0.0
        corrected = False
        estimated_pose = odom_pose.copy()

        if n_points >= 10:
            # 优先: scan-to-map 匹配 (不累积漂移, 更稳定)
            map_points = self.grid.get_occupied_points_around(
                odom_pose, self.config.scan_match_map_radius
            )

            if len(map_points) >= 10:
                # 将当前扫描变换到世界坐标系, 与地图占据点匹配
                scan_world = transform_points(current_points, odom_pose)
                icp_pose, icp_score = self.icp.match(
                    source_points=scan_world,
                    target_points=map_points,
                    init_pose=np.array([0.0, 0.0, 0.0])
                )

                if icp_score > self.config.scan_match_min_score:
                    # ICP 结果是对当前位姿的微调校正量
                    estimated_pose = odom_pose.copy()
                    estimated_pose[0] += icp_pose[0]
                    estimated_pose[1] += icp_pose[1]
                    estimated_pose[2] = normalize_angle(odom_pose[2] + icp_pose[2])
                    corrected = True
                    self.corrections_applied += 1
                    self.odom.set_pose(estimated_pose)

            elif self.prev_scan_points is not None and len(self.prev_scan_points) >= 10:
                # 回退: frame-to-frame 匹配 (地图还不够丰富时使用)
                odom_delta = odom_pose - self.pose
                icp_pose, icp_score = self.icp.match(
                    source_points=current_points,
                    target_points=self.prev_scan_points,
                    init_pose=odom_delta
                )

                if icp_score > self.config.scan_match_min_score:
                    estimated_pose = self.pose + icp_pose
                    estimated_pose[2] = normalize_angle(estimated_pose[2])
                    corrected = True
                    self.corrections_applied += 1
                    self.odom.set_pose(estimated_pose)

        # 更新位姿
        self.pose = estimated_pose.copy()
        self.trajectory.append(self.pose.copy())
        self.scan_count += 1
        self.icp_scores.append(icp_score)

        # 漂移安全指标: 连续未校正帧数
        if corrected:
            self.no_correction_count = 0
        else:
            self.no_correction_count += 1

        # 4. 更新占据栅格地图
        self.grid.update_from_scan(self.pose, ranges, angles, max_range)

        # 5. 关键帧机制: 仅在机器人移动足够距离后更新参考扫描
        if self.prev_scan_points is None or self.prev_pose is None:
            self.prev_scan_points = current_points
            self.prev_pose = self.pose.copy()
        else:
            dx = self.pose[0] - self.prev_pose[0]
            dy = self.pose[1] - self.prev_pose[1]
            dt = abs(normalize_angle(self.pose[2] - self.prev_pose[2]))
            dist = np.sqrt(dx * dx + dy * dy)
            if dist > self.config.scan_match_keyframe_dist or dt > self.config.scan_match_keyframe_rot:
                self.prev_scan_points = current_points
                self.prev_pose = self.pose.copy()

        return {
            'pose': self.pose.copy(),
            'icp_score': icp_score,
            'corrected': corrected,
            'coverage': self.grid.get_coverage(),
            'n_points': n_points
        }

    def process_scan_localization_only(self, odom_pose, ranges, angles, max_range):
        """纯定位模式: 只用 ICP 校正位姿, 不更新地图

        用于导航阶段: 地图已冻结, 仅用 scan-to-map ICP 在已有地图上定位。

        Args:
            odom_pose: [x, y, theta] 里程计位姿
            ranges: (N,) 激光距离数组
            angles: (N,) 激光角度数组
            max_range: 最大有效距离 (m)

        Returns:
            dict: {'pose', 'icp_score', 'corrected', 'n_points'}
        """
        current_points = ranges_to_points(ranges, angles, max_range)
        n_points = len(current_points)

        icp_score = 0.0
        corrected = False
        estimated_pose = odom_pose.copy()

        if n_points >= 10:
            map_points = self.grid.get_occupied_points_around(
                odom_pose, self.config.scan_match_map_radius
            )
            if len(map_points) >= 20:
                scan_world = transform_points(current_points, odom_pose)
                icp_pose, icp_score = self.icp.match(
                    source_points=scan_world,
                    target_points=map_points,
                    init_pose=np.array([0.0, 0.0, 0.0])
                )
                if icp_score > self.config.scan_match_min_score:
                    estimated_pose = odom_pose.copy()
                    estimated_pose[0] += icp_pose[0]
                    estimated_pose[1] += icp_pose[1]
                    estimated_pose[2] = normalize_angle(odom_pose[2] + icp_pose[2])
                    corrected = True
                    self.corrections_applied += 1
                    self.odom.set_pose(estimated_pose)

        self.pose = estimated_pose.copy()
        self.trajectory.append(self.pose.copy())
        self.scan_count += 1
        self.icp_scores.append(icp_score)

        # 不更新地图!

        return {
            'pose': self.pose.copy(),
            'icp_score': icp_score,
            'corrected': corrected,
            'n_points': n_points
        }

    def get_exploration_target(self):
        """获取主动探索目标 (前沿探索)

        Returns:
            target: [wx, wy] 目标点, None 表示无前沿
            path: list of [wx, wy] 路径, None 表示无路径
        """
        return self.explorer.get_exploration_target(self.pose, self.grid)

    def get_map_prob(self):
        """返回占据概率图 [0, 1]"""
        return self.grid.get_occupancy_prob()

    def get_ros_map_data(self):
        """返回 ROS OccupancyGrid 格式数据"""
        return self.grid.get_ros_map_data()

    def get_trajectory(self):
        """返回 SLAM 估计的位姿轨迹"""
        return np.array(self.trajectory) if self.trajectory else np.array([]).reshape(0, 3)

    def get_odom_trajectory(self):
        """返回里程计轨迹"""
        return self.odom.get_trajectory()

    def get_pose(self):
        """返回当前估计位姿"""
        return self.pose.copy()

    def get_coverage(self):
        """返回地图覆盖率"""
        return self.grid.get_coverage()

    def get_stats(self):
        """返回统计信息"""
        avg_icp = np.mean(self.icp_scores) if self.icp_scores else 0.0
        return {
            'scan_count': self.scan_count,
            'corrections_applied': self.corrections_applied,
            'avg_icp_score': avg_icp,
            'coverage': self.grid.get_coverage(),
            'trajectory_length': len(self.trajectory)
        }

    def set_pose(self, pose):
        """设置当前位姿 (用于初始化或外部校正)"""
        self.pose = np.array(pose, dtype=np.float64)
        self.odom.set_pose(self.pose)

    def reset(self):
        """重置 SLAM 估计器"""
        self.odom.reset()
        self.grid.reset()
        self.pose = np.array([0.0, 0.0, 0.0])
        self.trajectory = []
        self.prev_scan_points = None
        self.prev_pose = None
        self.scan_count = 0
        self.icp_scores = []
        self.corrections_applied = 0
        self.no_correction_count = 0
