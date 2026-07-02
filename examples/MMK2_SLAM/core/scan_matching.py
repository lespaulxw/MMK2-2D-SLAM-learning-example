"""
ICP 扫描匹配
=============
使用 ICP (Iterative Closest Point) 算法进行 2D 激光扫描匹配，
通过 KDTree 加速最近邻搜索，SVD 分解求解刚体变换。
"""
import numpy as np
from scipy.spatial import cKDTree
from config.slam_config import SLAMConfig


def transform_points(points, pose):
    """用位姿 [x, y, theta] 变换点集

    Args:
        points: (N, 2) 点集
        pose: [x, y, theta]

    Returns:
        transformed: (N, 2) 变换后的点集
    """
    x, y, theta = pose
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t],
                  [sin_t, cos_t]])
    t = np.array([x, y])
    return points @ R.T + t


def normalize_angle(angle):
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


class ICPMatcher:
    """ICP (Iterative Closest Point) 扫描匹配

    算法流程:
    1. 将 source 点集用初始位姿变换到目标坐标系
    2. 为每个 source 点在 target 中找最近邻 (KDTree)
    3. 计算对应点对的质心
    4. SVD 分解求旋转
    5. 由旋转矩阵求平移
    6. 更新位姿, 检查收敛
    7. 重复直到收敛或达到最大迭代
    """

    def __init__(self, config: SLAMConfig):
        self.max_iter = config.scan_match_max_iter
        self.tolerance = config.scan_match_tolerance
        self.max_translation = config.scan_match_max_translation
        self.max_rotation = config.scan_match_max_rotation
        self.max_correspondence_dist = config.scan_match_correspondence_dist

    def match(self, source_points, target_points, init_pose=None):
        """ICP 配准

        Args:
            source_points: (N, 2) 当前扫描点集 (机器人系)
            target_points: (M, 2) 参考扫描点集 (机器人系)
            init_pose: [x, y, theta] 初始位姿猜测

        Returns:
            pose: [x, y, theta] 匹配后的位姿
            score: 匹配得分 (0-1, 越高越好)
        """
        if len(source_points) < 3 or len(target_points) < 3:
            return np.array([0.0, 0.0, 0.0]), 0.0

        if init_pose is None:
            init_pose = np.array([0.0, 0.0, 0.0])
        else:
            init_pose = np.array(init_pose, dtype=np.float64)

        # 构建 target 的 KDTree
        target_tree = cKDTree(target_points)

        # 当前位姿
        current_pose = init_pose.copy()

        # 变换 source 点到目标坐标系
        transformed = transform_points(source_points, current_pose)

        prev_error = float('inf')

        for iteration in range(self.max_iter):
            # 1. 找最近邻
            distances, indices = target_tree.query(transformed, k=1)

            # 2. 过滤距离过远的对应点
            valid = distances < self.max_correspondence_dist
            if np.sum(valid) < 3:
                break

            src_valid = transformed[valid]
            tgt_valid = target_points[indices[valid]]

            # 3. 计算质心
            centroid_src = np.mean(src_valid, axis=0)
            centroid_tgt = np.mean(tgt_valid, axis=0)

            # 4. 去质心
            src_centered = src_valid - centroid_src
            tgt_centered = tgt_valid - centroid_tgt

            # 5. SVD 分解求旋转
            H = src_centered.T @ tgt_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T

            # 处理反射情况
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            # 6. 求平移
            t = centroid_tgt - R @ centroid_src

            # 7. 更新变换
            transformed = src_valid @ R.T + t  # 只更新有效点

            # 实际上需要变换所有 source 点
            # 累积变换
            delta_theta = np.arctan2(R[1, 0], R[0, 0])
            current_pose[0] += t[0]
            current_pose[1] += t[1]
            current_pose[2] = normalize_angle(current_pose[2] + delta_theta)

            # 重新变换所有 source 点
            transformed = transform_points(source_points, current_pose)

            # 8. 检查收敛
            mean_error = np.mean(distances[valid])
            if abs(prev_error - mean_error) < self.tolerance:
                break
            prev_error = mean_error

        # 计算最终得分
        distances, _ = target_tree.query(transformed, k=1)
        valid = distances < self.max_correspondence_dist
        if np.sum(valid) > 0:
            mean_dist = np.mean(distances[valid])
            score = max(0.0, 1.0 - mean_dist / self.max_correspondence_dist)
        else:
            score = 0.0

        # 限制位姿变化范围
        dx = current_pose[0] - init_pose[0]
        dy = current_pose[1] - init_pose[1]
        dt = normalize_angle(current_pose[2] - init_pose[2])

        if abs(dx) > self.max_translation or abs(dy) > self.max_translation or abs(dt) > self.max_rotation:
            # 变化过大, 可能是错误匹配, 使用初始位姿
            return init_pose.copy(), 0.0

        return current_pose, score
