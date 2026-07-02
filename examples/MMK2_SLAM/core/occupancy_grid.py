"""
占据栅格地图 (Occupancy Grid Map)
=================================
基于 log-odds 表示的占据栅格地图，
使用 Bresenham 射线投射实现逆传感器模型。

参考: Thrun, Burgard, Fox 《Probabilistic Robotics》第 9 章
"""
import numpy as np
from config.slam_config import SLAMConfig


class OccupancyGrid:
    """占据栅格地图 (log-odds 表示)

    属性:
        log_odds: (H, W) log-odds 占据值, 0=未知, 正=占据, 负=空闲
        resolution: 每格尺寸 (m)
        origin: [ox, oy] 地图左下角在世界坐标系中的位置
        width, height: 格数
    """

    def __init__(self, config: SLAMConfig):
        self.resolution = config.map_resolution
        self.width = config.map_width
        self.height = config.map_height
        self.origin = np.array(config.map_origin, dtype=np.float64)

        # log-odds 参数
        self.log_odd_lo = config.map_log_odd_lo
        self.log_odd_hi = config.map_log_odd_hi
        self.log_odd_free = config.map_log_odd_free
        self.log_odd_occupied = config.map_log_odd_occupied

        # 初始化为 0 (未知)
        self.log_odds = np.zeros((self.height, self.width), dtype=np.float32)

    def world_to_map(self, x, y):
        """世界坐标 -> 栅格坐标

        Returns:
            (mx, my) 栅格坐标, 越界返回 (-1, -1)
        """
        mx = int((x - self.origin[0]) / self.resolution)
        my = int((y - self.origin[1]) / self.resolution)

        if 0 <= mx < self.width and 0 <= my < self.height:
            return mx, my
        return -1, -1

    def map_to_world(self, mx, my):
        """栅格坐标 -> 世界坐标 (格中心)"""
        wx = self.origin[0] + (mx + 0.5) * self.resolution
        wy = self.origin[1] + (my + 0.5) * self.resolution
        return wx, wy

    def get_occupancy_prob(self):
        """返回 [0, 1] 概率图

        prob = 1 / (1 + exp(-log_odds))
        """
        # 避免 log_odds 过大导致溢出
        clipped = np.clip(self.log_odds, -50, 50)
        prob = 1.0 / (1.0 + np.exp(-clipped))
        return prob

    def get_ros_map_data(self):
        """返回 ROS OccupancyGrid 格式的数据 (0-100, -1=未知)"""
        prob = self.get_occupancy_prob()
        # 0-100 表示占据概率, -1 表示未知
        data = np.full_like(prob, -1, dtype=np.int8)
        known = np.abs(self.log_odds) > 0.1
        data[known] = (prob[known] * 100).astype(np.int8)
        return data

    def is_occupied(self, mx, my):
        """判断栅格是否占据"""
        if not (0 <= mx < self.width and 0 <= my < self.height):
            return False
        return self.log_odds[my, mx] > 0.5

    def is_free(self, mx, my):
        """判断栅格是否空闲"""
        if not (0 <= mx < self.width and 0 <= my < self.height):
            return False
        return self.log_odds[my, mx] < -0.5

    def is_unknown(self, mx, my):
        """判断栅格是否未知"""
        if not (0 <= mx < self.width and 0 <= my < self.height):
            return True
        return abs(self.log_odds[my, mx]) <= 0.1

    def get_occupied_points_around(self, pose, radius):
        """获取机器人周围的占据点 (世界坐标)

        用于 scan-to-map ICP 匹配: 从地图中提取机器人周围 radius 范围内的
        占据栅格中心点, 转换为世界坐标点云。

        Args:
            pose: [x, y, theta] 机器人位姿
            radius: 搜索半径 (m)

        Returns:
            points: (N, 2) 占据点世界坐标
        """
        cx, cy = pose[0], pose[1]

        # 计算搜索范围 (栅格坐标)
        min_mx = max(0, int((cx - radius - self.origin[0]) / self.resolution))
        max_mx = min(self.width - 1, int((cx + radius - self.origin[0]) / self.resolution))
        min_my = max(0, int((cy - radius - self.origin[1]) / self.resolution))
        max_my = min(self.height - 1, int((cy + radius - self.origin[1]) / self.resolution))

        if min_mx > max_mx or min_my > max_my:
            return np.array([]).reshape(0, 2)

        # 提取子区域, 向量化查找占据栅格
        sub = self.log_odds[min_my:max_my + 1, min_mx:max_mx + 1]
        occupied = np.argwhere(sub > 0.5)

        if len(occupied) == 0:
            return np.array([]).reshape(0, 2)

        # 转换为世界坐标
        occupied_mx = occupied[:, 1] + min_mx
        occupied_my = occupied[:, 0] + min_my
        wx = self.origin[0] + (occupied_mx + 0.5) * self.resolution
        wy = self.origin[1] + (occupied_my + 0.5) * self.resolution

        return np.column_stack([wx, wy])

    def update_from_scan(self, robot_pose, scan_ranges, scan_angles, max_range):
        """用一帧激光扫描更新地图

        Args:
            robot_pose: [x, y, theta] 机器人位姿 (世界系)
            scan_ranges: (N,) 距离数组, inf 表示未命中
            scan_angles: (N,) 射线角度 (机器人系, 相对于前方)
            max_range: 最大有效距离 (m)
        """
        # 地图更新 → 距离变换缓存失效
        if hasattr(self, '_dist_map'):
            self._dist_map = None

        rx, ry, rtheta = robot_pose

        # 机器人栅格坐标
        robot_mx, robot_my = self.world_to_map(rx, ry)
        if robot_mx < 0:
            return

        for i in range(len(scan_ranges)):
            r = scan_ranges[i]
            angle = scan_angles[i]

            # 计算射线在世界坐标系中的角度
            world_angle = rtheta + angle

            if np.isinf(r) or r >= max_range:
                # 未命中: 只标记近距离空闲区域
                end_dist = max_range * 0.8
                end_x = rx + np.cos(world_angle) * end_dist
                end_y = ry + np.sin(world_angle) * end_dist
                end_mx, end_my = self.world_to_map(end_x, end_y)
                if end_mx >= 0:
                    # 只标记空闲, 不标记占据
                    self._bresenham_update(robot_mx, robot_my, end_mx, end_my,
                                           mark_end_occupied=False)
            else:
                # 命中: 标记路径空闲, 端点占据
                end_x = rx + np.cos(world_angle) * r
                end_y = ry + np.sin(world_angle) * r
                end_mx, end_my = self.world_to_map(end_x, end_y)
                if end_mx >= 0:
                    self._bresenham_update(robot_mx, robot_my, end_mx, end_my,
                                           mark_end_occupied=True)

    def _bresenham_update(self, x0, y0, x1, y1, mark_end_occupied=True):
        """Bresenham 直线算法, 更新路径上的栅格

        路径上的栅格标记为空闲, 端点标记为占据
        """
        dx = abs(x1 - x0)
        dy = abs(y1 - y0)
        sx = 1 if x0 < x1 else -1
        sy = 1 if y0 < y1 else -1
        err = dx - dy

        x, y = x0, y0

        while True:
            # 标记路径上的栅格为空闲
            if 0 <= x < self.width and 0 <= y < self.height:
                self.log_odds[y, x] += self.log_odd_free

            if x == x1 and y == y1:
                # 端点标记为占据
                if mark_end_occupied and 0 <= x < self.width and 0 <= y < self.height:
                    self.log_odds[y, x] -= self.log_odd_free  # 撤销空闲标记
                    self.log_odds[y, x] += self.log_odd_occupied
                break

            e2 = 2 * err
            if e2 > -dy:
                err -= dy
                x += sx
            if e2 < dx:
                err += dx
                y += sy

        # 裁剪到 [lo, hi]
        np.clip(self.log_odds, self.log_odd_lo, self.log_odd_hi, out=self.log_odds)

    def compute_distance_transform(self):
        """计算全局距离变换

        使用 scipy.ndimage.distance_transform_edt 高效计算每个栅格
        到最近障碍物的距离。结果缓存在 self._dist_map 中。

        Returns:
            dist_map: (H, W) float32, 每个栅格到最近障碍物的距离 (栅格单位)
        """
        from scipy.ndimage import distance_transform_edt

        # 占据栅格为 True 的位置, 距离为 0; 其余为 True (需要计算距离)
        occupied = self.log_odds > 0.5
        # distance_transform_edt: 对每个 False 像素计算到最近 True 像素的欧氏距离
        # 我们需要反转: 对每个非占据像素计算到最近占据像素的距离
        dist_map = distance_transform_edt(~occupied).astype(np.float32)

        self._dist_map = dist_map
        return dist_map

    def get_inflation_cost(self, mx, my, inflation_radius, cost_weight=3.0,
                           min_clearance=0.20):
        """获取栅格的膨胀代价 (用于路径规划)

        需要先调用 compute_distance_transform() 计算距离图。
        距离障碍物越近, 代价越高。

        设计:
        - dist < min_clearance: 不可通行 (机器人无法通过)
        - min_clearance <= dist < inflation_radius: 指数衰减代价
        - dist >= inflation_radius: 基础代价 1.0

        Args:
            mx, my: 栅格坐标
            inflation_radius: 膨胀半径 (栅格单位)
            cost_weight: 膨胀代价权重 (越大越远离障碍物)
            min_clearance: 最小通行距离 (栅格单位), 小于此值不可通行

        Returns:
            cost: 1.0 (远离障碍) ~ inf (占据/太窄)
        """
        if not (0 <= mx < self.width and 0 <= my < self.height):
            return float('inf')

        if not hasattr(self, '_dist_map') or self._dist_map is None:
            self.compute_distance_transform()

        dist = self._dist_map[my, mx]

        # 不可通行: 距障碍物太近, 机器人无法通过
        if dist <= min_clearance:
            return float('inf')

        # 远离膨胀区域: 基础代价
        if dist >= inflation_radius:
            return 1.0

        # 指数衰减: 越近代价越高 (比线性更陡)
        # t = 0 在 min_clearance, t = 1 在 inflation_radius
        t = (dist - min_clearance) / (inflation_radius - min_clearance)
        # 代价 = 1 + weight * (1-t)^2
        # 在 min_clearance 处: 1 + weight
        # 在 inflation_radius 处: 1.0
        return 1.0 + cost_weight * (1.0 - t) ** 2

    def get_coverage(self):
        """返回地图覆盖率 (0-1)"""
        total = self.width * self.height
        known = np.sum(np.abs(self.log_odds) > 0.1)
        return known / total

    def reset(self):
        """重置地图"""
        self.log_odds.fill(0)
