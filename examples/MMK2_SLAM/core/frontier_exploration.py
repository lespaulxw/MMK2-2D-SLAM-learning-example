"""
前沿探索与路径规划
===================
实现基于前沿的自主探索策略：
1. 前沿检测: 在占据栅格地图上寻找未知-空闲边界
2. 前沿聚类: BFS 将连通前沿分组
3. 目标选择: 选择最近可达的前沿中心
4. A* 路径规划: 在栅格地图上规划无碰撞路径

参考: Yamauchi (1997) "A Frontier-Based Approach for Autonomous Exploration"
"""
import numpy as np
from collections import deque
import heapq
from config.slam_config import SLAMConfig


class FrontierExplorer:
    """前沿探索器

    整合前沿检测、目标选择和 A* 路径规划，
    用于主动 SLAM 自主探索。

    改进的目标选择:
    - 过滤靠近障碍物的前沿格 (安全边距)
    - 综合评分: 距离 + 安全度 + 访问惩罚
    - 前沿访问记忆: 避免反复选择同一区域

    属性:
        grid: OccupancyGrid 实例
        min_frontier_size: 最小前沿格数
        reach_threshold: 到达目标阈值 (m)
    """

    def __init__(self, config: SLAMConfig):
        self.min_frontier_size = config.frontier_min_size
        self.reach_threshold = config.frontier_reach_threshold
        # 路径规划安全参数
        self.inflation_radius = getattr(config, 'path_inflation_radius', 0.50)  # 米
        self.inflation_cost_weight = getattr(config, 'path_inflation_weight', 5.0)  # 膨胀代价权重
        self.min_clearance = getattr(config, 'path_min_clearance', 0.15)  # 最小通行距离 (米)
        # 前沿安全参数
        self.frontier_safety_margin = getattr(config, 'frontier_safety_margin', 0.40)  # 米
        # 前沿访问记忆: 记录最近访问过的前沿中心, 避免重复
        self.frontier_memory = []  # list of (wx, wy) 已访问前沿中心
        self.frontier_memory_max = getattr(config, 'frontier_memory_max', 8)
        self.frontier_visit_penalty = getattr(config, 'frontier_visit_penalty', 2.0)  # 米
        self.frontier_forget_dist = getattr(config, 'frontier_forget_dist', 1.5)  # 米 — 距离多近算"访问过"

    def find_frontiers(self, grid):
        """检测地图上的所有前沿 (带安全过滤)

        前沿定义: 空闲栅格中至少有一个未知邻居的栅格。
        过滤: 移除距障碍物太近的前沿格, 确保目标安全可达。

        Args:
            grid: OccupancyGrid 实例

        Returns:
            list of list: 每个元素是一个前沿簇,
                          每个簇是 [(mx, my), ...] 栅格坐标列表
        """
        log_odds = grid.log_odds
        height, width = log_odds.shape

        # 预计算距离变换, 用于安全过滤
        dist_map = grid.compute_distance_transform()
        safety_cells = self.frontier_safety_margin / grid.resolution  # 安全边距 (栅格)

        # 标记已访问
        visited = np.zeros((height, width), dtype=bool)
        frontiers = []

        # 8 邻域偏移
        neighbors_8 = [(-1, -1), (-1, 0), (-1, 1),
                       (0, -1),           (0, 1),
                       (1, -1),  (1, 0),  (1, 1)]

        for my in range(height):
            for mx in range(width):
                # 跳过已访问和非空闲栅格
                if visited[my, mx] or not grid.is_free(mx, my):
                    continue

                # BFS 搜索连通的空闲区域, 同时检测前沿
                queue = deque()
                queue.append((mx, my))
                visited[my, mx] = True

                # 当前连通空闲区域中的前沿点
                frontier_cells = []

                while queue:
                    cx, cy = queue.popleft()

                    # 检查是否是前沿点 (有未知邻居)
                    is_frontier = False
                    for dx, dy in neighbors_8:
                        nx, ny = cx + dx, cy + dy
                        if 0 <= nx < width and 0 <= ny < height:
                            if not visited[ny, nx] and grid.is_free(nx, ny):
                                visited[ny, nx] = True
                                queue.append((nx, ny))
                            elif grid.is_unknown(nx, ny):
                                is_frontier = True

                    if is_frontier:
                        # 安全过滤: 只保留距障碍物足够远的前沿格
                        if dist_map[cy, cx] >= safety_cells:
                            frontier_cells.append((cx, cy))

                # 过滤太小的前沿
                if len(frontier_cells) >= self.min_frontier_size:
                    frontiers.append(frontier_cells)

        return frontiers

    def select_target(self, frontiers, robot_pose, grid):
        """综合评分选择最优前沿目标

        对每个前沿簇:
        1. 计算质心
        2. 在簇内选择距质心最近且安全的格作为候选目标
        3. 综合评分 = 距离 - 安全奖励 - 访问惩罚

        Args:
            frontiers: find_frontiers() 的返回值
            robot_pose: [x, y, theta] 机器人位姿
            grid: OccupancyGrid 实例

        Returns:
            target: [wx, wy] 世界坐标目标点, None 表示无前沿
        """
        if not frontiers:
            return None

        rx, ry = robot_pose[0], robot_pose[1]

        # 距离变换用于安全度评估
        dist_map = grid.compute_distance_transform()
        res = grid.resolution
        safe_dist_cells = self.frontier_safety_margin / res  # 安全距离 (栅格)

        best_target = None
        best_score = float('inf')

        for frontier in frontiers:
            # 计算前沿质心 (世界坐标)
            wx_sum, wy_sum = 0.0, 0.0
            for mx, my in frontier:
                wx, wy = grid.map_to_world(mx, my)
                wx_sum += wx
                wy_sum += wy

            cx = wx_sum / len(frontier)
            cy = wy_sum / len(frontier)

            # 在簇内选择距质心最近且安全的格作为实际目标
            # 避免质心本身在墙边或未知区域
            best_cell = None
            best_cell_dist = float('inf')
            min_obs_dist = float('inf')

            for mx, my in frontier:
                obs_d = dist_map[my, mx] * res  # 距障碍物距离 (米)
                if obs_d < min_obs_dist:
                    min_obs_dist = obs_d

                # 距质心的距离
                wx, wy = grid.map_to_world(mx, my)
                d_to_centroid = np.sqrt((wx - cx)**2 + (wy - cy)**2)
                # 安全且距质心近
                if obs_d >= safe_dist_cells and d_to_centroid < best_cell_dist:
                    best_cell_dist = d_to_centroid
                    best_cell = (wx, wy)

            # 如果没有安全的格, 选距障碍物最远的格
            if best_cell is None:
                max_obs = 0.0
                for mx, my in frontier:
                    obs_d = dist_map[my, mx] * res
                    wx, wy = grid.map_to_world(mx, my)
                    d_to_centroid = np.sqrt((wx - cx)**2 + (wy - cy)**2)
                    # 综合: 距障碍物远 + 距质心近
                    score = d_to_centroid - obs_d * 2.0
                    if score < best_cell_dist:
                        best_cell_dist = score
                        best_cell = (wx, wy)

            if best_cell is None:
                continue

            target_x, target_y = best_cell

            # === 综合评分 ===
            dist = np.sqrt((target_x - rx) ** 2 + (target_y - ry) ** 2)

            # 安全奖励: 距障碍物越远, 有效距离越小 (最多奖励 2m)
            safety_bonus = min(min_obs_dist, 2.0)
            effective_dist = dist - safety_bonus

            # 访问惩罚: 与记忆中的已访问前沿太近 → 增加有效距离
            for mem_x, mem_y in self.frontier_memory:
                mem_dist = np.sqrt((target_x - mem_x) ** 2 + (target_y - mem_y) ** 2)
                if mem_dist < self.frontier_forget_dist:
                    effective_dist += self.frontier_visit_penalty
                    break

            if effective_dist < best_score:
                best_score = effective_dist
                best_target = [target_x, target_y]

        return best_target

    def plan_path(self, start_world, goal_world, grid):
        """A* 路径规划 (带障碍物膨胀安全边距)

        在栅格地图上使用 A* 算法规划从起点到终点的路径。
        靠近障碍物的路径有更高的代价, 使机器人保持安全距离。

        代价设定:
            - 空闲栅格: 1.0 + 膨胀代价
            - 未知栅格: 5.0 + 膨胀代价 (可以探索但不优先)
            - 占据栅格: 不可通行

        Args:
            start_world: [sx, sy] 世界坐标起点
            goal_world: [gx, gy] 世界坐标终点
            grid: OccupancyGrid 实例

        Returns:
            path: list of [wx, wy] 世界坐标路径点, None 表示规划失败
        """
        # 转换为栅格坐标
        start_mx, start_my = grid.world_to_map(start_world[0], start_world[1])
        goal_mx, goal_my = grid.world_to_map(goal_world[0], goal_world[1])

        # 边界检查
        if start_mx < 0 or goal_mx < 0:
            return None

        # 预计算距离变换 (用于膨胀代价)
        grid.compute_distance_transform()
        inflation_radius_cells = self.inflation_radius / grid.resolution
        min_clearance_cells = self.min_clearance / grid.resolution  # 最小通行距离

        # 如果目标占据, 尝试找最近的空闲点
        if grid.is_occupied(goal_mx, goal_my):
            goal_mx, goal_my = self._find_nearest_free(goal_mx, goal_my, grid)
            if goal_mx < 0:
                return None

        # A* 算法
        open_set = []
        heapq.heappush(open_set, (0.0, (start_mx, start_my)))

        came_from = {}
        g_score = {(start_mx, start_my): 0.0}

        # 8 邻域 (对角线代价 sqrt(2))
        neighbors_8 = [(-1, -1, 1.414), (-1, 0, 1.0), (-1, 1, 1.414),
                       (0, -1, 1.0),                      (0, 1, 1.0),
                       (1, -1, 1.414),  (1, 0, 1.0),  (1, 1, 1.414)]

        max_iterations = grid.width * grid.height
        iterations = 0

        while open_set and iterations < max_iterations:
            iterations += 1
            _, current = heapq.heappop(open_set)
            cx, cy = current

            # 到达目标
            if cx == goal_mx and cy == goal_my:
                return self._reconstruct_path(came_from, current, grid)

            for dx, dy, base_cost in neighbors_8:
                nx, ny = cx + dx, cy + dy

                # 边界检查
                if not (0 <= nx < grid.width and 0 <= ny < grid.height):
                    continue

                # 不可通行
                if grid.is_occupied(nx, ny):
                    continue

                # 膨胀代价: 靠近障碍物代价更高
                inflation = grid.get_inflation_cost(
                    nx, ny, inflation_radius_cells,
                    cost_weight=self.inflation_cost_weight,
                    min_clearance=min_clearance_cells
                )
                if inflation == float('inf'):
                    continue  # 太靠近障碍物, 不可通行

                # 基础代价 + 膨胀代价
                if grid.is_unknown(nx, ny):
                    step_cost = base_cost * 5.0 * inflation
                else:
                    step_cost = base_cost * inflation

                tentative_g = g_score[current] + step_cost

                neighbor = (nx, ny)
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g

                    # 启发式: 欧氏距离 (保持可接受性)
                    h = np.sqrt((nx - goal_mx) ** 2 + (ny - goal_my) ** 2)
                    f = tentative_g + h
                    heapq.heappush(open_set, (f, neighbor))

        return None  # 规划失败

    def _find_nearest_free(self, mx, my, grid, max_radius=10):
        """在给定点附近寻找最近的空闲栅格"""
        for r in range(1, max_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    if abs(dx) != r and abs(dy) != r:
                        continue
                    nx, ny = mx + dx, my + dy
                    if 0 <= nx < grid.width and 0 <= ny < grid.height:
                        if grid.is_free(nx, ny):
                            return nx, ny
        return -1, -1

    def _reconstruct_path(self, came_from, current, grid):
        """从 A* 结果重建路径, 转换为世界坐标"""
        path = []
        while current in came_from:
            mx, my = current
            wx, wy = grid.map_to_world(mx, my)
            path.append([wx, wy])
            current = came_from[current]

        # 添加起点
        mx, my = current
        wx, wy = grid.map_to_world(mx, my)
        path.append([wx, wy])

        path.reverse()
        return path

    def get_exploration_target(self, robot_pose, grid):
        """完整探索流程: 检测前沿 -> 选择目标 -> 规划路径

        流程:
        1. 检测前沿 (过滤靠近障碍物的格)
        2. 综合评分选择最优目标 (距离+安全+访问惩罚)
        3. 验证目标在空闲空间
        4. A* 规划安全路径
        5. 记录访问记忆

        Args:
            robot_pose: [x, y, theta]
            grid: OccupancyGrid 实例

        Returns:
            target: [wx, wy] 目标点, None 表示探索完成
            path: list of [wx, wy] 路径, None 表示无路径
        """
        frontiers = self.find_frontiers(grid)
        target = self.select_target(frontiers, robot_pose, grid)

        if target is None:
            return None, None

        # 验证目标在空闲空间, 否则找最近空闲点
        gmx, gmy = grid.world_to_map(target[0], target[1])
        if gmx < 0 or not grid.is_free(gmx, gmy):
            gmx, gmy = self._find_nearest_free(
                gmx if gmx >= 0 else 0,
                gmy if gmy >= 0 else 0,
                grid, max_radius=20
            )
            if gmx < 0:
                return None, None
            target[0], target[1] = grid.map_to_world(gmx, gmy)

        path = self.plan_path(
            [robot_pose[0], robot_pose[1]],
            target,
            grid
        )

        if path is None:
            # 路径规划失败, 仍返回目标 (可以直接朝目标移动)
            return target, None

        # 记录访问记忆
        self._add_frontier_memory(target)

        return target, path

    def _add_frontier_memory(self, target):
        """记录已访问的前沿中心, 用于避免重复选择"""
        self.frontier_memory.append((target[0], target[1]))
        if len(self.frontier_memory) > self.frontier_memory_max:
            self.frontier_memory.pop(0)

    def reset_memory(self):
        """重置前沿访问记忆"""
        self.frontier_memory = []
