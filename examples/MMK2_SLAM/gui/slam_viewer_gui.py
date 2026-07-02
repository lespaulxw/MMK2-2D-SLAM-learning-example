"""
SLAM 调试 GUI
==============
使用 Matplotlib 实现实时 2×2 面板调试界面, 与 GLFW 主循环共存。

面板布局:
  ┌──────────────────┬──────────────────┐
  │  占据栅格地图     │  激光扫描极坐标   │
  │  (机器人/轨迹/    │  (实时雷达扫描)   │
  │   路径/目标叠加)  │                  │
  ├──────────────────┼──────────────────┤
  │  轨迹俯视图       │  状态信息文本     │
  │  (SLAM vs 里程计) │  (覆盖率/ICP得分  │
  │                  │   /位姿/帧数等)   │
  └──────────────────┴──────────────────┘

使用 plt.ion() 非阻塞模式, 每帧 canvas.draw_idle() + flush_events() 更新。
"""
import warnings
import numpy as np
import matplotlib
# 选择可用的交互式后端 (TkAgg > Qt5Agg > Qt4Agg > default)
for _backend in ['TkAgg', 'Qt5Agg', 'Qt4Agg']:
    try:
        matplotlib.use(_backend)
        break
    except Exception:
        continue

# 抑制字形缺失警告 (DejaVu Sans 不含中文字形)
warnings.filterwarnings('ignore', message='Glyph .* missing', category=UserWarning)

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrow
from matplotlib.lines import Line2D

# 尝试设置中文字体, 失败则使用英文标签
_CJK_FONTS = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC',
              'Noto Sans CJK TC', 'AR PL UMing CN', 'Droid Sans Fallback']
_CJK_FONT_FOUND = None
import matplotlib.font_manager as fm
for _font_name in _CJK_FONTS:
    try:
        fm.findfont(_font_name, fallback_to_default=False)
        _CJK_FONT_FOUND = _font_name
        break
    except Exception:
        continue

if _CJK_FONT_FOUND:
    plt.rcParams['font.sans-serif'] = [_CJK_FONT_FOUND, 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

from config.slam_config import SLAMConfig


class SLAMViewerGUI:
    """SLAM 实时调试 GUI

    使用 Matplotlib 非阻塞模式, 可嵌入 MuJoCo GLFW 主循环。

    属性:
        fig: matplotlib Figure
        axes: 2×2 子图列表
        initialized: 是否已初始化数据
    """

    def __init__(self, config: SLAMConfig):
        self.config = config
        self.update_rate = config.gui_update_rate
        self._frame_counter = 0
        self._initialized = False

        # 创建窗口
        plt.ion()
        self.fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        self.fig.suptitle("MMK2 2D SLAM Debugger", fontsize=14, fontweight='bold')
        self.fig.canvas.manager.set_window_title("MMK2 SLAM Debugger")

        self.ax_map = axes[0, 0]
        self.ax_scan = axes[0, 1]
        self.ax_traj = axes[1, 0]
        self.ax_info = axes[1, 1]

        self._setup_axes()

        # 图形对象 (延迟创建)
        self.map_im = None
        self.map_robot_marker = None
        self.map_traj_line = None
        self.map_path_line = None
        self.map_target_marker = None

        self.scan_line = None
        self.scan_robot_dot = None

        self.traj_scam_line = None
        self.traj_odom_line = None
        self.traj_robot_dot = None

        # 导航模式状态
        self._navigation_mode = False
        self._nav_goal = None
        self._nav_goal_set = False  # 新的点击目标 (供主循环消费)

        self.info_text = None

        # 注册地图面板点击回调
        self.fig.canvas.mpl_connect('button_press_event', self._on_map_click)

        plt.show(block=False)
        plt.pause(0.01)

        print("[GUI] SLAM Viewer GUI initialized (Matplotlib non-blocking mode)")
        print("[GUI] Tip: Click on the map panel to set navigation goals in NAV mode")

    def _setup_axes(self):
        """配置子图样式"""
        # 地图面板
        self.ax_map.set_title("Occupancy Grid Map")
        self.ax_map.set_xlabel("X (m)")
        self.ax_map.set_ylabel("Y (m)")
        self.ax_map.set_aspect('equal')
        self.ax_map.set_facecolor('#1a1a2e')

        # 激光扫描面板
        self.ax_scan.set_title("LiDAR Scan (Polar)")
        self.ax_scan.set_xlabel("X (m)")
        self.ax_scan.set_ylabel("Y (m)")
        self.ax_scan.set_aspect('equal')
        self.ax_scan.set_facecolor('#1a1a2e')
        self.ax_scan.set_xlim(-self.config.lidar_cutoff_dist, self.config.lidar_cutoff_dist)
        self.ax_scan.set_ylim(-self.config.lidar_cutoff_dist, self.config.lidar_cutoff_dist)
        # 画距离圈
        for r in [1.0, 3.0, 5.0, 8.0]:
            if r < self.config.lidar_cutoff_dist:
                circle = plt.Circle((0, 0), r, fill=False, color='#444444', linestyle='--', linewidth=0.5)
                self.ax_scan.add_patch(circle)

        # 轨迹面板
        self.ax_traj.set_title("Trajectory (Top View)")
        self.ax_traj.set_xlabel("X (m)")
        self.ax_traj.set_ylabel("Y (m)")
        self.ax_traj.set_aspect('equal')
        self.ax_traj.set_facecolor('#1a1a2e')

        # 信息面板
        self.ax_info.set_title("Status Info")
        self.ax_info.axis('off')

    def update(self, slam, robot_pose, scan_ranges, scan_angles,
               target=None, planned_path=None, max_range=None):
        """非阻塞更新所有面板

        Args:
            slam: SLAMEstimator 实例
            robot_pose: [x, y, theta] 机器人当前位姿
            scan_ranges: (N,) 激光距离
            scan_angles: (N,) 激光角度
            target: [x, y] 探索目标 (可选)
            planned_path: list of [x, y] 规划路径 (可选)
            max_range: 最大有效距离
        """
        self._frame_counter += 1

        # 按更新频率刷新
        if self._frame_counter % max(1, int(30 / self.update_rate)) != 0:
            return

        if max_range is None:
            max_range = self.config.lidar_cutoff_dist

        # 1. 更新占据栅格地图面板
        self._update_map_panel(slam, robot_pose, target, planned_path)

        # 2. 更新激光扫描面板
        self._update_scan_panel(scan_ranges, scan_angles, max_range)

        # 3. 更新轨迹面板
        self._update_trajectory_panel(slam, robot_pose)

        # 4. 更新状态信息面板
        self._update_info_panel(slam, robot_pose, target)

        # 非阻塞刷新
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def set_navigation_mode(self, enabled=True):
        """切换导航模式"""
        self._navigation_mode = enabled
        mode_str = "NAVIGATION" if enabled else "MAPPING"
        self.fig.suptitle(f"MMK2 2D SLAM — [{mode_str}]", fontsize=14, fontweight='bold')
        if enabled:
            self.ax_map.set_title("Occupancy Grid Map (Frozen) — Click to set goal")
        else:
            self.ax_map.set_title("Occupancy Grid Map")
            self._nav_goal = None
            self._nav_goal_set = False
            # 隐藏目标标记
            if self.map_target_marker is not None:
                self.map_target_marker.set_data([], [])

    def _on_map_click(self, event):
        """地图面板点击回调 — 设置导航目标"""
        if not self._navigation_mode:
            return
        if event.inaxes != self.ax_map:
            return
        if event.xdata is None or event.ydata is None:
            return
        self._nav_goal = [event.xdata, event.ydata]
        self._nav_goal_set = True
        print(f"[GUI] Nav goal set: ({event.xdata:.2f}, {event.ydata:.2f})")

    def has_new_goal(self):
        """是否有新的点击目标"""
        return self._nav_goal_set

    def get_nav_goal(self):
        """获取导航目标"""
        return self._nav_goal

    def clear_new_goal(self):
        """清除新目标标记"""
        self._nav_goal_set = False

    def _update_map_panel(self, slam, robot_pose, target, planned_path):
        """更新占据栅格地图面板"""
        prob_map = slam.get_map_prob()  # (H, W) [0,1]
        grid = slam.grid

        # 计算地图显示范围 (世界坐标)
        extent = [
            grid.origin[0],
            grid.origin[0] + grid.width * grid.resolution,
            grid.origin[1],
            grid.origin[1] + grid.height * grid.resolution
        ]

        if self.map_im is None:
            # 首次创建
            self.map_im = self.ax_map.imshow(
                prob_map, cmap='RdYlGn_r', vmin=0, vmax=1,
                extent=extent, origin='lower', aspect='equal'
            )
            self.ax_map.set_xlim(extent[0], extent[1])
            self.ax_map.set_ylim(extent[2], extent[3])
        else:
            self.map_im.set_data(prob_map)
            self.map_im.set_extent(extent)

        # 机器人位置标记
        if self.map_robot_marker is None:
            self.map_robot_marker, = self.ax_map.plot(
                robot_pose[0], robot_pose[1], 'bo', markersize=8,
                markeredgecolor='white', markeredgewidth=1.5
            )
        else:
            self.map_robot_marker.set_data([robot_pose[0]], [robot_pose[1]])

        # 轨迹
        traj = slam.get_trajectory()
        if len(traj) > 1:
            if self.map_traj_line is None:
                self.map_traj_line, = self.ax_map.plot(
                    traj[:, 0], traj[:, 1], 'c-', linewidth=1.0, alpha=0.7
                )
            else:
                self.map_traj_line.set_data(traj[:, 0], traj[:, 1])

        # 规划路径
        if planned_path is not None and len(planned_path) > 1:
            path_arr = np.array(planned_path)
            if self.map_path_line is None:
                self.map_path_line, = self.ax_map.plot(
                    path_arr[:, 0], path_arr[:, 1], 'y--', linewidth=1.5, alpha=0.8
                )
            else:
                self.map_path_line.set_data(path_arr[:, 0], path_arr[:, 1])

        # 目标点
        if target is not None:
            if self.map_target_marker is None:
                self.map_target_marker, = self.ax_map.plot(
                    target[0], target[1], 'r*', markersize=12,
                    markeredgecolor='white', markeredgewidth=1.0
                )
            else:
                self.map_target_marker.set_data([target[0]], [target[1]])

        # 机器人朝向箭头
        arrow_len = 0.3
        dx = arrow_len * np.cos(robot_pose[2])
        dy = arrow_len * np.sin(robot_pose[2])
        # 移除旧箭头, 画新的 (简单实现)
        for patch in self.ax_map.patches[:]:
            if hasattr(patch, '_is_heading_arrow') and patch._is_heading_arrow:
                patch.remove()
        arrow = self.ax_map.arrow(
            robot_pose[0], robot_pose[1], dx, dy,
            head_width=0.1, head_length=0.05, fc='lime', ec='lime'
        )
        arrow._is_heading_arrow = True

    def _update_scan_panel(self, ranges, angles, max_range):
        """更新激光扫描面板"""
        # 转换为笛卡尔坐标 (机器人坐标系)
        valid = np.isfinite(ranges) & (ranges > 0.1) & (ranges < max_range)
        if np.sum(valid) > 0:
            r = ranges[valid]
            a = angles[valid]
            x = r * np.cos(a)
            y = r * np.sin(a)

            if self.scan_line is None:
                self.scan_line, = self.ax_scan.plot(
                    x, y, 'g.', markersize=2, alpha=0.8
                )
            else:
                self.scan_line.set_data(x, y)

        if self.scan_robot_dot is None:
            self.scan_robot_dot, = self.ax_scan.plot(
                0, 0, 'bo', markersize=6, markeredgecolor='white'
            )

    def _update_trajectory_panel(self, slam, robot_pose):
        """更新轨迹面板"""
        slam_traj = slam.get_trajectory()
        odom_traj = slam.get_odom_trajectory()

        # SLAM 轨迹
        if len(slam_traj) > 1:
            if self.traj_scam_line is None:
                self.traj_scam_line, = self.ax_traj.plot(
                    slam_traj[:, 0], slam_traj[:, 1], 'c-', linewidth=1.5,
                    label='SLAM'
                )
                self.ax_traj.legend(loc='upper right', fontsize=8)
            else:
                self.traj_scam_line.set_data(slam_traj[:, 0], slam_traj[:, 1])

        # 里程计轨迹
        if len(odom_traj) > 1:
            if self.traj_odom_line is None:
                self.traj_odom_line, = self.ax_traj.plot(
                    odom_traj[:, 0], odom_traj[:, 1], 'm--', linewidth=1.0,
                    alpha=0.6, label='Odometry'
                )
                self.ax_traj.legend(loc='upper right', fontsize=8)
            else:
                self.traj_odom_line.set_data(odom_traj[:, 0], odom_traj[:, 1])

        # 机器人当前位置
        if self.traj_robot_dot is None:
            self.traj_robot_dot, = self.ax_traj.plot(
                robot_pose[0], robot_pose[1], 'bo', markersize=8,
                markeredgecolor='white', markeredgewidth=1.5
            )
        else:
            self.traj_robot_dot.set_data([robot_pose[0]], [robot_pose[1]])

        # 自动调整范围
        all_x = []
        all_y = []
        if len(slam_traj) > 0:
            all_x.extend(slam_traj[:, 0])
            all_y.extend(slam_traj[:, 1])
        if len(odom_traj) > 0:
            all_x.extend(odom_traj[:, 0])
            all_y.extend(odom_traj[:, 1])
        if len(all_x) > 0:
            margin = 1.0
            xmin, xmax = min(all_x) - margin, max(all_x) + margin
            ymin, ymax = min(all_y) - margin, max(all_y) + margin
            self.ax_traj.set_xlim(xmin, xmax)
            self.ax_traj.set_ylim(ymin, ymax)

    def _update_info_panel(self, slam, robot_pose, target):
        """更新状态信息面板"""
        stats = slam.get_stats()

        # 模式指示
        mode_str = "NAVIGATION" if self._navigation_mode else "MAPPING"
        mode_color = "lime" if self._navigation_mode else "cyan"

        lines = [
            f"=== Mode: {mode_str} ===\n",
            f"Scan frames:     {stats['scan_count']}\n",
            f"ICP corrections: {stats['corrections_applied']}\n",
            f"Avg ICP score:   {stats['avg_icp_score']:.3f}\n",
            f"Map coverage:    {stats['coverage']*100:.1f}%\n",
            f"Trajectory pts:  {stats['trajectory_length']}\n",
            f"\n=== Robot Pose ===\n",
            f"X:     {robot_pose[0]:.3f} m\n",
            f"Y:     {robot_pose[1]:.3f} m\n",
            f"Theta: {np.degrees(robot_pose[2]):.1f} deg\n",
        ]

        if self._navigation_mode and self._nav_goal is not None:
            dist = np.sqrt(
                (self._nav_goal[0] - robot_pose[0])**2 +
                (self._nav_goal[1] - robot_pose[1])**2
            )
            lines.append(f"\n=== Nav Goal ===\n")
            lines.append(f"Goal X: {self._nav_goal[0]:.3f} m\n")
            lines.append(f"Goal Y: {self._nav_goal[1]:.3f} m\n")
            lines.append(f"Dist:   {dist:.3f} m\n")
        elif target is not None:
            dist = np.sqrt((target[0]-robot_pose[0])**2 + (target[1]-robot_pose[1])**2)
            lines.append(f"\n=== Exploration Target ===\n")
            lines.append(f"Target X: {target[0]:.3f} m\n")
            lines.append(f"Target Y: {target[1]:.3f} m\n")
            lines.append(f"Distance: {dist:.3f} m\n")

        info_str = "".join(lines)

        if self.info_text is None:
            self.info_text = self.ax_info.text(
                0.05, 0.95, info_str,
                transform=self.ax_info.transAxes,
                fontsize=10, verticalalignment='top',
                fontfamily='monospace',
                color='#e0e0e0'
            )
        else:
            self.info_text.set_text(info_str)

    def close(self):
        """关闭 GUI"""
        plt.close(self.fig)
        plt.ioff()
        print("[GUI] SLAM Viewer GUI closed")
