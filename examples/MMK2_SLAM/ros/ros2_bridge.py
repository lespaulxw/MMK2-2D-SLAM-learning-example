"""
ROS2 桥接
===========
将 SLAM 估计器的数据发布到 ROS2 话题，供 RViz2 可视化。

发布内容:
1. LaserScan    → /scan        (激光扫描, 兼容 slam_toolbox)
2. OccupancyGrid → /map        (占据栅格地图)
3. Path         → /planned_path (规划路径)
4. Path         → /robot_trajectory (机器人轨迹)
5. MarkerArray  → /mujoco_scene  (MuJoCo 场景几何体)
6. TF 树: map → odom → base_link → laser

ROS2 为可选项: 若 rclpy 未安装, 桥接功能自动禁用。
"""
import numpy as np
from scipy.spatial.transform import Rotation

try:
    import rclpy
    from rclpy.node import Node
    from tf2_ros import TransformBroadcaster
    from sensor_msgs.msg import LaserScan
    from nav_msgs.msg import OccupancyGrid, Path
    from visualization_msgs.msg import MarkerArray, Marker
    from geometry_msgs.msg import TransformStamped, PoseStamped, Quaternion
    from std_msgs.msg import Header
    ROS2_AVAILABLE = True
except ImportError:
    ROS2_AVAILABLE = False


def euler_to_quat(yaw):
    """2D 欧拉角 → 四元数 (xyzw)"""
    return Rotation.from_euler('z', yaw).as_quat()


def _broadcast_tf(broadcaster, parent_frame, child_frame, x, y, yaw, stamp):
    """广播 2D TF 变换"""
    t = TransformStamped()
    t.header.stamp = stamp
    t.header.frame_id = parent_frame
    t.child_frame_id = child_frame

    t.transform.translation.x = float(x)
    t.transform.translation.y = float(y)
    t.transform.translation.z = 0.0

    q = euler_to_quat(yaw)
    t.transform.rotation.x = float(q[0])
    t.transform.rotation.y = float(q[1])
    t.transform.rotation.z = float(q[2])
    t.transform.rotation.w = float(q[3])

    broadcaster.sendTransform(t)


def _create_laser_scan_msg(ranges, angles, max_range, frame_id, stamp, angle_min=None, angle_max=None):
    """构造 LaserScan 消息"""
    msg = LaserScan()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp

    msg.angle_min = float(angle_min) if angle_min is not None else float(angles[0])
    msg.angle_max = float(angle_max) if angle_max is not None else float(angles[-1])
    msg.angle_increment = float((msg.angle_max - msg.angle_min) / max(len(angles) - 1, 1))
    msg.time_increment = 0.0
    msg.scan_time = 0.1
    msg.range_min = 0.1
    msg.range_max = float(max_range)

    # 处理 inf/nan → 0.0 (ROS2 LaserScan 用 inf 表示无回波, 但某些实现不兼容)
    clean_ranges = np.array(ranges, dtype=np.float32)
    clean_ranges[~np.isfinite(clean_ranges)] = float('inf')
    msg.ranges = clean_ranges.tolist()

    return msg


def _create_map_msg(map_data, grid, frame_id, stamp):
    """构造 OccupancyGrid 消息"""
    msg = OccupancyGrid()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp

    msg.info.resolution = float(grid.resolution)
    msg.info.width = int(grid.width)
    msg.info.height = int(grid.height)

    # 地图原点 (左下角)
    msg.info.origin.position.x = float(grid.origin[0])
    msg.info.origin.position.y = float(grid.origin[1])
    msg.info.origin.position.z = 0.0
    msg.info.origin.orientation.w = 1.0

    # 数据 (行优先, 从左下角开始)
    # ROS 的 OccupancyGrid 数据是从下到上排列的, 但 numpy 是从上到下
    # 需要翻转
    data = np.flipud(map_data)
    msg.data = data.flatten().tolist()

    return msg


def _create_path_msg(path_list, frame_id, stamp):
    """构造 Path 消息"""
    msg = Path()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp

    if path_list is not None:
        for point in path_list:
            pose = PoseStamped()
            pose.header.frame_id = frame_id
            pose.header.stamp = stamp
            pose.pose.position.x = float(point[0])
            pose.pose.position.y = float(point[1])
            pose.pose.position.z = 0.0
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

    return msg


def _create_trajectory_path(trajectory, frame_id, stamp):
    """从位姿轨迹构造 Path 消息"""
    msg = Path()
    msg.header.frame_id = frame_id
    msg.header.stamp = stamp

    if len(trajectory) > 0:
        for pose in trajectory:
            ps = PoseStamped()
            ps.header.frame_id = frame_id
            ps.header.stamp = stamp
            ps.pose.position.x = float(pose[0])
            ps.pose.position.y = float(pose[1])
            ps.pose.position.z = 0.0
            q = euler_to_quat(pose[2])
            ps.pose.orientation.x = float(q[0])
            ps.pose.orientation.y = float(q[1])
            ps.pose.orientation.z = float(q[2])
            ps.pose.orientation.w = float(q[3])
            msg.poses.append(ps)

    return msg


def _create_scene_markers(mj_model, mj_data, frame_id, stamp):
    """从 MuJoCo 场景创建 MarkerArray"""
    from mujoco._structs import MjGeom
    markers = []
    current_id = 0

    for i in range(mj_model.ngeom):
        geom = mj_model.geom(i)
        geom_type = geom.type
        geom_pos = geom.pos.copy()
        geom_quat = geom.quat.copy()
        geom_size = geom.size.copy()
        geom_rgba = geom.rgba.copy()

        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.id = current_id
        marker.action = Marker.ADD
        marker.pose.position.x = float(geom_pos[0])
        marker.pose.position.y = float(geom_pos[1])
        marker.pose.position.z = float(geom_pos[2])
        marker.pose.orientation.x = float(geom_quat[0])
        marker.pose.orientation.y = float(geom_quat[1])
        marker.pose.orientation.z = float(geom_quat[2])
        marker.pose.orientation.w = float(geom_quat[3])

        marker.color.r = float(geom_rgba[0])
        marker.color.g = float(geom_rgba[1])
        marker.color.b = float(geom_rgba[2])
        marker.color.a = float(geom_rgba[3]) if geom_rgba[3] > 0 else 0.5

        if geom_type == 6:  # mjGEOM_BOX
            marker.type = Marker.CUBE
            marker.scale.x = float(geom_size[0]) * 2
            marker.scale.y = float(geom_size[1]) * 2
            marker.scale.z = float(geom_size[2]) * 2
        elif geom_type == 2:  # mjGEOM_SPHERE
            marker.type = Marker.SPHERE
            marker.scale.x = float(geom_size[0]) * 2
            marker.scale.y = float(geom_size[0]) * 2
            marker.scale.z = float(geom_size[0]) * 2
        elif geom_type == 5:  # mjGEOM_CYLINDER
            marker.type = Marker.CYLINDER
            marker.scale.x = float(geom_size[0]) * 2
            marker.scale.y = float(geom_size[0]) * 2
            marker.scale.z = float(geom_size[1]) * 2
        elif geom_type == 3:  # mjGEOM_CAPSULE
            marker.type = Marker.CYLINDER
            marker.scale.x = float(geom_size[0]) * 2
            marker.scale.y = float(geom_size[0]) * 2
            marker.scale.z = float(geom_size[1]) * 2 + float(geom_size[0]) * 2
        else:
            continue

        markers.append(marker)
        current_id += 1

    return markers


class SLAMROS2Bridge:
    """ROS2 桥接器 (可选)

    将 SLAM 数据发布到 ROS2 话题, 供 RViz2 可视化。
    若 rclpy 不可用, 所有方法变为空操作。

    使用:
        bridge = SLAMROS2Bridge(config)
        bridge.publish_all(robot, slam, path)

    属性:
        node: rclpy.Node (None if ROS2 不可用)
    """

    def __init__(self, config):
        self.config = config
        self.enabled = ROS2_AVAILABLE and config.use_ros2

        if not self.enabled:
            print("[ROS2Bridge] ROS2 disabled" + 
                  (" (rclpy not available)" if not ROS2_AVAILABLE else " (config.use_ros2=False)"))
            self.node = None
            return

        if not rclpy.ok():
            rclpy.init()

        self.node = Node('mmk2_slam_bridge')

        # 发布者
        self.pub_scan = self.node.create_publisher(LaserScan, '/scan', 10)
        self.pub_map = self.node.create_publisher(OccupancyGrid, '/map', 10)
        self.pub_path = self.node.create_publisher(Path, '/planned_path', 10)
        self.pub_traj = self.node.create_publisher(Path, '/robot_trajectory', 10)
        self.pub_markers = self.node.create_publisher(MarkerArray, '/mujoco_scene', 10)

        # TF 广播者
        self.tf_broadcaster = TransformBroadcaster(self.node)

        print("[ROS2Bridge] ROS2 bridge initialized (topics: /scan /map /planned_path /robot_trajectory /mujoco_scene)")

    def publish_all(self, robot, slam, planned_path=None, target=None):
        """一次性发布所有 ROS2 数据

        Args:
            robot: MMK2SlamRobot 实例
            slam: SLAMEstimator 实例
            planned_path: list of [x, y] 或 None
            target: [x, y] 或 None
        """
        if not self.enabled or self.node is None:
            return

        stamp = self.node.get_clock().now().to_msg()

        # 1. 发布 LaserScan
        ranges, angles = robot.get_lidar_scan()
        if len(ranges) > 0:
            scan_msg = _create_laser_scan_msg(
                ranges, angles, self.config.lidar_cutoff_dist,
                'laser', stamp
            )
            self.pub_scan.publish(scan_msg)

        # 2. 发布 OccupancyGrid
        map_data = slam.get_ros_map_data()
        map_msg = _create_map_msg(map_data, slam.grid, 'map', stamp)
        self.pub_map.publish(map_msg)

        # 3. 发布规划路径
        path_msg = _create_path_msg(planned_path, 'map', stamp)
        self.pub_path.publish(path_msg)

        # 4. 发布机器人轨迹
        traj_msg = _create_trajectory_path(slam.get_trajectory(), 'map', stamp)
        self.pub_traj.publish(traj_msg)

        # 5. 发布 MuJoCo 场景 MarkerArray
        try:
            markers = _create_scene_markers(robot.mj_model, robot.mj_data, 'map', stamp)
            marker_array = MarkerArray(markers=markers)
            self.pub_markers.publish(marker_array)
        except Exception as e:
            pass

        # 6. 广播 TF: map → odom → base_link → laser
        slam_pose = slam.get_pose()
        odom_pose = slam.odom.get_pose()
        robot_pose = robot.get_robot_pose_2d()
        laser_pose = robot.get_laser_pose_2d()

        # map → odom (SLAM 校正量)
        dx = slam_pose[0] - odom_pose[0]
        dy = slam_pose[1] - odom_pose[1]
        dtheta = slam_pose[2] - odom_pose[2]
        _broadcast_tf(self.tf_broadcaster, 'map', 'odom', dx, dy, dtheta, stamp)

        # odom → base_link
        _broadcast_tf(self.tf_broadcaster, 'odom', 'base_link',
                      odom_pose[0], odom_pose[1], odom_pose[2], stamp)

        # base_link → laser
        # laser 相对于 base_link 的偏移
        dlx = laser_pose[0] - robot_pose[0]
        dly = laser_pose[1] - robot_pose[1]
        dltheta = laser_pose[2] - robot_pose[2]
        _broadcast_tf(self.tf_broadcaster, 'base_link', 'laser',
                      dlx, dly, dltheta, stamp)

    def spin_once(self):
        """处理 ROS2 事件 (非阻塞)"""
        if not self.enabled or self.node is None:
            return
        rclpy.spin_once(self.node, timeout_sec=0.0)

    def shutdown(self):
        """关闭 ROS2"""
        if not self.enabled:
            return
        if self.node is not None:
            self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        print("[ROS2Bridge] ROS2 bridge shutdown")
