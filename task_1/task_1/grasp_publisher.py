
#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped,  TransformStamped
from tf2_ros import Buffer, TransformListener, TransformBroadcaster

import numpy as np
from scipy.spatial.transform import Rotation as R

class GraspAndPlacePosePublisher(Node):
    def __init__(self):
        super().__init__('grasp_publisher')

        self.tf_broadcaster = TransformBroadcaster(self)

        # Configurazione dei marker per grasp
        self.grasp_configs = [
            {
                'frame': 'aruco_marker_1',
                'topic': '/grasp_pose_1',
                'offset_z': -0.155
            },
            {
                'frame': 'aruco_marker_2',
                'topic': '/grasp_pose_2',
                'offset_z': -0.125
            }

        ]

        # Configurazione dei marker per pose di arrivo (place) → vengono pubblicate su grasp_pose_3 e grasp_pose_4
        self.place_configs = [
            {
                'frame': 'aruco_marker_3',
                'topic': '/grasp_pose_4',  # Destinazione per oggetto da marker_2
                'offset_z': 0.075
            },
            {
                'frame': 'aruco_marker_4',
                'topic': '/grasp_pose_3',  # Destinazione per oggetto da marker_1
                'offset_z': 0.125
            }
        ]

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # Publisher grasp
        self.grasp_publishers = {
            cfg['frame']: self.create_publisher(PoseStamped, cfg['topic'], 10)
            for cfg in self.grasp_configs
        }

        # Publisher place
        self.place_publishers = {
            cfg['frame']: self.create_publisher(PoseStamped, cfg['topic'], 10)
            for cfg in self.place_configs
        }

        self.create_timer(0.1, self.update_poses)

    def update_poses(self):
        self.publish_grasp_poses()
        self.publish_place_poses()

    def publish_grasp_poses(self):
        for cfg in self.grasp_configs:
            try:
                now = rclpy.time.Time()
                transform = self.tf_buffer.lookup_transform(
                    'base_footprint', cfg['frame'], now)

                trans = transform.transform.translation
                rot   = transform.transform.rotation

                marker_pos = np.array([trans.x, trans.y, trans.z])
                marker_rot = R.from_quat([rot.x, rot.y, rot.z, rot.w])

                # offset nel frame del marker (modificato: aggiunto +0.1 su Y)
                # offset_in_marker = np.array([0.028, 0.002, cfg['offset_z']])
                offset_in_marker = np.array([0.008, 0.025, cfg['offset_z']])
                offset_in_base = marker_rot.apply(offset_in_marker)
                grasp_pos = marker_pos + offset_in_base

                # rotazione di allineamento: identica per tutti i marker
                offset_rot = R.from_euler('zy', [-90, -90], degrees=True)
                grasp_rot  = marker_rot * offset_rot

                # ——— PUBBLICO COME PoseStamped ———
                grasp_pose = PoseStamped()
                grasp_pose.header.stamp    = self.get_clock().now().to_msg()
                grasp_pose.header.frame_id = 'base_footprint'
                grasp_pose.pose.position.x = float(grasp_pos[0])
                grasp_pose.pose.position.y = float(grasp_pos[1])
                grasp_pose.pose.position.z = float(grasp_pos[2])
                q = grasp_rot.as_quat()
                grasp_pose.pose.orientation.x = q[0]
                grasp_pose.pose.orientation.y = q[1]
                grasp_pose.pose.orientation.z = q[2]
                grasp_pose.pose.orientation.w = q[3]

                self.grasp_publishers[cfg['frame']].publish(grasp_pose)
                self.get_logger().info(
                    f"Publishing grasp pose for {cfg['frame']} on {cfg['topic']}")

                # ——— PUBBLICO ANCHE COME TF ———
                t = TransformStamped()
                t.header.stamp    = grasp_pose.header.stamp
                t.header.frame_id = 'base_footprint'
                t.child_frame_id = f"grasp_{cfg['frame']}"
                t.transform.translation.x = float(grasp_pos[0])
                t.transform.translation.y = float(grasp_pos[1])
                t.transform.translation.z = float(grasp_pos[2])
                t.transform.rotation.x    = q[0]
                t.transform.rotation.y    = q[1]
                t.transform.rotation.z    = q[2]
                t.transform.rotation.w    = q[3]

                self.tf_broadcaster.sendTransform(t)

            except Exception as e:
                self.get_logger().warn(
                    f"Could not lookup transform for {cfg['frame']}: {e}")

    def publish_place_poses(self):
        for cfg in self.place_configs:
            try:
                now = rclpy.time.Time()
                transform = self.tf_buffer.lookup_transform('base_footprint', cfg['frame'], now)

                trans = transform.transform.translation
                rot = transform.transform.rotation

                marker_position = np.array([trans.x, trans.y, trans.z])
                marker_rotation = R.from_quat([rot.x, rot.y, rot.z, rot.w])

                offset_in_marker = np.array([0.008, 0.025, cfg['offset_z']])
                offset_in_base = marker_rotation.apply(offset_in_marker)
                place_position = marker_position + offset_in_base

                 # rotazione di allineamento: identica per tutti i marker
                offset_rot = R.from_euler('yzy', [-45, -90, -90], degrees=True)
                place_rotation  = marker_rotation * offset_rot

                place_pose = PoseStamped()
                place_pose.header.stamp = self.get_clock().now().to_msg()
                place_pose.header.frame_id = 'base_footprint'
                place_pose.pose.position.x = place_position[0]
                place_pose.pose.position.y = place_position[1]
                place_pose.pose.position.z = place_position[2]

                q = place_rotation.as_quat()
                place_pose.pose.orientation.x = q[0]
                place_pose.pose.orientation.y = q[1]
                place_pose.pose.orientation.z = q[2]
                place_pose.pose.orientation.w = q[3]

                self.place_publishers[cfg['frame']].publish(place_pose)

                self.get_logger().info(f"Publishing destination pose for {cfg['frame']} on {cfg['topic']}")

                 # ——— PUBBLICO ANCHE COME TF ———
                t = TransformStamped()
                t.header.stamp    = place_pose.header.stamp
                t.header.frame_id = 'base_footprint'
                t.child_frame_id = f"grasp_{cfg['frame']}"
                t.transform.translation.x = float(place_position[0])
                t.transform.translation.y = float(place_position[1])
                t.transform.translation.z = float(place_position[2])
                t.transform.rotation.x    = q[0]
                t.transform.rotation.y    = q[1]
                t.transform.rotation.z    = q[2]
                t.transform.rotation.w    = q[3]

                self.tf_broadcaster.sendTransform(t)

            except Exception as e:
                self.get_logger().warn(f"Could not lookup transform for {cfg['frame']}: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = GraspAndPlacePosePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
