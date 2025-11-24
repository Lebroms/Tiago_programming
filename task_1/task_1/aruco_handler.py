#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import PoseStamped, TransformStamped
import cv2
from cv_bridge import CvBridge
import numpy as np
from scipy.spatial.transform import Rotation as R
from tf2_ros import Buffer, TransformListener, TransformBroadcaster

class ArucoDetector(Node):
    def __init__(self):
        super().__init__('aruco_detector')
        self.bridge = CvBridge()
        self.camera_matrix = None
        self.dist_coeffs = None

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self.tf_broadcaster = TransformBroadcaster(self)

        # Publisher PoseStamped per ogni marker
        self.marker_publishers = {}
        # Memoria delle ultime pose viste (per TF persistente)
        self.last_marker_poses = {}

        # Timer per ripubblicare continuamente i TF ogni 0.1s
        self.create_timer(0.1, self.publish_last_transforms)

        # Subscriptions
        self.create_subscription(CameraInfo, '/head_front_camera/rgb/camera_info', self.camera_info_cb, 10)
        self.create_subscription(Image, '/head_front_camera/rgb/image_raw', self.image_cb, 10)

        # Config ArUco
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters_create()
        self.marker_size = 0.06  # metri

    def camera_info_cb(self, msg: CameraInfo):
        self.camera_matrix = np.array(msg.k).reshape((3, 3))
        self.dist_coeffs = np.array(msg.d)
        self.get_logger().info("Camera calibration received.")

    def image_cb(self, msg: Image):
        if self.camera_matrix is None:
            return

        cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)

        corners, ids, _ = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        if ids is None:
            return

        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(corners, self.marker_size, self.camera_matrix, self.dist_coeffs)

        for i, marker_id in enumerate(ids.flatten()):
            tvec = tvecs[i][0]
            rvec = rvecs[i][0]

            R_mat, _ = cv2.Rodrigues(rvec)
            quat = R.from_matrix(R_mat).as_quat()

            ps = PoseStamped()
            ps.header.stamp = self.get_clock().now().to_msg()
            ps.header.frame_id = 'head_front_camera_optical_frame'
            ps.pose.position.x = float(tvec[0])
            ps.pose.position.y = float(tvec[1])
            ps.pose.position.z = float(tvec[2])
            ps.pose.orientation.x = float(quat[0])
            ps.pose.orientation.y = float(quat[1])
            ps.pose.orientation.z = float(quat[2])
            ps.pose.orientation.w = float(quat[3])

            try:
                transform = self.tf_buffer.lookup_transform('base_footprint', ps.header.frame_id, rclpy.time.Time())
                trans = transform.transform.translation
                rot = transform.transform.rotation

                T = np.eye(4)
                T[:3, 3] = [trans.x, trans.y, trans.z]
                R_q = R.from_quat([rot.x, rot.y, rot.z, rot.w])
                T[:3, :3] = R_q.as_matrix()

                marker_in_cam = np.array([ps.pose.position.x, ps.pose.position.y, ps.pose.position.z, 1.0])
                marker_in_base = T @ marker_in_cam

                marker_rot_cam = R.from_quat([ps.pose.orientation.x, ps.pose.orientation.y, ps.pose.orientation.z, ps.pose.orientation.w])
                marker_rot_base = R_q * marker_rot_cam

                ps_base = PoseStamped()
                ps_base.header.stamp = ps.header.stamp
                ps_base.header.frame_id = 'base_footprint'
                ps_base.pose.position.x = marker_in_base[0]
                ps_base.pose.position.y = marker_in_base[1]
                ps_base.pose.position.z = marker_in_base[2]
                ps_base.pose.orientation.x = marker_rot_base.as_quat()[0]
                ps_base.pose.orientation.y = marker_rot_base.as_quat()[1]
                ps_base.pose.orientation.z = marker_rot_base.as_quat()[2]
                ps_base.pose.orientation.w = marker_rot_base.as_quat()[3]

                # Pubblica PoseStamped
                topic_name = f"/aruco_pose_{marker_id}"
                if marker_id not in self.marker_publishers:
                    pub = self.create_publisher(PoseStamped, topic_name, 10)
                    self.marker_publishers[marker_id] = pub
                    self.get_logger().info(f"Created publisher for marker {marker_id} → {topic_name}")

                self.marker_publishers[marker_id].publish(ps_base)

                # Salva l'ultima pose per TF persistente
                self.last_marker_poses[marker_id] = ps_base

            except Exception as e:
                self.get_logger().warn(f"Transform failed for marker {marker_id}: {e}")

    def publish_last_transforms(self):
        now = self.get_clock().now().to_msg()
        for marker_id, ps_base in self.last_marker_poses.items():
            t = TransformStamped()
            t.header.stamp = now
            t.header.frame_id = ps_base.header.frame_id
            t.child_frame_id = f"aruco_marker_{marker_id}"
            t.transform.translation.x = ps_base.pose.position.x
            t.transform.translation.y = ps_base.pose.position.y
            t.transform.translation.z = ps_base.pose.position.z
            t.transform.rotation.x = ps_base.pose.orientation.x
            t.transform.rotation.y = ps_base.pose.orientation.y
            t.transform.rotation.z = ps_base.pose.orientation.z
            t.transform.rotation.w = ps_base.pose.orientation.w
            self.tf_broadcaster.sendTransform(t)

def main(args=None):
    rclpy.init(args=args)
    node = ArucoDetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
