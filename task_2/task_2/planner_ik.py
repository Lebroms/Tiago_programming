

#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from rclpy.action import ActionClient, ActionServer
from personal_interfaces.action import PlanAndExecute
import numpy as np
import roboticstoolbox as rtb

def ctraj_np(T1, T2, steps):
    return [T1 + (T2 - T1) * (i / (steps - 1)) for i in range(steps)]

class PlannerWithTraj(Node):
    def __init__(self):
        super().__init__('planner_with_traj')
        self.robot = rtb.ERobot.URDF('/home/lebroms/tiago_public_ws/src/task_1/task_1/urdf/tiago_robot.urdf')

        self.torso = 'torso_lift_joint'
        self.arm = [f'arm_{i}_joint' for i in range(1, 8)]
        self.grip = ['gripper_left_finger_joint', 'gripper_right_finger_joint']
        self.curr_t = 0.0
        self.curr_q = np.zeros(len(self.arm))

        self.create_subscription(JointState, '/joint_states', self._js_cb, 10)
        self._as = ActionServer(self, PlanAndExecute, 'planning_action', self._exec_cb)

        self.cli_t = ActionClient(self, FollowJointTrajectory, '/torso_controller/follow_joint_trajectory')
        self.cli_a = ActionClient(self, FollowJointTrajectory, '/arm_controller/follow_joint_trajectory')
        self.cli_g = ActionClient(self, FollowJointTrajectory, '/gripper_controller/follow_joint_trajectory')

    def _js_cb(self, msg):
        lut = dict(zip(msg.name, msg.position))
        self.curr_t = lut.get(self.torso, 0.0)
        self.curr_q = np.array([lut[j] for j in self.arm])

    def _exec_cb(self, gh):
        req = gh.request
        task = req.task_type
        self.get_logger().info(f'Task ricevuto: {task}')

        success = False
        if task == 'move':
            success = self._plan_to_pose(req.target_pose.pose)
        elif task == 'lift':
            success = self._lift(+0.15)
        elif task == 'place':
            success = self._lift(-0.15)
        elif task == 'back_to_home':
            success = self._go_to_home_position()
        elif task in ['open_gripper', 'close_gripper']:
            success = self._gripper(task)

        if success:
            gh.succeed()
        else:
            gh.abort()
        return PlanAndExecute.Result(success=success)

    
    def _plan_to_pose(self, pose):
        R = self._quat2R(pose.orientation)
        t = np.array([pose.position.x, pose.position.y, pose.position.z + 0.15])  # ADD OFFSET IN Z
        T_target = np.eye(4)
        T_target[:3, :3] = R
        T_target[:3, 3] = t

        q0_full = np.r_[self.curr_t, self.curr_q]
        T0 = self.robot.fkine(q0_full).A
        Ts = ctraj_np(T0, T_target, 3)

        q_traj = []
        q_prev = q0_full
        for T in Ts:
            sol = self.robot.ik_NR(T, q0=q_prev, pinv=True)
            if sol is None or len(sol) == 0:
                sol = q_prev[None, :]
            q_prev = sol[0]
            q_traj.append(q_prev)

        return self._send_trajectory(np.array(q_traj))


    def _lift(self, delta_z):
        q0_full = np.r_[self.curr_t, self.curr_q]
        T0 = self.robot.fkine(q0_full).A
        T1 = T0.copy()
        T1[2, 3] += delta_z
        return self._plan_to_pose_stub(T1)

    def _plan_to_pose_stub(self, T_target):
        q0_full = np.r_[self.curr_t, self.curr_q]
        T0 = self.robot.fkine(q0_full).A
        Ts = ctraj_np(T0, T_target, 5)

        q_traj = []
        q_prev = q0_full
        for T in Ts:
            sol = self.robot.ik_NR(T, q0=q_prev, pinv=True)
            if sol is None or len(sol) == 0:
                sol = q_prev[None, :]
            q_prev = sol[0]
            q_traj.append(q_prev)

        return self._send_trajectory(np.array(q_traj))

    
    
    def _send_trajectory(self, q_traj):
        traj_torso = JointTrajectory(joint_names=[self.torso])
        traj_arm = JointTrajectory(joint_names=self.arm)
        dt = 1.0

        for i, q in enumerate(q_traj):
            pt_t = JointTrajectoryPoint(positions=[float(q[0])], time_from_start=Duration(sec=int((i+1)*dt)))
            pt_a = JointTrajectoryPoint(positions=[float(x) for x in q[1:]], time_from_start=Duration(sec=int((i+1)*dt)))
            traj_torso.points.append(pt_t)
            traj_arm.points.append(pt_a)

        if not self.cli_t.wait_for_server(timeout_sec=5.0) or not self.cli_a.wait_for_server(timeout_sec=5.0):
            return False

        goal_t = FollowJointTrajectory.Goal(trajectory=traj_torso)
        goal_a = FollowJointTrajectory.Goal(trajectory=traj_arm)

        send_t = self.cli_t.send_goal_async(goal_t)
        send_a = self.cli_a.send_goal_async(goal_a)
        rclpy.spin_until_future_complete(self, send_t)
        rclpy.spin_until_future_complete(self, send_a)
        gh_t = send_t.result()
        gh_a = send_a.result()

        res_future_t = gh_t.get_result_async()
        res_future_a = gh_a.get_result_async()
        rclpy.spin_until_future_complete(self, res_future_t)
        rclpy.spin_until_future_complete(self, res_future_a)

        result_t = res_future_t.result().result
        result_a = res_future_a.result().result

        return result_t.error_code == 0 and result_a.error_code == 0

    def _gripper(self, task):
        open_val = 0.08
        close_val = 0.012
        pos = open_val if task == 'open_gripper' else close_val

        traj = JointTrajectory(
            joint_names=self.grip,
            points=[JointTrajectoryPoint(positions=[pos, open_val], time_from_start=Duration(sec=2))]
        )
        if not self.cli_g.wait_for_server(timeout_sec=5.0):
            return False
        send = self.cli_g.send_goal_async(FollowJointTrajectory.Goal(trajectory=traj))
        rclpy.spin_until_future_complete(self, send)
        return True

    def _quat2R(self, o):
        w, x, y, z = o.w, o.x, o.y, o.z
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - z*w),     2*(x*z + y*w)],
            [2*(x*y + z*w),     1 - 2*(x**2 + z**2), 2*(y*z - x*w)],
            [2*(x*z - y*w),     2*(y*z + x*w),     1 - 2*(x**2 + y**2)]
        ])
    
    
    def _go_to_home_position(self):
        torso_position = [0.35]
        arm_position = [0.07, 0.1, -3.1, 1.36, 2.05, 0.01, -0.05]
        traj_torso = JointTrajectory(
            joint_names=[self.torso],
            points=[JointTrajectoryPoint(positions=torso_position, time_from_start=Duration(sec=1, nanosec=500_000_000))]
        )
        traj_arm = JointTrajectory(
            joint_names=self.arm,
            points=[JointTrajectoryPoint(positions=arm_position, time_from_start=Duration(sec=1, nanosec=500_000_000))]
        )
        if not self.cli_t.wait_for_server(timeout_sec=5.0) or not self.cli_a.wait_for_server(timeout_sec=5.0):
            return False

        goal_t = FollowJointTrajectory.Goal(trajectory=traj_torso)
        goal_a = FollowJointTrajectory.Goal(trajectory=traj_arm)

        send_t = self.cli_t.send_goal_async(goal_t)
        send_a = self.cli_a.send_goal_async(goal_a)
        rclpy.spin_until_future_complete(self, send_t)
        rclpy.spin_until_future_complete(self, send_a)
        gh_t = send_t.result()
        gh_a = send_a.result()

        res_future_t = gh_t.get_result_async()
        res_future_a = gh_a.get_result_async()
        rclpy.spin_until_future_complete(self, res_future_t)
        rclpy.spin_until_future_complete(self, res_future_a)

        result_t = res_future_t.result().result
        result_a = res_future_a.result().result

        return result_t.error_code == 0 and result_a.error_code == 0

def main():
    rclpy.init()
    node = PlannerWithTraj()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()