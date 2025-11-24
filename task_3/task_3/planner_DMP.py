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

from roboticstoolbox import DHRobot, RevoluteDH
from movement_primitives.dmp import CartesianDMP
from pytransform3d.transformations import transform_from_pq, pq_from_transform
import os
from spatialmath import SE3
import pytransform3d.trajectories as ptr
import matplotlib.pyplot as plt
from personal_interfaces.srv import Attach
import subprocess
import shlex

def ctraj_np(T1, T2, steps):
    return [T1 + (T2 - T1) * (i / (steps - 1)) for i in range(steps)]

class Planner_DMP(Node):
    def __init__(self):
        super().__init__('planner_DMP')
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

        self.dmp_dataset_dir = "/home/lebroms/Downloads/tiago_traiettorie_SG"

        self.cli_attach = self.create_client(Attach, '/attach')

        self.cli_detach = self.create_client(Attach, '/detach')


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
            success = self._plan_with_dmp(req.marker_id, req.target_pose.pose)
        elif task == 'lift':
            success = self._lift(+0.15)
        elif task == 'place':
            success = self._lift(-0.15)
        elif task == 'back_to_home':
            success = self._go_to_home_position()
        elif task in ['open_gripper', 'close_gripper']:
            success = self._gripper(task)
        elif task == 'take':
            success = self._take(
            model_name_2 = req.model_name_2,
            link_name_2  = req.link_name_2
            )
        elif task == 'put':
            success = self._put(
            model_name_2 = req.model_name_2,
            link_name_2  = req.link_name_2
            )
        

        if success:
            gh.succeed()
        else:
            gh.abort()
        return PlanAndExecute.Result(success=success)

    
    def _plan_with_dmp(self, marker_id, pose):
        # Mappa il marker_id alla directory e al nome file base
        mapping = {
            2: ("2a-3b", "pick", "2a"),
            4: ("2a-2b", "place", "2b"),
            1: ("3a-3b", "pick", "3a"),
            3: ("2a-2b", "place", "2b")
        }

        folder, phase, position = mapping[marker_id]
        dataset_dir = os.path.join(self.dmp_dataset_dir, folder)
        file = f"{phase}_{position}_rep1.txt"
        dir = os.path.join(dataset_dir, file)

        trajs = np.loadtxt(dir, skiprows=1)  # trajs è (N, 8) [tempo + 7 giunti]

        n_steps = 300
        dt = 0.01
        execution_time = (n_steps - 1) * dt

        N = len(trajs)

        # Indici distribuiti uniformemente
        sample_indices = np.linspace(0, N - 1, n_steps, dtype=int)

        # Campionamento
        q_traj_sampled = trajs[sample_indices]

        q_traj = np.array([[0.35, *q[1:]] for q in q_traj_sampled])  # sempre (N, 8), al posto del tempo torso costante

        # cinematica diretta per ogni configurazione
        trajectory_from_dataset = [self.robot.fkine(q).A for q in q_traj]  # ogni punto è lista 4x4 SE3 in numpy (senza .A in SE3)

        # Conversione in pos+quat per DMP
        Y = ptr.pqs_from_transforms(np.array(trajectory_from_dataset))  # shape (N, 7)

        T = np.linspace(0, execution_time, n_steps)
        dmp = CartesianDMP(execution_time=execution_time, dt=dt, n_weights_per_dim=25, smooth_scaling=True)
        dmp.imitate(T, Y) 
        _, Y_dmp = dmp.open_loop() 
        trajectory = ptr.transforms_from_pqs(Y_dmp)  # (n_steps x 4 x 4) pose con matrici omogenee

        R = self._quat2R(pose.orientation)
        t = np.array([pose.position.x, pose.position.y, pose.position.z + 0.15]) 
        T_target = np.eye(4)
        T_target[:3, :3] = R
        T_target[:3, 3] = t

        # Calcolo FK della configurazione attuale
        q0_full = np.r_[self.curr_t, self.curr_q]
        T0 = self.robot.fkine(q0_full).A

        # Conversione in pos + quaternion
        start_pq = ptr.pqs_from_transforms(np.array([T0]))[0]

        new_goal_pose = T_target

        new_goal_pq = ptr.pqs_from_transforms(new_goal_pose) 

        dmp.configure(start_y=start_pq, goal_y=new_goal_pq)
        _, Y_dmp_generalized = dmp.open_loop()

        trajectory_generalized = ptr.transforms_from_pqs(Y_dmp_generalized)

        n_total = len(trajectory_generalized)
        n_reduced = 6  # punti da tenere
        indices = np.linspace(0, n_total - 1, n_reduced, dtype=int)
        trajectory_reduced = [trajectory_generalized[i] for i in indices]

        joint_trajectory = []
        q_curr = q0_full
        c=0
        for T_d in trajectory_reduced:
            T_se3 = SE3(T_d)
            sol = self.robot.ikine_NR(T_se3, q0=q_curr, pinv=True)
            if sol is not None:
                q_curr = sol.q
            if not sol.success:
                c=c+1
            joint_trajectory.append(q_curr)
        self.get_logger().info(f'{c} IK fallite')    
        joint_trajectory = np.array(joint_trajectory)

        positions = np.array([T_d[:3, 3] for T_d in trajectory])
        positions_new = np.array([T[:3, 3] for T in trajectory_reduced])

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Traiettoria originale')
        ax.plot(positions_new[:, 0], positions_new[:, 1], positions_new[:, 2], label='Nuova Traiettoria (DMP)', linestyle='--')
        ax.scatter(positions_new[-1, 0], positions_new[-1, 1], positions_new[-1, 2], color='orange', label='Nuovo goal')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()
        plt.title('Generalizzazione DMP')
        plt.show()

        # Invio traiettoria
        return self._send_trajectory(joint_trajectory)

        





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
    
    def _take(self, model_name_2: str, link_name_2: str) -> bool:
        '''self.get_logger().info(f"[DEBUG][_take] Chiamo /attach per attaccare {model_name_2}:{link_name_2}")
        # prepara la request
        req = Attach.Request(
            model_name_1='tiago',
            link_name_1='gripper_left_finger_link',
            model_name_2=model_name_2,
            link_name_2=link_name_2
        )
        # chiama il servizio
        if not self.cli_attach.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('[DEBUG][_take] Servizio /attach non disponibile')
            return False
        fut = self.cli_attach.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        if fut.result() is None:
            self.get_logger().error('[DEBUG][_take] Attach fallito (risposta nulla)')
            return False
        self.get_logger().info('[DEBUG][_take] Oggetto attaccato con successo')'''
        command = (
        'ros2 service call /attach gazebo_ros_link_attacher/srv/Attach '
        '"{model_name_1: \'tiago\', link_name_1: \'gripper_left_finger_link\', '
        'model_name_2: \'cocacola\', link_name_2: \'link\'}"'
        )

        result = subprocess.run(command, shell=True, capture_output=True, text=True)


        if result.returncode == 0:
            print("Servizio chiamato con successo:")
            print(result.stdout)
        else:
            print("Errore nella chiamata al servizio:")
            print(result.stderr)

        return True

    def _put(self, model_name_2: str, link_name_2: str) -> bool:
        self.get_logger().info(f"[DEBUG][_put] Chiamo /detach per staccare {model_name_2}:{link_name_2}")
        req = Attach.Request(
            model_name_1='tiago',
            link_name_1='gripper_left_finger_link',
            model_name_2=model_name_2,
            link_name_2=link_name_2
        )
        if not self.cli_detach.wait_for_service(timeout_sec=2.0):
            self.get_logger().error('[DEBUG][_put] Servizio /detach non disponibile')
            return False
        fut = self.cli_detach.call_async(req)
        rclpy.spin_until_future_complete(self, fut)
        if fut.result() is None:
            self.get_logger().error('[DEBUG][_put] Detach fallito (risposta nulla)')
            return False
        self.get_logger().info('[DEBUG][_put] Oggetto staccato con successo')
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
    node = Planner_DMP()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()