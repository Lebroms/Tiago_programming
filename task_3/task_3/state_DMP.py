
#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from personal_interfaces.action import PlanAndExecute
from rclpy.action import ActionClient
from std_msgs.msg import Empty
from geometry_msgs.msg import PoseStamped
from threading import Timer
import time

class StateMachine_DMP(Node):
    def __init__(self):
        super().__init__('state_machine_DMP')
        self.grasp_poses = {}
        for mid in [1, 2, 3, 4]:
            self.create_subscription(
                PoseStamped, f'/grasp_pose_{mid}',
                lambda msg, i=mid: self._cb(i, msg), 10
            )

        self._client = ActionClient(self, PlanAndExecute, 'planning_action')
        self._sub_start = self.create_subscription(
            Empty, '/start_sm', self._on_start, 1
        )
        self.get_logger().info('StateMachine pronta: in ascolto su /start_sm')

        self.sequence = [
            {'task_type': 'open_gripper'},
            {'task_type': 'move', 'marker_id': 2},
            {'task_type': 'place'},
            {'task_type': 'close_gripper'},
            
            {'task_type': 'lift'},
            {'task_type': 'move', 'marker_id': 4},
            {'task_type': 'place'},
            {'task_type': 'open_gripper'},
            
            {'task_type': 'lift'},
            {'task_type': 'back_to_home'},
            {'task_type': 'move', 'marker_id': 1},
            {'task_type': 'place'},
            {'task_type': 'close_gripper'},
            
            {'task_type': 'lift'},
            {'task_type': 'move', 'marker_id': 3},
            {'task_type': 'place'},
            {'task_type': 'open_gripper'},
            
        ]
        self.current = 0

    def _cb(self, i, msg):
        self.grasp_poses[i] = msg
        p = msg.pose.position
        self.get_logger().info(
            f'[DBG] grasp_pose_{i}: x={p.x:.3f} y={p.y:.3f} z={p.z:.3f}'
        )

    def _on_start(self, msg):
        self.get_logger().info('Ricevuto trigger start, avvio sequenza')
        self.destroy_subscription(self._sub_start)
        self._send_next()

    def _send_next(self):
        if self.current >= len(self.sequence):
            self.get_logger().info('Tutti i task completati.')
            return

        entry = self.sequence[self.current]
        task_type = entry['task_type']
        goal = PlanAndExecute.Goal(task_type=task_type)

        if entry['task_type'] in ['take', 'put']:
            # copia i parametri dal dizionario nel goal
            goal.model_name_2 = entry['model_name_2']
            goal.link_name_2  = entry['link_name_2']

        if task_type == 'move':
            mid = entry.get('marker_id')
            if mid and mid not in self.grasp_poses:
                self.get_logger().warn(f'Manca grasp_pose_{mid}, riprovo tra 0.1s')
                Timer(0.1, self._send_next).start()
                return
            goal.target_pose = self.grasp_poses[mid]
            goal.marker_id = mid

        self.get_logger().info(f'Invio #{self.current}: {task_type}')
        if not self._client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn('Planner non pronto, riprovo')
            Timer(0.5, self._send_next).start()
            return

        send = self._client.send_goal_async(goal, feedback_callback=self._fb)
        send.add_done_callback(self._resp)

    def _fb(self, fb):
        self.get_logger().info(f'[feedback] {fb.feedback.status}')

    def _resp(self, fut):
        gh = fut.result()
        if not gh.accepted:
            self.get_logger().error('Goal rifiutato')
            return
        gh.get_result_async().add_done_callback(self._done)

    def _done(self, fut):
        res = fut.result().result
        self.get_logger().info(f'[result] success={res.success}')
        self.current += 1
        self._send_next()


def main():
    rclpy.init()
    node = StateMachine_DMP()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()
