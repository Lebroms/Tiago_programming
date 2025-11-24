import rclpy
import time
from rclpy.node import Node
from rclpy.action import ActionServer, GoalResponse, CancelResponse
from rclpy.action.server import ServerGoalHandle
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from personal_interfaces.action import MoveHead

class MoveHeadServerNode(Node):
    def __init__(self):
        super().__init__("move_head_server")
        # Creazione del server di azione
        self.move_camera_server_ = ActionServer(
            self,
            MoveHead,
            "move_head",
            goal_callback=self.goal_callback,
            cancel_callback=self.cancel_callback,
            execute_callback=self.execute_callback)
        
        self.get_logger().info("Il server dell'action è stato avviato.")
        # Creazione del publisher per comandare la testa
        self.joint_pub = self.create_publisher(JointTrajectory, '/head_controller/joint_trajectory', 10)

    # Funzione per accettare o rifiutare il goal sulla base dei valori di ingresso dal client
    def goal_callback(self, goal_request: MoveHead.Goal):
        self.get_logger().info("Ricevuto un goal")
        if goal_request.left_limit < -0.217 or goal_request.right_limit > 0.217:
            self.get_logger().warn("Rifiutato: limiti fuori range")
            return GoalResponse.REJECT
        self.get_logger().info("Goal accettato")
        return GoalResponse.ACCEPT
    
    # Funzione per gestire la richiesta di cancellazione del goal
    def cancel_callback(self, goal_handle: ServerGoalHandle):
        self.get_logger().info("Ricevuta una richiesta di cancellazione")
        return CancelResponse.ACCEPT

    # Funzione principale che esegue il goal richiesto
    def execute_callback(self, goal_handle: ServerGoalHandle):
        
        left = goal_handle.request.left_limit
        right = goal_handle.request.right_limit
        step = goal_handle.request.step_angle

        feedback_msg = MoveHead.Feedback()

        angle = left
        # Ciclo per far muovere la testa da sinistra a destra
        while angle <= right:
            traj = JointTrajectory()
            traj.joint_names = ['head_1_joint', 'head_2_joint']
            point = JointTrajectoryPoint()
            point.positions = [angle, -0.57]
            point.time_from_start = rclpy.duration.Duration(seconds=1.0).to_msg()
            traj.points.append(point)
            self.joint_pub.publish(traj)

            # Pubblicazione del feedback corrente
            feedback_msg.current_angle = angle
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Muovo a: {angle:.3f} rad')

            # Controlla se il goal è stato cancellato
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().warn('Goal cancellato!')
                return MoveHead.Result(success=False)

            angle += step
            time.sleep(1.0)  # attesa tra un movimento e l'altro

        '''# Torna alla posizione iniziale (left_limit)
        traj = JointTrajectory()
        traj.joint_names = ['head_1_joint', 'head_2_joint']
        point = JointTrajectoryPoint()
        point.positions = [left, -0.57]  # ritorno alla posizione iniziale
        point.time_from_start = rclpy.duration.Duration(seconds=2.0).to_msg()
        traj.points.append(point)
        self.joint_pub.publish(traj)

        self.get_logger().info(f'Ritorno alla posizione iniziale: {left:.3f} rad')
        time.sleep(2.0)  # attesa per completare il movimento
        '''

        # Segnala il completamento del goal
        goal_handle.succeed()
        result = MoveHead.Result()
        result.success = True
        self.get_logger().info('Scansione completata con successo.')
        return result 

def main(args=None):
    rclpy.init(args=args)
    node = MoveHeadServerNode()
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()