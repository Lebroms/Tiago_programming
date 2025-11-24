import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.action.client import ClientGoalHandle, GoalStatus
from personal_interfaces.action import MoveHead

class MoveHeadClientNode(Node):
    def __init__(self):
        super().__init__("move_head_client")
        self.client_ = ActionClient(self, MoveHead, "move_head")
    
    # Attende il server dell'action, invia un goal e registra una callback per la risposta
    # Registra anche una callback opzionale per il feedback
    def send_goal(self, left_limit, right_limit, step_angle):
        self.client_.wait_for_server()
        goal = MoveHead.Goal()
        goal.left_limit = left_limit
        goal.right_limit = right_limit
        goal.step_angle = step_angle
        self.client_.send_goal_async(
            goal, feedback_callback=self.goal_feedback_callback). \
            add_done_callback(self.goal_response_callback)

    # Invia una richiesta di cancellazione del goal
    def cancel_goal(self):
        self.get_logger().info("Invio richiesta di cancellazione del goal")
        self.goal_handle_.cancel_goal_async()

    # Riceve la risposta al goal e, se accettato, registra una callback per il risultato
    def goal_response_callback(self, future):
        self.goal_handle_ = future.result()
        if self.goal_handle_.accepted:
            self.get_logger().info("Goal accettato")
            self.goal_handle_.get_result_async().add_done_callback(self.goal_result_callback)
        else:
            self.get_logger().info("Goal rifiutato")

    # Riceve il risultato del goal e stampa lo stato
    def goal_result_callback(self, future):
        status = future.result().status
        result = future.result().result
        if status == GoalStatus.STATUS_SUCCEEDED:
            self.get_logger().info("Completato con successo")
        elif status == GoalStatus.STATUS_ABORTED:
            self.get_logger().error("Annullato dal server")
        elif status == GoalStatus.STATUS_CANCELED:
            self.get_logger().warn("Cancellato dal client")
        self.get_logger().info("Risultato: " + str(result.success))
        rclpy.shutdown()

    # Stampa il feedback ricevuto durante l'esecuzione del goal
    def goal_feedback_callback(self, feedback_msg):
        number = feedback_msg.feedback.current_angle
        self.get_logger().info("Feedback ricevuto: " + str(number))


def main(args=None):
    rclpy.init(args=args)
    node = MoveHeadClientNode()
    node.send_goal(-0.217, 0.217, 0.1)
    rclpy.spin(node)
    rclpy.shutdown()


if __name__ == "__main__":
    main()