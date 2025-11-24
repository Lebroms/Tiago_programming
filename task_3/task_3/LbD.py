
from spatialmath import SE3
from roboticstoolbox import DHRobot, RevoluteDH
from movement_primitives.dmp import CartesianDMP
import pytransform3d.trajectories as ptr
import matplotlib.pyplot as plt
import time
from mpl_toolkits.mplot3d import Axes3D  
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import roboticstoolbox as rtb
import os
import numpy as np
from scipy.spatial.transform import Rotation as R
from mpl_toolkits.mplot3d import Axes3D


# === Robot ===

robot = rtb.ERobot.URDF('/home/lebroms/tiago_public_ws/src/task_1/task_1/urdf/tiago_robot.urdf')
print(robot)

dmp_dataset_dir = "/home/lebroms/Downloads/tiago_traiettorie_SG/2a-2b/place_2b_rep1.txt"

trajs = np.loadtxt(dmp_dataset_dir, skiprows=1)  # trajs è (N, 8) [tempo + 7 giunti]

q0 = np.array([0.35, 0.0701089324886101, 0.0922960064981821, -3.099911325862799, 1.3626732529722807, 2.050055466436567, -0.5200183123913877, 1.1400032090925691])

# Estrai l'ultima riga (l'ultima configurazione della traiettoria)
last_row = trajs[-1]  # shape (8,)

# Costruisci qf: 0.35 (torso) + ultimi 7 elementi (giunti)
qf = np.r_[0.35, last_row[1:]]  # shape (8,)


# Configurazione iniziale e finale dell'end-effector

# === Parametri ===
N = len(trajs)
n_samples = 300

# Indici distribuiti uniformemente
sample_indices = np.linspace(0, N - 1, n_samples, dtype=int)

# Applica campionamento
q_traj_sampled = trajs[sample_indices]



n_steps = 300
dt = 0.01
execution_time = (n_steps - 1) * dt

   

# === Cinematica Diretta: T_start e T_end ===
T_start = robot.fkine(q0)  # matrice 4x4 (SE3)
T_end = robot.fkine(qf) # matrice 4x4 (SE3)
T_end_numpy = robot.fkine(qf).A # matrice 4x4 (numpy)
T_start_numpy = robot.fkine(q0).A # matrice 4x4 (numpy)

q_traj = np.array([[0.35, *q[1:]] for q in q_traj_sampled])  # shape (N, 8)

# --- Calcola la cinematica diretta per ogni configurazione
trajectory_from_dataset = [robot.fkine(q).A for q in q_traj]  # lista di 4x4 SE3 in numpy

# --- Conversione in pos+quat per DMP
Y = ptr.pqs_from_transforms(np.array(trajectory_from_dataset))  # shape (N, 7)

# Conversione in pos+quaternion per DMP

T = np.linspace(0, execution_time, n_steps)
dmp = CartesianDMP(execution_time=execution_time, dt=dt, n_weights_per_dim=25, smooth_scaling=True)
dmp.imitate(T, Y) # apprendere la traiettoria: T= time for each step; Y= state at each step
_, Y_dmp = dmp.open_loop() # output DMP --> n_steps x 7


trajectory = ptr.transforms_from_pqs(Y_dmp)  # (n_steps x 4 x 4) --> pose rappresentate da matrici omogenee

# Plot
positions = np.array([T_d[:3, 3] for T_d in trajectory])

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Traiettoria end-effector')
ax.scatter(positions[0, 0], positions[0, 1], positions[0, 2], color='green', label='Start')
ax.scatter(positions[-1, 0], positions[-1, 1], positions[-1, 2], color='red', label='End')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

"""Verifica che la DMP si "adatta" anche a nuovo Goal"""
# === Nuovo goal ===

# Posa target che vuoi usare come nuovo goal
target_position = np.array([0.6727480435895852, 0.29370424859305505, 0.788481741739837])
target_orientation_quat = np.array([0.9539926102817636, -0.12183213839777697,
                                    0.10212065648377229, 0.25421723210776953])

# Scipy usa ordine (w, x, y, z), quindi serve riordinare
r = R.from_quat([target_orientation_quat[0],  # x
                 target_orientation_quat[1],  # y
                 target_orientation_quat[2],  # z
                 target_orientation_quat[3]]) # w
R_matrix = r.as_matrix()  # 3x3

# Costruisci SE3
T_goal_new = np.eye(4)
T_goal_new[:3, :3] = R_matrix
T_goal_new[:3, 3] = target_position

new_goal_pose = T_goal_new

new_goal_pq = ptr.pqs_from_transforms(new_goal_pose) # pos + quat
start_pq = ptr.pqs_from_transforms(T_start_numpy)
dmp.configure(start_y=start_pq, goal_y=new_goal_pq)
_, Y_dmp_generalized = dmp.open_loop()

'''# 1. Normalizzazione
norms = np.linalg.norm(Y_dmp_generalized[:, 3:], axis=1, keepdims=True)
Y_dmp_generalized[:, 3:] /= norms

# 2. Correzione di continuità
for i in range(1, len(Y_dmp_generalized)):
    if np.dot(Y_dmp_generalized[i, 3:], Y_dmp_generalized[i - 1, 3:]) < 0:
        Y_dmp_generalized[i, 3:] *= -1'''


trajectory_generalized = ptr.transforms_from_pqs(Y_dmp_generalized)

n_total = len(trajectory_generalized)
n_reduced = 100  # punti da tenere
indices = np.linspace(0, n_total - 1, n_reduced, dtype=int)
trajectory_reduced = [trajectory_generalized[i] for i in indices]

joint_trajectory = []
q_curr = q0
for T_d in trajectory_reduced:
    T_se3 = SE3(T_d)
    sol = robot.ikine_NR(T_se3, q0=q_curr, pinv=True)
    if sol is not None:
        q_curr = sol.q
    if not sol.success:
        print("IK fallita per passo!")
    joint_trajectory.append(q_curr)

joint_trajectory = np.array(joint_trajectory)

'''n_keep = 300
indices = np.linspace(0, len(joint_trajectory) - 1, n_keep, dtype=int)
joint_trajectory = joint_trajectory[indices]'''

# Plot per vedere quano si discostano le traiettorie
positions_new = np.array([T[:3, 3] for T in trajectory_generalized])

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




# Estrai posizioni delle traiettorie
positions = np.array([T_d[:3, 3] for T_d in trajectory])  # originale
positions_new = np.array([T[:3, 3] for T in trajectory_reduced])  # generalizzata

# Plot 3D delle traiettorie
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(positions[:, 0], positions[:, 1], positions[:, 2], label='Traiettoria originale')
ax.plot(positions_new[:, 0], positions_new[:, 1], positions_new[:, 2],
        label='Nuova Traiettoria (DMP)', linestyle='--')
ax.scatter(positions_new[-1, 0], positions_new[-1, 1], positions_new[-1, 2],
           color='orange', label='Nuovo goal')

# Aggiunta di terne (assii X,Y,Z) sull'end-effector lungo la traiettoria generalizzata
skip = 1  # ogni quanti step disegnare la terna
scale = 0.05  # lunghezza delle frecce

'''for i in range(0, len(trajectory), skip):
    T = trajectory[i]
    pos = T[:3, 3]
    R = T[:3, :3]
    # Assi locali
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]
    # Disegna i tre assi con colore (r,g,b)
    ax.quiver(*pos, *x_axis, length=scale, color='r')
    ax.quiver(*pos, *y_axis, length=scale, color='g')
    ax.quiver(*pos, *z_axis, length=scale, color='b')

# Etichette e legenda
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('Generalizzazione DMP con Orientamento End-Effector')
plt.show()'''


for i in range(0, len(trajectory_reduced), skip):
    T = trajectory_reduced[i]
    pos = T[:3, 3]
    R = T[:3, :3]
    # Assi locali
    x_axis = R[:, 0]
    y_axis = R[:, 1]
    z_axis = R[:, 2]
    # Disegna i tre assi con colore (r,g,b)
    ax.quiver(*pos, *x_axis, length=scale, color='r')
    ax.quiver(*pos, *y_axis, length=scale, color='g')
    ax.quiver(*pos, *z_axis, length=scale, color='b')

# Etichette e legenda
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.title('Generalizzazione DMP')
plt.show()


def get_joint_positions(robot, q):
    """
    Restituisce la lista di posizioni 3D (x,y,z) di tutti i giunti (base + link ends)
    """
    positions = [np.array([0, 0, 0])]  # base fixed at origin
    T = SE3()  # identità

    for i, angle in enumerate(q):
        T = T * robot.links[i].A(angle)  # trasformazione giunto i-esimo
        positions.append(T.t)  # estrai traslazione (posizione)
    return np.array(positions)  # shape: (n_links+1, 3)

class GazeboTrajectoryPublisher(Node):
    def __init__(self, joint_trajectory):
        super().__init__('gazebo_trajectory_publisher')
        self.joint_trajectory = joint_trajectory
        self.index = 0

        # ROS Publisher
        self.publisher_ = self.create_publisher(Float64MultiArray, '/arm_controller/commands', 10)
        self.timer = self.create_timer(0.1, self.timer_callback)  # 10 Hz

        # Setup matplotlib 3D figure
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.ax.set_xlim([-0.5, 0.5])
        self.ax.set_ylim([-0.5, 0.5])
        self.ax.set_zlim([0, 0.6])
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')

        # Linea traiettoria end-effector
        self.traj_line, = self.ax.plot([], [], [], 'r-', label='Traiettoria EE')

        # Linee robot (link)
        self.robot_lines = []
        n_links = len(joint_trajectory[0])  # numero joint
        for _ in range(n_links):
            line, = self.ax.plot([], [], [], 'b-', linewidth=3)
            self.robot_lines.append(line)

        # Dati accumulati per traiettoria
        self.ee_x = []
        self.ee_y = []
        self.ee_z = []

        plt.legend()
        plt.ion()
        plt.show()

    def timer_callback(self):
        if self.index >= len(self.joint_trajectory):
            self.get_logger().info('Traiettoria completata.')
            self.destroy_timer(self.timer)
            plt.ioff()
            plt.show()
            return

        q = self.joint_trajectory[self.index]
        msg = Float64MultiArray()
        msg.data = q.tolist()
        self.publisher_.publish(msg)
        self.get_logger().info(f'Pubblicato: {msg.data}')

        # Ottieni posizione giunti robot
        joint_positions = get_joint_positions(robot, q)  # (n_links+1, 3)

        # Aggiorna linee robot: ogni linea collega joint i con joint i+1
        for i, line in enumerate(self.robot_lines):
            xs = [joint_positions[i][0], joint_positions[i+1][0]]
            ys = [joint_positions[i][1], joint_positions[i+1][1]]
            zs = [joint_positions[i][2], joint_positions[i+1][2]]
            line.set_data(xs, ys)
            line.set_3d_properties(zs)

        # Aggiungi nuovo punto end-effector (ultimo joint)
        ee_pos = joint_positions[-1]
        self.ee_x.append(ee_pos[0])
        self.ee_y.append(ee_pos[1])
        self.ee_z.append(ee_pos[2])

        self.traj_line.set_data(self.ee_x, self.ee_y)
        self.traj_line.set_3d_properties(self.ee_z)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        self.index += 1



def main():
    rclpy.init()
    node = GazeboTrajectoryPublisher(joint_trajectory)
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()

