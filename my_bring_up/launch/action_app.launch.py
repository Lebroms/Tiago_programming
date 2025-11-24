#!/usr/bin/env python3

import launch
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import ExecuteProcess, RegisterEventHandler
from launch.event_handlers import OnProcessExit

def generate_launch_description():
    ld = LaunchDescription()

    # 1) task_1: head scan
    action_server = Node(
        package='task_1', executable='action_server',
        name='action_server', output='screen'
    )
    action_client = Node(
        package='task_1', executable='action_client',
        name='action_client', output='screen'
    )
    aruco_detector = Node(
        package='task_1', executable='aruco_handler',
        name='aruco_detector', output='screen'
    )
    # 1b) grasp_publisher parte insieme ad aruco_detector
    grasp_pub = Node(
        package='task_1', executable='grasp_publisher',
        name='grasp_publisher', output='screen'
    )

    ld.add_action(action_server)
    ld.add_action(action_client)
    ld.add_action(aruco_detector)
    ld.add_action(grasp_pub)  # <--- lanciato in parallelo ad aruco_detector

    # 2) torso lift (subito dopo action_client)
    send_torso = ExecuteProcess(
        cmd=[
            'ros2','action','send_goal','/torso_controller/follow_joint_trajectory',
            'control_msgs/action/FollowJointTrajectory',
            '{"trajectory":{"joint_names":["torso_lift_joint"],'
            '"points":[{"positions":[0.35],"time_from_start":{"sec":3,"nanosec":0}}]}}',
            '--feedback'
        ], output='screen'
    )
    ld.add_action(RegisterEventHandler(
        OnProcessExit(
            target_action=action_client,
            on_exit=[send_torso]
        )
    ))

    # 3) arm to home (subito dopo send_torso)
    send_arm = ExecuteProcess(
        cmd=[
            'ros2','action','send_goal','/arm_controller/follow_joint_trajectory',
            'control_msgs/action/FollowJointTrajectory',
            '{"trajectory":{"joint_names":["arm_1_joint","arm_2_joint","arm_3_joint","arm_4_joint","arm_5_joint","arm_6_joint","arm_7_joint"],'
            '"points":[{"positions":[0.07,0.1,-3.1,1.36,2.05,0.01,-0.05],"time_from_start":{"sec":3,"nanosec":0}}]}}',
            '--feedback'
        ], output='screen'
    )
    ld.add_action(RegisterEventHandler(
        OnProcessExit(
            target_action=send_torso,
            on_exit=[send_arm]
        )
    ))

    # 4) trigger StateMachine: al termine di send_arm pubblico su /start_sm
    publish_start = ExecuteProcess(
        cmd=[
            'ros2','topic','pub','/start_sm','std_msgs/msg/Empty','{}','--once'
        ],
        output='screen'
    )
    ld.add_action(RegisterEventHandler(
        OnProcessExit(
            target_action=send_arm,
            on_exit=[publish_start]
        )
    ))

    '''# 5) planner_node e state_machine partono al trigger /start_sm
    planner = Node(
        package='task_2', executable='planner_node',
        name='planner_node', output='screen'
    )
    state_machine = Node(
        package='task_2', executable='state_machine',
        name='state_machine', output='screen'
    )
    
    ld.add_action(RegisterEventHandler(
        OnProcessExit(
            target_action=publish_start,
            on_exit=[planner, state_machine]
        )
    ))'''

    return ld