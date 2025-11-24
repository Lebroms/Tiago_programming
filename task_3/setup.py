from setuptools import find_packages, setup

package_name = 'task_1'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    include_package_data=True,  # <--- Aggiunto
    package_data={
        'task_1': ['urdf/*.urdf']  # <--- Aggiunto
    },
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='lebroms',
    maintainer_email='lebroms@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'action_server = task_1.action_server:main',
            'action_client = task_1.action_client:main',
            'aruco_handler = task_1.aruco_handler:main',
            'grasp_publisher = task_1.grasp_publisher:main',
            'moveit_pose_commander = task_1.moveit_pose_commander:main',
            'state_machine = task_1.state_machine:main',
            'planner_ik = task_1.planner_ik:main',
            'state_DMP = task_1.state_DMP:main',
            'planner_DMP = task_1.planner_DMP:main'


        ],
    },
)
