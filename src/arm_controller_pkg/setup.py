from setuptools import find_packages, setup

package_name = 'arm_controller_pkg'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='scr',
    maintainer_email='scr@todo.todo',
    description='ARM controller package',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'cargo_manager_node = arm_controller_pkg.cargo_manager_node:main',
            'gripper_node = arm_controller_pkg.gripper_node:main',
            'load_node1 = arm_controller_pkg.load_node1:main',
            'load_node2 = arm_controller_pkg.load_node2:main',
            'load_node3 = arm_controller_pkg.load_node3:main',
            'load_node4 = arm_controller_pkg.load_node4:main',
            'load_node5 = arm_controller_pkg.load_node5:main',
            'load_node6 = arm_controller_pkg.load_node6:main',
            'amr_robot_node = arm_controller_pkg.amr_robot_node:main',
            'load_node_timing_log = arm_controller_pkg.load_node_timing_log:main',
            'unload_node = arm_controller_pkg.unload_node:main',
            'manual_command_node = arm_controller_pkg.manual_command_node:main',
        ],
    },
)
