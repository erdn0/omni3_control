from setuptools import find_packages, setup

package_name = 'omni3_control'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='robocupmsl',
    maintainer_email='tasocak131@gmail.com',
    description='3-wheel omni robot kinematics and control',
    license='MIT',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'kinematics_node = omni3_control.kinematics_node:main',
            'move_1m_node    = omni3_control.move_1m_node:main',
            'go_stop_node    = omni3_control.go_stop_node:main',
            'go_stop_fb_node = omni3_control.go_stop_fb_node:main',
        ],
    },
)
