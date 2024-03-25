from setuptools import find_packages, setup

package_name = 'camera_run'

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
    maintainer='samet',
    maintainer_email='sametd@hotmail.de',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "intel_pub = camera_run.intel_pub:main",
            "intel_sub = camera_run.intel_sub:main",
            "intel_lane = camera_run.intel_lane:main",
            "intel_lane_sub = camera_run.intel_lane_sub:main",
            'slope_sub = camera_run.slope_sub:main',
            'bot = camera_run.bot:main',
            'ackermann_to_vesc = camera_run.ackermann_to_vesc:main',
            'slope_node = camera_run.slope_node:main',
            'detection_node = camera_run.LaneDetection:main',
            'steering = camera_run.steering:main',
            'follower = camera_run.lane_follower:main',
            'detection = camera_run.2_new_line:main',
            'tele = camera_run.teleop:main'
            
        ],
    },
)
