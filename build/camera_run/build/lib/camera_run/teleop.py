#!/usr/bin/env python3

import rclpy
from ackermann_msgs.msg import AckermannDrive
from rclpy.node import Node

import getch

class TeleopNode:
    def __init__(self):
        self.node = rclpy.create_node('teleop_node')
        self.publisher = self.node.create_publisher(AckermannDrive, '/ackermann_cmd', 10)
        self.ackermann_msg = AckermannDrive()
        self.node.get_logger().info("please press the control keys")

    def teleop_control(self):
        speed = 0.0
        steering_angle = 0.0
        STEERING_BIAS = 0.0
        direction = 'left'
        
        while rclpy.ok():
            if direction == 'left':
                speed = -2.0
                steering_angle = 2.0
            elif direction == 'right':
                speed = 0.4
                steering_angle = -0.4
            else:
                speed = 0.8
                steering_angle = 0.0
        

            self.ackermann_msg.speed = speed
            self.ackermann_msg.steering_angle = steering_angle + STEERING_BIAS

            self.publisher.publish(self.ackermann_msg)
            self.node.get_logger().info(f"Speed: {speed}, Steering Angle: {steering_angle}")

            #rclpy.spin_once(self.node)

    
def main(args=None):
    rclpy.init(args=args)
    teleop_node = TeleopNode()
    try:
        teleop_node.teleop_control()
    except KeyboardInterrupt:
        pass

    #teleop_node.node.destroy_node()
    #rclpy.shutdown()
    rclpy.spin(teleop_node)

    

if __name__ == '__main__':
    main()