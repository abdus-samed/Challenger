import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from matplotlib import image as mpimg
from std_msgs.msg import Float64

import getch
from ackermann_msgs.msg import AckermannDrive


class SlopeNode:
    def __init__(self):
        #super().__init__("slope_subscriber")
        self.node = rclpy.create_node('slope_node')
        #self.subscription = self.create_subscription(Float64, "slope", self.callback, 10)
        self.publisher = self.node.create_publisher(AckermannDrive,  '/ackermann_cmd',1)        
        #self.ackermann_pub = self.node.create_publisher(AckermannDrive, '/ackermann_cmd', 2)
        self.ackermann_msg = AckermannDrive()

    def follower(self):
        speed = 0.0
        steering_angle = 0.0
        STEERING_BIAS = -0.021

        while rclpy.ok():
            speed = 0.4
            steering_angle = 0.4

        

            self.ackermann_msg.speed = speed
            self.ackermann_msg.steering_angle = steering_angle + STEERING_BIAS

            self.publisher.publish(self.ackermann_msg)
            self.node.get_logger().info(f"Speed: {speed}, Steering Angle: {steering_angle}")

            #rclpy.spin_once(self.node)
        
       
def main(args = None):
    rclpy.init(args = args)
    slope_node = SlopeNode()
    rclpy.spin(slope_node)
    slope_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main":
    main()