import cv2 as cv2
import numpy as np
import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import math
import os
from cv_bridge import CvBridge
from ackermann_msgs.msg import AckermannDrive




class Steering(Node):
    def __init__(self):
        #self.node = rclpy.create_node('detection_node')
        super().__init__('steering_node')
        self.steering_publisher = self.create_publisher(AckermannDrive, '/ackermann_cmd',10)
        self.ackermann_msg = AckermannDrive()
        timer_period = 0.5
        self.timer = self.create_timer(timer_period, self.frame_callback)
        
    
        
    def frame_callback(self):
        
        directionDev = 'left'
        speed = 0.0
        steering_angle = 0.0
        STEERING_BIAS = -0.021
        
        if directionDev == 'left':
            speed = -2.5
            steering_angle = 0.8
            
        else:
            speed = 0.5
            steering_angle = 0.4
            
        self.ackermann_msg.speed = speed
        self.ackermann_msg.steering_angle = steering_angle + STEERING_BIAS

        self.steering_publisher.publish(self.ackermann_msg)
        self.get_logger().info(f"Speed: {speed}, Steering Angle: {steering_angle}")
        self.get_logger().info(f"Direction: {directionDev}")
        
        self.get_logger().warning("Sending IMG frame")
      
        
        

def main(args = None):
    rclpy.init(args = args)
    steering_node = Steering()
    rclpy.spin(steering_node)
    steering_node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main":
    main()