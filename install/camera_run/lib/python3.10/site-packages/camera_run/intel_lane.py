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
import pyrealsense2 as rs
from sensor_msgs.msg import Image

class IntelLane(Node):
    def __init__(self):
        super().__init__("intel_lane")
        
        timer_period = 0.05
        self.timer = self.create_timer(timer_period,self.timer_callback)
        
        self.pipe = rs.pipeline() 
        self.cfg = rs.config()
        self.cfg.enable_stream(rs.stream.color,640,480,rs.format.bgr8,30)
        self.pipe.start(self.cfg)
        self.br_rgb = CvBridge()
               
    def timer_callback(self):
        frames = self.pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        frame = np.asanyarray(color_frame.get_data())
        
        #image = self.br_rgb.imgmsg_to_cv2(frame)
        
        cv2.imshow(frame)
        
        
      
        
        

def main(args = None):
    rclpy.init(args = args)
    intel_lane = IntelLane()
    rclpy.spin(intel_lane)
    intel_lane.destroy_node()
    rclpy.shutdown()


if __name__ == "__main":
    main()