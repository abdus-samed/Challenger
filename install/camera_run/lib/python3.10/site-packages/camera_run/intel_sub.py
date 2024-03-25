import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from matplotlib import pyplot as plt
import math
from matplotlib import image as mpimg


class IntelSubscriber(Node):
    def __init__(self):
        super().__init__("intel_subscriber")
        self.subscription_rgb = self.create_subscription(Image, "rgb_frame", self.rgb_frame_callback, 10)
        self.br_rgb = CvBridge()

    def rgb_frame_callback(self, data):
        self.get_logger().warning("Receiving RGB frame")
        current_frame = self.br_rgb.imgmsg_to_cv2(data)
        
       
        
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        img2 = cv2.medianBlur(gray,5)
        cannyed = cv2.Canny(gray, 200,300)
        
       
        
        thresh2 = cv2.adaptiveThreshold(img2,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
                cv2.THRESH_BINARY,11,2)
        
               
        
        cv2.imshow("gaussian",thresh2)
        cv2.imshow("RGB", current_frame)
        cv2.imshow('Cropped', cannyed )
        cv2.waitKey(1)




def main(args = None):
    rclpy.init(args = args)
    intel_subscriber = IntelSubscriber()
    rclpy.spin(intel_subscriber)
    intel_subscriber.destroy_node()
    rclpy.shutdown()


if __name__ == "__main":
    main()