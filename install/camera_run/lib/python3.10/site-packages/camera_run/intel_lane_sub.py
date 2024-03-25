import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from matplotlib import pyplot as plt ,cm , colors
import math
from matplotlib import image as mpimg
import os
from scipy import optimize
from std_msgs.msg import Int32MultiArray
from std_msgs.msg import Float64MultiArray
from std_msgs.msg import Float64
from std_msgs.msg import _float64



img_size = (960,540)


height = 540
width = 960
scale = 1

## BEV params
top_roi = 0.4395 # 0.4395 # 0-1 (0 is top)
bottom_roi = 0.6347 #0.6347 # 0-1 (0 is top)
width_use = 1 # 0-1 (For cropping vertically. Not used)
height_multiplier = 3.5 # How much to stretch vertically
skew_level = 0.887 # 0-1 (1 is triangle, 0 is rectangle)

roi_in = np.float32(np.floor([
	((1-width_use)*width*scale,0), # Top-left corner
	((1-width_use)*width*scale,(bottom_roi-top_roi)*height*scale), # Bottom-left corner            
	(width_use*width*scale,(bottom_roi-top_roi)*height*scale), # Bottom-right corner
	(width_use*width*scale,0) # Top-right corner
]))
roi_out = np.float32(np.floor([
	((1-width_use)*width*scale,0), # Top-left corner
	(width*skew_level*0.5*width_use*scale,(bottom_roi-top_roi)*height*height_multiplier*scale), # Bottom-left corner            
	(width*(1-skew_level*0.5*width_use)*scale,(bottom_roi-top_roi)*height*height_multiplier*scale), # Bottom-right corner
	(width_use*width*scale,0) # Top-right corner
 ]))

transform_matrix = cv2.getPerspectiveTransform(roi_in,roi_out)

region_of_interest_vertices = [
    (0, height),
    (width / 2, height / 2),
    (width, height),
]


def region_of_interest(img, vertices):
        # Define a blank matrix that matches the image height/width.
        mask = np.zeros_like(img)    # Retrieve the number of color channels of the image.
    
        match_mask_color = 255
      
        # Fill inside the polygon
        cv2.fillPoly(mask, vertices, match_mask_color)
    
        # Returning the image only where mask pixels match
        masked_image = cv2.bitwise_and(img, mask)
        return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=3):
    # If there are no lines to draw, exit.
    if lines is None:
        return    # Make a copy of the original image.
    img = np.copy(img)   
    # Create a blank image that matches the original in size.
    line_img = np.zeros(
        (
            img.shape[0],
            img.shape[1],
            3
        ),
        dtype=np.uint8,
    )    
        # Loop over all lines and draw them on the blank image.
    for line in lines:
        for x1, y1, x2, y2 in line:
           img = cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)   
                # Merge the image with the lines onto the original.
    #img = cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)    
                # Return the modified image.
    return img





# Perspective points to be warped
src = np.float32([[354, 264],
                    [414, 264],
                    [120, 384],
                    [600, 384]])

# Window to be shown
dst = np.float32([[120, 0],
                    [600, 0],
                    [120, 426],
                    [600, 426]])

# Matrix to warp the image for birdseye window
matrix = cv2.getPerspectiveTransform(src, dst)
# Inverse matrix to unwarp the image for final window
minv = cv2.getPerspectiveTransform(dst, src)


# Get the birdseye window dimensions
#height, width = birdseye.shape[:2]

# Divide the birdseye view into 2 halves to separate left & right lanes
#birdseyeLeft  = birdseye[0:height, 0:width // 2]
#birdseyeRight = birdseye[0:height, width // 2:width]

# Display birdseye view image
# cv2.imshow("Birdseye" , birdseye)
# cv2.imshow("Birdseye Left" , birdseyeLeft)
# cv2.imshow("Birdseye Right", birdseyeRight)



class IntelLaneSubscriber(Node):
    def __init__(self):
        super().__init__("intel_subscriber")
        self.subscription_rgb = self.create_subscription(Image, "lane_topic", self.frame_callback, 10)
        self.br_rgb = CvBridge()
        self.publisher_slope = self.create_publisher(Float64,'slope',10)
    

    def frame_callback(self, data):
        self.get_logger().warning("Receiving IMG frame")
        current_frame = self.br_rgb.imgmsg_to_cv2(data)
        
        gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.medianBlur(gray,5)
        cannyed = cv2.Canny(blur, 200,300)
        cropped_image = region_of_interest(
            cannyed,
            np.array([region_of_interest_vertices], np.int32
            ),
        )
        
        lines = cv2.HoughLinesP(
            cropped_image,
            rho=6,
            theta=np.pi/60,
            threshold=160,
            lines=np.array([]),
            minLineLength = 40,
            maxLineGap = 25
                )
#print(lines)
#line_image = draw_lines(image, lines)    

        left_line_x = []
        left_line_y = []
        right_line_x = []
        right_line_y = []
        slope_ar = []


        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1) # <-- Calculating the slope.
                msg = Float64()
                msg.data = slope
                self.publisher_slope.publish(msg)
                self.get_logger().info("Publishing the Slope")
                #slope_ar.extend(slope)
                if math.fabs(slope) < 0.5: # <-- Only consider extreme slope
                    continue
                if slope <= 0: # <-- If the slope is negative, left group.
                    left_line_x.extend([x1, x2])
                    left_line_y.extend([y1, y2])
                else: # <-- Otherwise, right group.
                    right_line_x.extend([x1, x2])
                    right_line_y.extend([y1, y2])
                    
             # here we publish color_image
        
            
        min_y = cropped_image.shape[0] * (3 / 5) # <-- Just below the horizon
        max_y = cropped_image.shape[0] # <-- The bottom of the image

        poly_left = np.poly1d(np.polyfit(
            left_line_y,
            left_line_x,
            deg=1
        ))

        left_x_start = int(poly_left(max_y))
        left_x_end = int(poly_left(min_y))

        poly_right = np.poly1d(np.polyfit(
            right_line_y,
            right_line_x,
            deg=1
        ))

        right_x_start = int(poly_right(max_y))
        right_x_end = int(poly_right(min_y))


        line_image = draw_lines(
            cropped_image,
            [[
                [int(left_x_start), int(max_y), int(left_x_end), int(min_y)],
                [int(right_x_start), int(max_y), int(right_x_end), int(min_y)],
            ]],
            (64,224,208),
            thickness=5
                )
        
        #cropped = img_in = current_frame[int(top_roi*height):int(bottom_roi*height),0:width]
        #cropped_height , cropped_width, _ = cropped.shape
        
        #birdseye = cv2.warpPerspective(line_image, transform_matrix,(cropped_width,int(cropped_height*height_multiplier)),flags=(cv2.INTER_LINEAR))     
        
        #first_horizontal_line = cv2.line(line_image, (int(left_x_start),int(poly_left)), (int(right_x_start),int(poly_right)), color=(0,0,255),thickness = 3)
        #second_horizontal_line =cv2.line(first_horizontal_line, (int(left_line_x[1]),int(y2)), (int(right_line_x[1]),int(y2)), color=(0,0,255),thickness = 3)
        
        canny2 = cv2.Canny(line_image,50,150)
        cropped_image_2 = region_of_interest(
            canny2,
            np.array([region_of_interest_vertices], np.int32
            ),
        )
        '''
        lines2 = cv2.HoughLinesP(
            line_image,
            rho=1,
            theta=np.pi/180,
            threshold=15,
            lines=np.array([]),
            minLineLength = 50,
            maxLineGap = 50
                )
                '''
        '''  
        for line in lines2:
            for x1, x2, y1, y2 in line:
                cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),5)
                '''
                
        
                
        #horizontal = cv2.addWeighted(current_frame,0.8,lines2,1,0)
              
        cv2.imshow("Original", current_frame)
        #cv2.imshow('Gray-Scaled', gray )
        #cv2.imshow('Blurred', blur)
        #cv2.imshow('Cannyed', cannyed)
        cv2.imshow('Lines', line_image)
        #cv2.imshow('Warped-Image',birdseye )
        #cv2.imshow('horizontal-Lines', second_horizontal_line)
        #cv2.imshow('hough of hough',lines2)
        #cv2.imshow('horizontal',cropped_image_2)
        
        cv2.waitKey(1)
        
    
        
    

def main(args = None):
    rclpy.init(args = args)
    intel_lane_subscriber = IntelLaneSubscriber()
    rclpy.spin(intel_lane_subscriber)
    intel_lane_subscriber.destroy_node()
    rclpy.shutdown()

if __name__ == "__main":
    main()