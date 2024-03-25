import cv2
import numpy as np
import os
from scipy import optimize
from matplotlib import pyplot as plt, cm, colors

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ackermann_msgs.msg import AckermannDrive
from cv_bridge import CvBridge
from std_msgs.msg import Float64
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import pyrealsense2 as rs
from matplotlib import image as mpimg
import math


#xm_per_pix = 3.7 / 720
#ym_per_pix = 30 / 720

#CWD_PATH = os.getcwd()

class LaneDetection(Node):

#########################################################################           
#################### AB HIER Wieder Alt

#####################################################################
    
    def __init__(self):
        
        super().__init__('detection_node')
        #timer_period = 0.05
        #self.timer = self.create_timer(timer_period, self.timer_callback)
        #self.timer_ = self.create_timer(1.0 / 10.0, self.timer_callback)
        self.subscription_rgb = self.create_subscription(Image,"rgb_frame", self.timer_callback, 10)
        self.publisher = self.create_publisher(AckermannDrive, "/ackermann_cmd" ,1)
        self.ack_msg = AckermannDrive()
        #self.cap = cv2.VideoCapture(2)
        timer_period = 0.05
        self.br_rgb = CvBridge()
        
             
        
        try:
            self.pipe = rs.pipeline()
            self.cfg  = rs.config()
            self.cfg.enable_stream(rs.stream.color, 1280,720, rs.format.bgr8, 30)
        
            self.pipe.start(self.cfg)
            self.timer = self.create_timer(timer_period, self.timer_callback)
        
            
        except Exception as e:
            print(e)
            self.get_logger().error('INTEL REALSENSE NOT CONNECTED')
        
        
        #self.cap = self.br_rgb
        
   

    #inpImage =   cv2.VideoCapture('/home/ADC/src/camera_run/camera_run/Challenger.mp4')
    def new_processImage(self,color_image):
        pts_src = np.array([[220,430],[950, 430],[0, 520],[1180, 520]], dtype=np.float32)       #
        pts_dst = np.array([[0  , 0 ],[1280, 0 ],[0, 720],[1280, 720]], dtype=np.float32)       #
        
        # Compute the perspective transform matrix and apply it
        M = cv2.getPerspectiveTransform(pts_src, pts_dst)
        warped = cv2.warpPerspective(color_image, M, (1280, 720))

        # Convert to grayscale and apply Canny edge detection
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        # Set a threshold to define 'white' in the image
        # This threshold will need to be tuned for your lighting conditions and lane color
        threshold = 150
        # Create a binary mask where white lanes are white and everything else is black
        _, lane_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        edges = cv2.Canny(lane_mask, 50, 150)

        # Assuming 'Edges' is your image loaded as a numpy array
        bottom_row =    edges[719, :]  # Accessing the bottom row
        right_col =     edges[:, 1279]
        left_col =      edges[:, 0]
        top_row =       edges[0, :]
        mid_row =       edges[360, :]

        # Finding indices where pixel value is 255
        indices_down =  np.where(bottom_row == 255)[0]  # [0] is used to get the column indices
        indices_right = np.where(right_col == 255)[0]
        indices_left =  np.where(left_col == 255)[0]
        indices_top =   np.where(top_row == 255)[0]
        indices_mid =   np.where(mid_row == 255)[0]

        #################################### Different Cases for first window ################################

        # Case 1: Off-Lane
        if len(indices_down) == 0:
            cv2.putText(lane_mask, "Off-Lane!!!", (640, 720-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        elif len(indices_down) >= 2:
            lane_thickness_pixel = indices_down[1]-indices_down[0]
            #print('lane_thickness = ',lane_thickness_pixel,' pixels')
            pixel_per_mm = lane_thickness_pixel/50


        # Case 2: There is only one lane detected
        if len(indices_down) == 2:
            if indices_down[1]<1280/2 and not last_lane =="Right":
                #print('Only Left lane is detected')
                last_lane = "Left"
                cv2.putText(lane_mask, "Left Lane", ((indices_down[1]), 720-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            elif indices_down[0]>1280/2 and not last_lane =="Left":
                #print('Only Right lane is detected')
                last_lane = "Right"
                cv2.putText(lane_mask, "Right Lane", ((indices_down[1]), 720-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            lane_middle = indices_down[1] + round(750/2/pixel_per_mm) # Virtual middle


        # Case 3: There are more lines detected!
        if len(indices_down) > 4:
            #print('Flase Lanes are detected!')
            indices_down_corrected = sorted(indices_down, key=lambda x: abs(x - 640))
            indices_down_corrected = indices_down_corrected[:4]
            indices_down_corrected = sorted(indices_down_corrected)

            if 0.5*60<=indices_down_corrected[1]-indices_down_corrected[0]<=2*60 and 0.5*60<=indices_down_corrected[3]-indices_down_corrected[2]<=2*60:
                indices_down = indices_down_corrected
                #print('Corrected indices_down = ',indices_down)


        # Case 4: There are 2 lines detected
        if len(indices_down) == 4:
            #print('Both lanes are detected')
            last_lane = []
            cv2.putText(lane_mask, "Left Lane" , ((indices_down[1])+0  , 720-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(lane_mask, "Right Lane", ((indices_down[3])-250, 720-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            lane_middle = round((indices_down[2]+indices_down[1])/2)            # real middle



        #################################### Different Cases for second window ################################

        # Case 2
        if 2 <= len(indices_mid) <= 3:
                lane_middle_upper = indices_mid[1] + round(750/2/pixel_per_mm) # Virtual middle

        # Case 3
        if len(indices_mid) > 4:
            #print('Flase Lanes are detected!')
            indices_mid_corrected = sorted(indices_mid, key=lambda x: abs(x - 640))
            indices_mid_corrected = indices_mid_corrected[:4]
            indices_mid_corrected = sorted(indices_mid_corrected)

            if 0.5*60<=indices_mid_corrected[1]-indices_mid_corrected[0]<=2*60 and 0.5*60<=indices_mid_corrected[3]-indices_mid_corrected[2]<=2*60:
                indices_mid = indices_mid_corrected
                #print('Corrected indices_mid = ',indices_mid)     

        # Case 4
        if len(indices_mid) == 4:
            lane_middle_upper = round((indices_mid[2]+indices_mid[1])/2)   


        #################################### Calculate the measured signals ##################################
        
        off_center = round((1280/2 - lane_middle)/pixel_per_mm)
        off_center = max(-750, min(off_center, 750))
        '''
        if not first_frame:
            #print('ratio = ',off_center/last_off_center)
            if not 0.1*abs(last_off_center)<=abs(off_center)<=10.0*abs(last_off_center):
                off_center = last_off_center
                '''
                
        
        last_off_center = off_center
        first_frame = False
        cv2.putText(lane_mask, str(off_center), (lane_middle, 720-50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Heading Angle
        heading_angle = np.arctan2(360 - 720-50, lane_middle_upper - lane_middle) + math.pi/2 #Radian
        heading_angle = round(heading_angle,2)
        cv2.putText(lane_mask, str(heading_angle), (lane_middle_upper, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.arrowedLine(lane_mask, (lane_middle, 720-50), (lane_middle_upper, 360), (255, 255, 255), 2)
        

        ################################### Display the resulting frame ##################################

        # Assuming all images are the same size
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        lane_mask = cv2.cvtColor(lane_mask, cv2.COLOR_GRAY2BGR)

        # Draw the region of interest on the color image
        pts = pts_src.astype(np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(color_image, [pts], True, (0, 255, 0), 2)
        
        return lane_mask, off_center
    
    def processImage(self,inpImage):

        # Apply HLS color filtering to filter out white lane lines
        hls = cv2.cvtColor(inpImage, cv2.COLOR_BGR2HLS)
        lower_white = np.array([0, 160, 10])
        upper_white = np.array([255, 255, 255])
        mask = cv2.inRange(inpImage, lower_white, upper_white)
        hls_result = cv2.bitwise_and(inpImage, inpImage, mask = mask)

        # Convert image to grayscale, apply threshold, blur & extract edges
        gray = cv2.cvtColor(hls_result, cv2.COLOR_BGR2GRAY)
        ret, thresh = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        blur = cv2.GaussianBlur(thresh,(3, 3), 0)
        canny = cv2.Canny(blur, 40, 60)
        
        #cv2.imshow('canny',inpImage)
        #cv2.waitKey(10)
        
        
        return  inpImage,hls_result, gray, thresh, blur, canny
    
    def perspectiveWarp(self,inpImage):

        # Get image size
        img_size = (inpImage.shape[1], inpImage.shape[0])

        # Perspective points to be warped
        src = np.float32([[380,430],[1060,430],[200,510],[1270,510]])    #Aktuell
        #src = np.float32([[590, 440],[690, 440],[200, 640],[1000, 640]]) #ORIGINAL
        
        # Window to be shown
        dst = np.float32([[200, 0],[1200, 0],[200, 710],[1200, 710]])

                        
        # Matrix to warp the image for birdseye window
        matrix = cv2.getPerspectiveTransform(src, dst)
        # Inverse matrix to unwarp the image for final window
        minv = cv2.getPerspectiveTransform(dst, src)
        birdseye = cv2.warpPerspective(inpImage, matrix, img_size)

        # Get the birdseye window dimensions
        height, width = birdseye.shape[:2]

        # Divide the birdseye view into 2 halves to separate left & right lanes
        birdseyeLeft  = birdseye[0:height, 0:width // 2]
        birdseyeRight = birdseye[0:height, width // 2:width]

        # Display birdseye view image
        # cv2.imshow("Birdseye" , birdseye)
        # cv2.imshow("Birdseye Left" , birdseyeLeft)
        # cv2.imshow("Birdseye Right", birdseyeRight)
        # cv2.waitKey(10)
        

        return birdseye, birdseyeLeft, birdseyeRight, minv

    def plotHistogram(self,inpImage):

        histogram = np.sum(inpImage[inpImage.shape[0] // 2:, :], axis = 0)

        midpoint = np.int(histogram.shape[0] / 2)
        leftxBase = np.argmax(histogram[:midpoint])
        rightxBase = np.argmax(histogram[midpoint:]) + midpoint

        plt.xlabel("Image X Coordinates")
        plt.ylabel("Number of White Pixels")

        # Return histogram and x-coordinates of left & right lanes to calculate
        # lane width in pixels
        return histogram, leftxBase, rightxBase

    def slide_window_search(self,binary_warped, histogram):

        # Find the start of left and right lane lines using histogram info
        out_img = np.dstack((binary_warped, binary_warped, binary_warped)) * 255
        midpoint = np.int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        # A total of 9 windows will be used
        nwindows = 9
        window_height = np.int(binary_warped.shape[0] / nwindows)
        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        leftx_current = leftx_base
        rightx_current = rightx_base
        margin = 100
        minpix = 50
        left_lane_inds = []
        right_lane_inds = []   
        
        for window in range(nwindows):
                win_y_low = binary_warped.shape[0] - (window + 1) * window_height
                win_y_high = binary_warped.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high),
                (0,255,0), 2)
                cv2.rectangle(out_img, (win_xright_low,win_y_low), (win_xright_high,win_y_high),
                (0,255,0), 2)
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                if len(good_left_inds) > minpix:
                    leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
                if len(good_right_inds) > minpix:
                    rightx_current = np.int(np.mean(nonzerox[good_right_inds]))
            #### END - Loop to iterate through windows and search for lane lines #######

        left_lane_inds = np.concatenate(left_lane_inds)
        right_lane_inds = np.concatenate(right_lane_inds)

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]

        # Apply 2nd degree polynomial fit to fit curves
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)


        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0] * ploty**2 + left_fit[1] * ploty + left_fit[2]
        right_fitx = right_fit[0] * ploty**2 + right_fit[1] * ploty + right_fit[2]

        ltx = np.trunc(left_fitx)
        rtx = np.trunc(right_fitx)
        plt.plot(right_fitx)
        # plt.show()

        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        # plt.imshow(out_img)
        plt.plot(left_fitx,  ploty, color = 'yellow')
        plt.plot(right_fitx, ploty, color = 'yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

        return ploty, left_fit, right_fit, ltx, rtx

    def general_search(self,binary_warped, left_fit, right_fit):

        nonzero = binary_warped.nonzero()
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        margin = 100
        left_lane_inds = ((nonzerox > (left_fit[0]*(nonzeroy**2) + left_fit[1]*nonzeroy +
        left_fit[2] - margin)) & (nonzerox < (left_fit[0]*(nonzeroy**2) +
        left_fit[1]*nonzeroy + left_fit[2] + margin)))

        right_lane_inds = ((nonzerox > (right_fit[0]*(nonzeroy**2) + right_fit[1]*nonzeroy +
        right_fit[2] - margin)) & (nonzerox < (right_fit[0]*(nonzeroy**2) +
        right_fit[1]*nonzeroy + right_fit[2] + margin)))

        leftx = nonzerox[left_lane_inds]
        lefty = nonzeroy[left_lane_inds]
        rightx = nonzerox[right_lane_inds]
        righty = nonzeroy[right_lane_inds]
        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)
        ploty = np.linspace(0, binary_warped.shape[0]-1, binary_warped.shape[0])
        left_fitx = left_fit[0]*ploty**2 + left_fit[1]*ploty + left_fit[2]
        right_fitx = right_fit[0]*ploty**2 + right_fit[1]*ploty + right_fit[2]


        ############################# VISUALIZATION #####################################

        out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
        window_img = np.zeros_like(out_img)
        out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 0]
        out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 0, 255]

        left_line_window1 = np.array([np.transpose(np.vstack([left_fitx-margin, ploty]))])
        left_line_window2 = np.array([np.flipud(np.transpose(np.vstack([left_fitx+margin,
                                    ploty])))])
        left_line_pts = np.hstack((left_line_window1, left_line_window2))
        right_line_window1 = np.array([np.transpose(np.vstack([right_fitx-margin, ploty]))])
        right_line_window2 = np.array([np.flipud(np.transpose(np.vstack([right_fitx+margin, ploty])))])
        right_line_pts = np.hstack((right_line_window1, right_line_window2))

        cv2.fillPoly(window_img, np.int_([left_line_pts]), (0, 255, 0))
        cv2.fillPoly(window_img, np.int_([right_line_pts]), (0, 255, 0))
        result = cv2.addWeighted(out_img, 1, window_img, 0.3, 0)

        # plt.imshow(result)
        plt.plot(left_fitx,  ploty, color = 'yellow')
        plt.plot(right_fitx, ploty, color = 'yellow')
        plt.xlim(0, 1280)
        plt.ylim(720, 0)

        ret = {}
        ret['leftx'] = leftx
        ret['rightx'] = rightx
        ret['left_fitx'] = left_fitx
        ret['right_fitx'] = right_fitx
        ret['ploty'] = ploty

        return ret

    def measure_lane_curvature(self,ploty, leftx, rightx):
        xm_per_pix = 0.4 / 1280         #scaling parameters for transforming pixels into real distance
        ym_per_pix = 0.6 / 720
        leftx = leftx[::-1]             # Reverse to match top-to-bottom in y
        rightx = rightx[::-1]           # Reverse to match top-to-bottom in y

        # Choose the maximum y-value, corresponding to the bottom of the image
        y_eval = np.max(ploty)

        # Fit new polynomials to x, y in world space
        left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)

        # Calculate the new radii of curvature
        left_curverad  = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
        right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])
        
        # Now our radius of curvature is in meters
        # print(left_curverad, 'm', right_curverad, 'm')

       

        # Decide if it is a left or a right curve
        if leftx[0] - leftx[-1] > 60:
            curve_direction = 'Left Curve'
            
        elif leftx[-1] - leftx[0] > 60:
            curve_direction = 'Right Curve'
            
        else:
            curve_direction = 'Straight'
                 
        
                
        #while rclpy.ok():
        
        #print(direction)
                    
        return (left_curverad + right_curverad) / 2.0, curve_direction

        
        
    def draw_lane_lines(self,original_image, warped_image, Minv, draw_info):

        leftx = draw_info['leftx']
        rightx = draw_info['rightx']
        left_fitx = draw_info['left_fitx']
        right_fitx = draw_info['right_fitx']
        ploty = draw_info['ploty']

        warp_zero = np.zeros_like(warped_image).astype(np.uint8)
        color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

        pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
        pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
        pts = np.hstack((pts_left, pts_right))

        mean_x = np.mean((left_fitx, right_fitx), axis=0)
        pts_mean = np.array([np.flipud(np.transpose(np.vstack([mean_x, ploty])))])

        cv2.fillPoly(color_warp, np.int_([pts]), (0, 255, 0))
        cv2.fillPoly(color_warp, np.int_([pts_mean]), (0, 255, 255))

        newwarp = cv2.warpPerspective(color_warp, Minv, (original_image.shape[1], original_image.shape[0]))
        result = cv2.addWeighted(original_image, 1, newwarp, 0.3, 0)

        return pts_mean, result

    def offCenter(self,meanPts, inpFrame):
        xm_per_pix = 0.6 / 1280
        ym_per_pix = 0.6 / 720

        # Calculating deviation in meters
        mpts = meanPts[-1][-1][-2].astype(int)
        pixelDeviation = inpFrame.shape[1] / 2 - abs(mpts)
        deviation = pixelDeviation * xm_per_pix
        direction = "left" if deviation < 0 else "right"
        
        return deviation, direction



    def addText(self,img, radius, direction, deviation, devDirection):

        # Add the radius and center position to the image
        font = cv2.FONT_HERSHEY_TRIPLEX

        if (direction != 'Straight'):
            text = 'Radius of Curvature: ' + '{:04.0f}'.format(radius) + 'm'
            text1 = 'Curve Direction: ' + (direction)

        else:
            text = 'Radius of Curvature: ' + 'N/A'
            text1 = 'Curve Direction: ' + (direction)
            

        cv2.putText(img, text , (50,100), font, 0.8, (0,100, 200), 2, cv2.LINE_AA)
        cv2.putText(img, text1, (50,150), font, 0.8, (0,100, 200), 2, cv2.LINE_AA)

        # Deviation
        deviation_text = 'Off Center: ' + str(round(abs(deviation), 3)) + 'm' + ' to the ' + devDirection
        cv2.putText(img, deviation_text, (50, 200), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,100, 200), 2, cv2.LINE_AA)
        #steering_text = 'Steering angle:' +str(round())

        return img
    
    #image = readVideo()

    def calc_steering_angle(self,centerOffset):
        Lane_Center = 0.0
        max_steering = 45.0
        kp = 5.0
        ki = 0.1
        kd = 0.5

        previous_error = 0.0
        integral = 0.0
        
        offset_error = -(centerOffset-Lane_Center)
        proportional = kp*offset_error
        integral = integral+ki*offset_error
        derivative = kd*(offset_error-previous_error)
        previous_error = offset_error
        control_signal = proportional+integral+derivative
        steering_angle1 = control_signal
        
        #test
        steering_angle1 = steering_angle1+0.5
        
        #steering_angle1 = np.clip(steering_angle1,-max_steering,max_steering)
        
        steering_angle1 = np.clip(steering_angle1,0.1,0.9)
        
        return steering_angle1



    '''
    while True:

        _, frame = image.read()


        # Apply perspective warping by calling the "perspectiveWarp()" function
        # Then assign it to the variable called (birdView)
        # Provide this function with:
        # 1- an image to apply perspective warping (frame)
        birdView, birdViewL, birdViewR, minverse = perspectiveWarp(frame)


        # Apply image processing by calling the "processImage()" function
        # Then assign their respective variables (img, hls, grayscale, thresh, blur, canny)
        # Provide this function with:
        # 1- an already perspective warped image to process (birdView)
        img, hls, grayscale, thresh, blur, canny = processImage(birdView)
        imgL, hlsL, grayscaleL, threshL, blurL, cannyL = processImage(birdViewL)
        imgR, hlsR, grayscaleR, threshR, blurR, cannyR = processImage(birdViewR)


        # Plot and display the histogram by calling the "get_histogram()" function
        # Provide this function with:
        # 1- an image to calculate histogram on (thresh)
        hist, leftBase, rightBase = plotHistogram(thresh)
        # print(rightBase - leftBase)
        plt.plot(hist)
        # plt.show()


        ploty, left_fit, right_fit, left_fitx, right_fitx = slide_window_search(thresh, hist)
        plt.plot(left_fit)
        # plt.show()


        draw_info = general_search(thresh, left_fit, right_fit)
        # plt.show()


        curveRad, curveDir = measure_lane_curvature(ploty, left_fitx, right_fitx)


        # Filling the area of detected lanes with green
        meanPts, result = draw_lane_lines(frame, thresh, minverse, draw_info)


        deviation, directionDev = offCenter(meanPts, frame)


        # Adding text to our final image
        finalImg = addText(result, curveRad, curveDir, deviation, directionDev)

        # Displaying final image
        cv2.imshow("Final", finalImg)
        
        
            
        if cv2.waitKey(1) == 13:
            break  
              
        # Wait for the ENTER key to be pressed to stop playback
        #if cv2.waitKey(1) == 13:
            #break
        '''
        
                   
               
    def timer_callback(self):
        
        #self.get_logger().warning("Receiving RGB frame")
        #frame = self.br_rgb.imgmsg_to_cv2(data)
        #frame = cv2.videocapture('Challenger.mp4')
        #print('FRANE: ',frame)
        
        
        
        frames= self.pipe.wait_for_frames()
        color_frame = frames.get_color_frame()
        print(color_frame)
        frame = np.asanyarray(color_frame.get_data())
        print('FRAME: ',frame)
        
        
        '''

        birdView, birdViewL, birdViewR,minverse = self.perspectiveWarp(frame)
        img, hls,grayscale, thresh,blur, canny = self.processImage(birdView)
        imgL, hlsL,grayscaleL, threshL,blurL, cannyL = self.processImage(birdViewL)
        imgR, hlsR,grayscaleR, threshR,blurR, cannyR = self.processImage(birdViewR)
        
        hist, leftBase, rightBase = self.plotHistogram(thresh)
        ploty, left_fit, right_fit, left_fitx, right_fitx = self.slide_window_search(thresh,hist)
        draw_info = self.general_search(thresh,left_fit, right_fit)
        curveRad,curveDir = self.measure_lane_curvature(ploty, left_fitx, right_fitx)
        meanPts, results = self.draw_lane_lines(frame,thresh,minverse,draw_info)
        #cv2.imshow('',results)
        deviation, directionDev = self.offCenter(meanPts,frame)
        steering_angle = self.calc_steering_angle(deviation)
        print('PID STEERING ANGLE:', steering_angle)
        print('DEVIATIN OFFCENTER:',deviation)
        finalImg = self.addText(results,curveRad,curveDir,deviation,directionDev)
        '''
        final_image,offset = self.new_processImage(frame)
        steering_angle = self.calc_steering_angle(offset)
        

        speed = -0.4
        #steering_angle = 0.5
        STEERING_BIAS = -0.021

        # Initialization of the Ackermann message
        self.ack_msg.speed = speed
        self.ack_msg.steering_angle = steering_angle + STEERING_BIAS
                
        # Troubleshooting - Debugging Monitor
        print('REAL STEERING ANGLE',steering_angle)
        print('STEERING_ANGLE',steering_angle)

        #self.publisher.publish(self.ackermann_msg.speed)
        self.publisher.publish(self.ack_msg)
        #self.get_logger().info(f"Speed: {speed}, Steering Angle: {steering_angle}")
        #print('DevDirection:',directionDev)
        #print('CurveRad',curveRad)
        # cv2.imshow('Original',image)
        cv2.imshow('Original',frame)
        #cv2.imshow('Final',img)
        #cv2.imshow('Lanes',finalImg)
        cv2.imshow('Final Image: ', final_image)
        
        cv2.waitKey(1)
        
        #if cv2.waitKey(1) == 13:
            # break
    
        #image.release() 
        #cv2.destroyAllWindows()  
        
    #image.release()
    #cv2.destroyAllWindows() 

def main(args = None):
    rclpy.init(args = args)
    detection_node = LaneDetection()
    rclpy.spin(detection_node)
    detection_node.destroy_node
    rclpy.shutdown()
    
if __name__ == '__main':
    main()
    
    