import cv2
import numpy as np
import pyrealsense2 as rs
import os
import math

#################################### Choose the Run Mode ################################

RunMode = "Video"                   # Either Video or Camera

#################################### Initialize Settings ################################
first_frame = True
last_lane = []
lane_middle = 0
pixel_per_mm = 1
lane_middle_upper = 0

if RunMode == "Camera":
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    #config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)

    pipeline.start(config)
    pts_src = np.array([[220,430],[950, 430],[0, 520],[1180, 520]], dtype=np.float32)       #
    pts_dst = np.array([[0  , 0 ],[1280, 0 ],[0, 720],[1280, 720]], dtype=np.float32)       #


if RunMode == "Video":
    CWD_PATH = os.getcwd()
    video = cv2.VideoCapture(os.path.join(CWD_PATH, 'Challenger_new_2.mp4'))
    pts_src = np.array([[310,425],[942, 425],[81, 523],[1187, 523]], dtype=np.float32)        # Tested for Challenger Video
    pts_dst = np.array([[0  , 0 ],[1280, 0 ],[0, 720],[1280, 720]], dtype=np.float32)        # Tested for Challenger Video


try:
    while True:
        #################################### Preparing the frames for analysis ################################

        # Wait for a coherent pair of frames: depth and color
        if RunMode == "Video":
            _, color_image = video.read()
            
        elif RunMode == "Camera":
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            # Convert images to numpy arrays
            color_image = np.asanyarray(color_frame.get_data())
            if not color_frame:
                continue

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

        # Make smaller to show
        cv2.putText(color_image, "Original"  , (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(warped,      "Bird View" , (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(edges,       "Edge Det." , (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA)
        cv2.putText(lane_mask,   "Binary Ms.", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2, cv2.LINE_AA)
        color_image =   cv2.resize(color_image, (0, 0), fx=0.5, fy=0.5)
        warped =        cv2.resize(warped,      (0, 0), fx=0.5, fy=0.5)
        edges =         cv2.resize(edges,       (0, 0), fx=0.5, fy=0.5)
        lane_mask =     cv2.resize(lane_mask,   (0, 0), fx=0.5, fy=0.5)

        height, width, channels = color_image.shape

        # Create a new image that's twice the width and height of the original images
        combined_image = np.zeros((height * 2, width * 2, channels), dtype=np.uint8)

        # Place each image into the combined_image
        combined_image[:height, :width] = color_image   # Top-left
        combined_image[:height, width:] = warped        # Top-right
        combined_image[height:, :width] = edges         # Bottom-left
        combined_image[height:, width:] = lane_mask     # Bottom-right

        # Show the combined image
        cv2.imshow('Combined Image', combined_image)

        # Save the last frames for debugging
        cv2.imwrite('LastFrame.jpg', combined_image)

        ########################################### End ##############################################

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
finally:
    # Stop streaming
    if RunMode == "Camera":
        pipeline.stop()
    cv2.destroyAllWindows()
