#!/usr/bin/env python3


import sys
import rclpy
from rclpy.node import Node
import cv2
import time
import numpy as np

from std_msgs.msg           import String
from sensor_msgs.msg        import Image
from geometry_msgs.msg      import Point
from cv_bridge              import CvBridge, CvBridgeError


#---------- Blob detecting function: returns keypoints and mask
#-- return keypoints, reversemask
def blob_detect(image,                  #-- The frame (cv standard)
                hsv_min,                #-- minimum threshold of the hsv filter [h_min, s_min, v_min]
                hsv_max,                #-- maximum threshold of the hsv filter [h_max, s_max, v_max]
                blur=0,                 #-- blur value (default 0)
                blob_params=None,       #-- blob parameters (default None)
                search_window=None,     #-- window where to search as [x_min, y_min, x_max, y_max] adimensional (0.0 to 1.0) starting from top left corner
                imshow=False
               ):

    #- Blur image to remove noise
    if blur > 0: 
        image    = cv2.blur(image, (blur, blur))
        #- Show result
        if imshow:
            cv2.imshow("Blur", image)
            cv2.waitKey(0)
        
    #- Search window
    if search_window is None: search_window = [0.0, 0.0, 1.0, 1.0]
    
    #- Convert image from BGR to HSV
    hsv     = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    #- Apply HSV threshold
    mask    = cv2.inRange(hsv,hsv_min, hsv_max)
    
    #- Show HSV Mask
    if imshow:
        cv2.imshow("HSV Mask", mask)
    
    #- dilate makes the in range areas larger
    mask = cv2.dilate(mask, None, iterations=2)
    #- Show HSV Mask
    if imshow:
        cv2.imshow("Dilate Mask", mask)   
        cv2.waitKey(0)
        
    mask = cv2.erode(mask, None, iterations=2)
    
    #- Show dilate/erode mask
    if imshow:
        cv2.imshow("Erode Mask", mask)
        cv2.waitKey(0)
    
    #- Cut the image using the search mask
    mask = apply_search_window(mask, search_window)
    
    if imshow:
        cv2.imshow("Searching Mask", mask)
        cv2.waitKey(0)

    #- build default blob detection parameters, if none have been provided
    if blob_params is None:
        # Set up the SimpleBlobdetector with default parameters.
        params = cv2.SimpleBlobDetector_Params()
         
        # Change thresholds
        params.minThreshold = 0
        params.maxThreshold = 100
         
        # Filter by Area.
        params.filterByArea = True
        params.minArea = 30
        params.maxArea = 20000
         
        # Filter by Circularity
        params.filterByCircularity = True
        params.minCircularity = 0.1
         
        # Filter by Convexity
        params.filterByConvexity = True
        params.minConvexity = 0.5
         
        # Filter by Inertia
        params.filterByInertia =True
        params.minInertiaRatio = 0.5
         
    else:
        params = blob_params     

    #- Apply blob detection
    detector = cv2.SimpleBlobDetector_create(params)

    # Reverse the mask: blobs are black on white
    reversemask = 255-mask
    
    if imshow:
        cv2.imshow("Reverse Mask", reversemask)
        cv2.waitKey(0)
        
    keypoints = detector.detect(reversemask)

    return keypoints, reversemask

#---------- Draw detected blobs: returns the image
#-- return(im_with_keypoints)
def draw_keypoints(image,                   #-- Input image
                   keypoints,               #-- CV keypoints
                   line_color=(0,0,255),    #-- line's color (b,g,r)
                   imshow=False             #-- show the result
                  ):
    
    #-- Draw detected blobs as red circles.
    #-- cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    im_with_keypoints = cv2.drawKeypoints(image, keypoints, np.array([]), line_color, cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
 
    if imshow:
        # Show keypoints
        cv2.imshow("Keypoints", im_with_keypoints)
        
    return(im_with_keypoints)

#---------- Draw search window: returns the image
#-- return(image)
def draw_window(image,              #- Input image
                window_adim,        #- window in adimensional units
                color=(255,0,0),    #- line's color
                line=5,             #- line's thickness
                imshow=False        #- show the image
               ):
    
    rows = image.shape[0]
    cols = image.shape[1]
    
    x_min_px    = int(cols*window_adim[0])
    y_min_px    = int(rows*window_adim[1])
    x_max_px    = int(cols*window_adim[2])
    y_max_px    = int(rows*window_adim[3])  
    
    #-- Draw a rectangle from top left to bottom right corner
    image = cv2.rectangle(image,(x_min_px,y_min_px),(x_max_px,y_max_px),color,line)
    
    if imshow:
        # Show keypoints
        cv2.imshow("Keypoints", image)

    return(image)

#---------- Draw X Y frame
#-- return(image)
def draw_frame(image,
               dimension=0.3,      #- dimension relative to frame size
               line=2              #- line's thickness
    ):
    
    rows = image.shape[0]
    cols = image.shape[1]
    size = min([rows, cols])
    center_x = int(cols/2.0)
    center_y = int(rows/2.0)
    
    line_length = int(size*dimension)
    
    #-- X
    image = cv2.line(image, (center_x, center_y), (center_x+line_length, center_y), (0,0,255), line)
    #-- Y
    image = cv2.line(image, (center_x, center_y), (center_x, center_y+line_length), (0,255,0), line)
    
    return (image)

#---------- Apply search window: returns the image
#-- return(image)
def apply_search_window(image, window_adim=[0.0, 0.0, 1.0, 1.0]):
    rows = image.shape[0]
    cols = image.shape[1]
    x_min_px    = int(cols*window_adim[0])
    y_min_px    = int(rows*window_adim[1])
    x_max_px    = int(cols*window_adim[2])
    y_max_px    = int(rows*window_adim[3])    
    
    #--- Initialize the mask as a black image
    mask = np.zeros(image.shape,np.uint8)
    
    #--- Copy the pixels from the original image corresponding to the window
    mask[y_min_px:y_max_px,x_min_px:x_max_px] = image[y_min_px:y_max_px,x_min_px:x_max_px]   
    
    #--- return the mask
    return(mask)
    
#---------- Apply a blur to the outside search region
#-- return(image)
def blur_outside(image, blur=5, window_adim=[0.0, 0.0, 1.0, 1.0]):
    rows = image.shape[0]
    cols = image.shape[1]
    x_min_px    = int(cols*window_adim[0])
    y_min_px    = int(rows*window_adim[1])
    x_max_px    = int(cols*window_adim[2])
    y_max_px    = int(rows*window_adim[3])    
    
    #--- Initialize the mask as a black image
    mask    = cv2.blur(image, (blur, blur))
    
    #--- Copy the pixels from the original image corresponding to the window
    mask[y_min_px:y_max_px,x_min_px:x_max_px] = image[y_min_px:y_max_px,x_min_px:x_max_px]   
    
    
    
    #--- return the mask
    return(mask)
    
#---------- Obtain the camera relative frame coordinate of one single keypoint
#-- return(x,y)
def get_blob_relative_position(image, keyPoint):
    rows = float(image.shape[0])
    cols = float(image.shape[1])
    # print(rows, cols)
    center_x    = 0.5*cols
    center_y    = 0.5*rows
    # print(center_x)
    x = (keyPoint.pt[0] - center_x)/(center_x)
    y = (keyPoint.pt[1] - center_y)/(center_y)
    return(x,y)


class BlobDetector(Node):

    def __init__(self, thr_min, thr_max, blur=15, blob_params=None, detection_window=None):
        super().__init__("blob_detector") 
        self.set_threshold(thr_min, thr_max)
        self.set_blur(blur)
        self.set_blob_params(blob_params)
        self.detection_window = detection_window
        
        self._t0 = time.time()
        
        self.blob_point = Point()
    
        self.get_logger().info(">> Publishing image to topic image_blob")
        self.image_pub = self.create_publisher(Image,"/blob/image_blob",1)
        self.mask_pub = self.create_publisher(Image,"/blob/image_mask",1)
        self.get_logger().info(">> Publishing position to topic point_blob")
        self.blob_pub  = self.create_publisher(Point,"/blob/point_blob",1)

        self.bridge = CvBridge()
        self.image_sub = self.create_subscription(Image,"/camera/image",self.callback, 1)
        self.get_logger().info("<< Subscribed to topic /raspicam2/image")
        
    def set_threshold(self, thr_min, thr_max):
        self._threshold = [thr_min, thr_max]
        
    def set_blur(self, blur):
        self._blur = blur
      
    def set_blob_params(self, blob_params):
        self._blob_params = blob_params
        
    # def get_blob_relative_position(self, cv_image, keyPoint):
        # rows = float(cv_image.shape[0])
        # cols = float(cv_image.shape[1])
        # # print(rows, cols)
        # center_x    = 0.5*cols
        # center_y    = 0.5*rows
        # # print(center_x)
        # x = (keyPoint.pt[0] - center_x)/(center_x)
        # y = (keyPoint.pt[1] - center_y)/(center_y)
        # return(x,y)
        
        
    def callback(self,data):
        #--- Assuming image is 320x240
        cv_image = None
        data.step = data.width * 3
        self.get_logger().info(str(len(data.data)))
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            self.get_logger().error(e)

        (rows,cols,channels) = cv_image.shape
        if cols > 60 and rows > 60 :
            #--- Detect blobs
            keypoints, mask   = blob_detect(cv_image, self._threshold[0], self._threshold[1], self._blur,
                                            blob_params=self._blob_params, search_window=self.detection_window )
            #--- Draw search window and blobs
            cv_image    = blur_outside(cv_image, 10, self.detection_window)

            cv_image    = draw_window(cv_image, self.detection_window, line=1)
            cv_image    = draw_frame(cv_image)
            
            cv_image    = draw_keypoints(cv_image, keypoints) 
            
            try:
                self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))
                self.mask_pub.publish(self.bridge.cv2_to_imgmsg(mask, "8UC1"))
            except CvBridgeError as e:
                self.get_logger().error(e)            

            blobsX = []
            blobsY = []
            for i, keyPoint in enumerate(keypoints):
                #--- Here you can implement some tracking algorithm to filter multiple detections
                #--- We are simply getting the first two results and assuming they are the lanes
                x = keyPoint.pt[0]
                y = keyPoint.pt[1]
                s = keyPoint.size
                self.get_logger().info("kp %d: s = %3d   x = %3d  y= %3d"%(i, s, x, y))
                
                #--- Find x and y position in camera adimensional frame
                x, y = get_blob_relative_position(cv_image, keyPoint)
                
                blobsX.append(x)
                blobsY.append(y)
                if i > 0:
                    #find the point between the lanes to aim at
                    middlePointX = (blobsX[0] + blobsX[1]) / 2
                    middlePointY = (blobsY[0] + blobsY[1]) / 2
                    self.blob_point.x = middlePointX
                    self.blob_point.y = middlePointY
                    self.blob_pub.publish(self.blob_point) 
                    break
                    
            fps = 1.0/(time.time()-self._t0)
            self._t0 = time.time()
            
def main(args=None):
    rclpy.init(args=args)
    # blue_min = (77,40,0)
    # blue_max = (101, 255, 255) 
    # blue_min = (82,31,62)
    # blue_max = (106, 116, 193)     
    blue_min = (55,40,0)
    blue_max = (150, 255, 255)     
    
    blur = 5
    
    #--- detection window respect to camera frame in [x_min, y_min, x_max, y_max] adimensional (0 to 1)
    x_min   = 0.1
    x_max   = 0.9
    y_min   = 0.4
    y_max   = 0.9
    
    detection_window = [x_min, y_min, x_max, y_max]
    
    params = cv2.SimpleBlobDetector_Params()
         
    # Change thresholds
    params.minThreshold = 0
    params.maxThreshold = 100
     
    # Filter by Area.
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 20000
     
    # Filter by Circularity
    params.filterByCircularity = False
    #params.minCircularity = 0.1
     
    # Filter by Convexity
    params.filterByConvexity = False
    params.minConvexity = 0.2
     
    # Filter by Inertia - more elongated blobs i.e. lines or lanes
    params.filterByInertia = True
    params.maxInertiaRatio = 0.5   

    node = BlobDetector(blue_min, blue_max, blur, params, detection_window)
    
    rclpy.spin(node)
    cv2.destroyAllWindows()
    rclpy.shutdown()
    

if __name__ == '__main__':
    main()

