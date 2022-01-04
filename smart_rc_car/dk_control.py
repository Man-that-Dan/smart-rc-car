#!/usr/bin/env python3
"""
Gets the position of the blob and it commands to steer the wheels
Subscribes to 
    /blob/point_blob
    
Publishes commands to 
    /dkcar/control/cmd_vel    
"""
import math, time
import threading
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from geometry_msgs.msg import Point

K_LAT_DIST_TO_STEER     = 2.0

def saturate(value, min, max):
    if value <= min: return(min)
    elif value >= max: return(max)
    else: return(value)

class FollowLanes(Node):
    def __init__(self):
        super().__init__("follow_lanes") 
        self.blob_x         = 0.0
        self.blob_y         = 0.0
        self._time_detected = 0.0
        
        self.sub_center = self.create_subscription(Point, "/blob/point_blob",  self.update_path,1)
        self.get_logger().info("Subscribers set")
        
        self.pub_twist = self.create_publisher(Twist, "/dkcar/control/cmd_vel",5)
        self.get_logger().info("Publisher set")
        
        self._message = Twist()
        
        self._time_steer        = 0
        self._steer_sign_prev   = 0
        
    @property
    def is_detected(self): return(time.time() - self._time_detected < 1.0)
        
    def update_path(self, message):
        self.blob_x = message.x
        self.blob_y = message.y
        self._time_detected = time.time()
        #-- Get the control action
        steer_action, throttle_action    = self.get_control_action() 
            
        self.get_logger().info("Steering = %3.1f"%(steer_action))
            
        #-- update the message
        self._message.linear.x  = throttle_action
        self._message.angular.z = steer_action
            
        #-- publish it
        self.pub_twist.publish(self._message)
        # self.get_logger().info("Ball detected: %.1f  %.1f "%(self.blob_x, self.blob_y))

    def get_control_action(self):
        """
        Based on the current ranges, calculate the command
        
        """
        steer_action    = 0.0
        throttle_action = 0.0
        
        if self.is_detected:
            #--- Apply steering, proportional to how close the object is
            steer_action   =-K_LAT_DIST_TO_STEER*self.blob_x
            steer_action   = saturate(steer_action, -1.5, 1.5)
            self.get_logger().info("Steering command %.2f"%steer_action) 
            throttle_action = 0.37 
            
        return (steer_action, throttle_action)
        
    def run(self):
        
        #--- Set the control rate
        rate = self.create_rate(5)

        while rclpy.ok():
            #-- Get the control action
            steer_action, throttle_action    = self.get_control_action() 
            
            self.get_logger().info("Steering = %3.1f"%(steer_action))
            
            #-- update the message
            self._message.linear.x  = throttle_action
            self._message.angular.z = steer_action
            
            #-- publish it
            self.pub_twist.publish(self._message)

                  
            
def main(args=None):
    rclpy.init(args=args)
    node = FollowLanes()
    # Spin in a separate thread
    # thread = threading.Thread(target=rclpy.spin, args=(node, ), daemon=True)
    # node.run()
    rclpy.spin(node)
    
    rclpy.shutdown()
    #thread.join()


if __name__ == "__main__":
    main()