import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDrive
from std_msgs.msg import String
from LaneDetection import offCenter

class LaneFollow(Node):
    def __init__(self):
        #self.node = rclpy.create_node('detection_node')
        super().__init__('lane_follower_node')
        self.follower_publisher = self.create_publisher(AckermannDrive, '/ackermann_cmd',10)
        self.ackermann_msg = AckermannDrive()
        timer_period = 0.5
        #self.timer = self.create_timer(timer_period, self.of)
        self.follower_subscriber = self.create_subscription(String,'steering',self.steering_callback,2)
        
    def steering_callback(self,msg):
        curve = offCenter()
        steering_angle = msg.data
        STEERING_BIAS = -0.021
        
        speed = 0.4
        
        
        self.ackermann_msg.speed = speed
        self.ackermann_msg.steering_angle = steering_angle + STEERING_BIAS

        self.publish(self.ackermann_msg)
        self.node.get_logger().info(f"Speed: {speed}, Steering Angle: {steering_angle}")

        




def main(args = None):
    rclpy.init(args=args)
    lane_follower_node = LaneFollow()
    rclpy.spin(lane_follower_node)
    lane_follower_node.node.destroy_node
    rclpy.shutdown()
    
if __name__ == '__main':
    main()