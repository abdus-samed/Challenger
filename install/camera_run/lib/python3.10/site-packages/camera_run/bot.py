import rclpy
from rclpy.node import Node
import serial
import time

class VESCNode(Node):
    def __init__(self):
        super().__init__('vesc_node')
        self.ser = serial.Serial('/dev/ttyACM0', baudrate=115200, timeout=1)
        self.create_timer(1, self.timer_callback)

    def timer_callback(self):
        # Example: Send a command and print the response
        response = self.send_command("faults\n")
        self.get_logger().info(f"Faults: {response}")

    def send_command(self, command):
        self.ser.write(command.encode('utf-8'))
        time.sleep(0.1)  # Wait for response
        response = self.ser.readline().decode('ISO-8859-1').strip()
        return response

def main(args=None):
    rclpy.init(args=args)
    vesc_node = VESCNode()
    rclpy.spin(vesc_node)
    vesc_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
