import rclpy
from rclpy.node import Node
from std_srvs.srv import SetBool
import serial
import time
import threading


class GripperNode(Node):
    def __init__(self):
        super().__init__('gripper_node')
        self.srv = self.create_service(SetBool, '/gripper_control', self.control_cb)

        try:
            self.ser = serial.Serial("/dev/ttyUSB0", 115200, timeout=1)
            time.sleep(2.0)
            self.get_logger().info('[GRIPPER] serial connected')
            self.ser.write(b"open\n")
            self.get_logger().info('[GRIPPER] initialized: open')
        except Exception as e:
            self.get_logger().error(f'[GRIPPER] serial error: {e}')

        self.get_logger().info('[GRIPPER] gripper_node started')

    def control_cb(self, request, response):
        try:
            if request.data:
                self.ser.write(b"grip\n")
                self.get_logger().info('[GRIPPER] grip')
                response.message = 'grip command sent'
            else:
                self.ser.write(b"open\n")
                self.get_logger().info('[GRIPPER] open')
                response.message = 'open command sent'
            response.success = True
        except Exception as e:
            self.get_logger().error(f'[GRIPPER] error: {e}')
            response.success = False
            response.message = str(e)
        return response


def main(args=None):
    rclpy.init(args=args)
    node = GripperNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    node.get_logger().info('[GRIPPER] terminal input ready: open / close / exit')

    try:
        while rclpy.ok():
            cmd = input().strip().lower()

            if not hasattr(node, 'ser') or not node.ser.is_open:
                node.get_logger().error('[GRIPPER] serial not connected')
                continue

            if cmd == 'open':
                node.ser.write(b"open\n")
                node.get_logger().info('[GRIPPER] terminal: open')
            elif cmd in ['close', 'grip']:
                node.ser.write(b"grip\n")
                node.get_logger().info('[GRIPPER] terminal: grip')
            elif cmd == 'exit':
                break
            elif cmd != '':
                node.get_logger().warn('[GRIPPER] unknown command')

    except KeyboardInterrupt:
        pass
    except EOFError:
        pass
    finally:
        if hasattr(node, 'ser') and node.ser.is_open:
            node.ser.close()
            node.get_logger().info('[GRIPPER] serial closed')
        rclpy.shutdown()
        node.destroy_node()


if __name__ == '__main__':
    main()
