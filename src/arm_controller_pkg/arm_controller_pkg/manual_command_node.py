import rclpy
from rclpy.node import Node
from arm_interfaces.srv import ArmCommand


class ManualCommandNode(Node):
    def __init__(self):
        super().__init__('manual_command_node')
        self.client = self.create_client(ArmCommand, '/arm_command')
        self.get_logger().info('[MANUAL] manual_command_node started')
        self.get_logger().info('[MANUAL] commands: load / unload / status / exit')

    def call_arm(self, action, object_id, location):
        req = ArmCommand.Request()
        req.action = action
        req.object_id = object_id
        req.location = location

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('[MANUAL] waiting for /arm_command service...')

        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()

        if res:
            self.get_logger().info(f'[MANUAL] result: success={res.success}, message={res.message}')
        else:
            self.get_logger().error('[MANUAL] no response')

    def run(self):
        print('\n' + '=' * 40)
        print('ARM Manual Command Node')
        print('=' * 40)
        print('object_id: 1=2x2_red  2=2x2_green  3=2x2_blue  4=2x2_yellow')
        print('           5=4x2_red  6=4x2_green  7=4x2_blue  8=4x2_yellow')
        print('location:  STORAGE / WORKBENCH / CUSTOMER')
        print('=' * 40)

        while rclpy.ok():
            try:
                cmd = input('\ncommand (load/unload/exit): ').strip().lower()

                if cmd == 'exit':
                    break

                elif cmd == 'load':
                    object_id = int(input('object_id (1-8): ').strip())
                    location = input('location (STORAGE): ').strip().upper() or 'STORAGE'
                    self.call_arm('LOAD', object_id, location)

                elif cmd == 'unload':
                    object_id = int(input('object_id (1-8): ').strip())
                    location = input('location (WORKBENCH/CUSTOMER): ').strip().upper()
                    self.call_arm('UNLOAD', object_id, location)

                elif cmd == '':
                    continue

                else:
                    print(f'unknown command: {cmd}')

            except ValueError:
                print('[ERROR] invalid input')
            except KeyboardInterrupt:
                break


def main(args=None):
    rclpy.init(args=args)
    node = ManualCommandNode()
    try:
        node.run()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()