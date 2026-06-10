import rclpy
from rclpy.node import Node
from arm_interfaces.srv import ArmCommand


class ManualCommandNode(Node):
    def __init__(self):
        super().__init__('manual_command_node')
        self.client = self.create_client(ArmCommand, '/arm_command')
        self.get_logger().info('[MANUAL] manual_command_node started')
        self.get_logger().info('[MANUAL] commands: load / unload / exit')

    def call_arm(self, action, object_ids, location=''):
        req = ArmCommand.Request()
        req.action = action
        req.object_ids = object_ids

        # 현재 load/unload 노드는 location 값을 사용하지 않는다.
        # 물건을 항상 같은 위치에 내려놓을 것이므로 빈 문자열로 고정한다.
        req.location = location

        while not self.client.wait_for_service(timeout_sec=1.0):
            self.get_logger().warn('[MANUAL] waiting for /arm_command service...')

        future = self.client.call_async(req)
        rclpy.spin_until_future_complete(self, future)
        res = future.result()

        if res:
            self.get_logger().info(
                f'[MANUAL] result: success={res.success}, '
                f'slots={list(res.slots)}, object_ids={list(res.object_ids)}, '
                f'message={res.message}'
            )
        else:
            self.get_logger().error('[MANUAL] no response')

    def parse_object_ids(self, raw):
        object_ids = [int(x.strip()) for x in raw.split(',') if x.strip()]
        if not object_ids:
            raise ValueError('empty object_id list')
        return object_ids

    def run(self):
        print('\n' + '=' * 40)
        print('ARM Manual Command Node')
        print('=' * 40)
        print('object_id: 1=2x2_red  2=2x2_green  3=2x2_blue  4=2x2_yellow')
        print('           5=4x2_red  6=4x2_green  7=4x2_blue  8=4x2_yellow')
        print('여러 개 입력 시 쉼표로 구분: 1,3,5')
        print('unload는 location 입력 없이 항상 같은 위치로 내려놓음')
        print('=' * 40)

        while rclpy.ok():
            try:
                cmd = input('\ncommand (load/unload/exit): ').strip().lower()

                if cmd == 'exit':
                    break

                elif cmd == 'load':
                    raw = input('object_id (쉼표 구분, 예: 1,3,5): ').strip()
                    object_ids = self.parse_object_ids(raw)
                    self.call_arm('LOAD', object_ids)

                elif cmd == 'unload':
                    raw = input('object_id (쉼표 구분, 예: 1,3): ').strip()
                    object_ids = self.parse_object_ids(raw)
                    self.call_arm('UNLOAD', object_ids)

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
