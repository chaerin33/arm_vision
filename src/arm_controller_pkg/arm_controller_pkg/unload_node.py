import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from arm_interfaces.srv import ArmCommand, Cargo
from std_srvs.srv import SetBool
import rbpodo as rb
import numpy as np
import time
import threading


ROBOT_IP = "10.0.2.8"

HOME_JOINT_DEG = np.array([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0])

# TCP 좌표 [x, y, z, rx, ry, rz] (mm, degree) - 실측 후 채워넣기
SLOT_ITPL = {
    1: [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ],
    2: [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ],
    3: [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ],
    4: [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ],
    5: [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ],
    6: [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ],
}

# TCP 좌표 [x, y, z, rx, ry, rz] (mm, degree) - 실측 후 채워넣기
DESTINATION_ITPL = {
    "WORKBENCH": [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ],
    "CUSTOMER": [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ],
}

Z_DOWN_MM = 20.0
Z_UP_MM = -20.0
J_VEL, J_ACC = 255, 255
L_VEL, L_ACC = 500, 800


class UnloadNode(Node):
    def __init__(self):
        super().__init__('unload_node')
        self.cbg = ReentrantCallbackGroup()

        try:
            self.robot = rb.Cobot(ROBOT_IP)
            self.rc = rb.ResponseCollector()
            self.robot.set_operation_mode(self.rc, rb.OperationMode.Real)
            self.robot.set_speed_bar(self.rc, 1.0)
            self.get_logger().info('[UNLOAD] robot connected')
        except Exception as e:
            self.get_logger().error(f'[UNLOAD] robot connection error: {e}')

        self.gripper_client = self.create_client(
            SetBool, '/gripper_control', callback_group=self.cbg)
        self.cargo_client = self.create_client(
            Cargo, '/cargo', callback_group=self.cbg)
        self.srv = self.create_service(
            ArmCommand, '/arm_command', self.arm_command_cb, callback_group=self.cbg)

        self._busy_lock = threading.Lock()
        self._busy = False

        self.get_logger().info('[UNLOAD] unload_node started')

    # --- 서비스 호출 헬퍼 ---

    def call_service(self, client, request, timeout=10.0):
        future = client.call_async(request)
        start = time.time()
        while rclpy.ok() and (time.time() - start < timeout):
            if future.done():
                return future.result()
            time.sleep(0.05)
        self.get_logger().error(f'[UNLOAD] service timeout: {client.srv_name}')
        return None

    def call_gripper(self, grip: bool):
        req = SetBool.Request()
        req.data = grip
        res = self.call_service(self.gripper_client, req, timeout=6.0)
        if res and res.success:
            self.get_logger().info(f'[GRIPPER] {"grip" if grip else "open"}')
            return True
        self.get_logger().error('[GRIPPER] failed')
        return False

    def call_cargo(self, action, slot=0, object_id=0):
        req = Cargo.Request()
        req.action = action
        req.slot = slot
        req.object_id = object_id
        return self.call_service(self.cargo_client, req)

    # --- 로봇 이동 ---

    def go_home(self):
        self.robot.move_j(self.rc, HOME_JOINT_DEG, J_VEL, J_ACC)
        self.robot.wait_for_move_finished(self.rc)

    def wait_move(self):
        self.robot.wait_for_move_finished(self.rc)

    def move_to_slot(self, slot):
        waypoints = SLOT_ITPL.get(slot)
        if not waypoints:
            self.get_logger().error(f'[UNLOAD] no waypoints for slot={slot}')
            return False
        self.robot.move_itpl_clear(self.rc)
        for wp in waypoints:
            self.robot.move_itpl_add(self.rc, wp, L_VEL)
        self.robot.move_itpl_run(self.rc, L_ACC, rb.MoveITPLOption.Intended)
        self.wait_move()
        self.get_logger().info(f'[UNLOAD] slot={slot} reached')
        return True

    def move_to_destination(self, location):
        waypoints = DESTINATION_ITPL.get(location.upper())
        if not waypoints:
            self.get_logger().error(f'[UNLOAD] no waypoints for location={location}')
            return False
        self.robot.move_itpl_clear(self.rc)
        for wp in waypoints:
            self.robot.move_itpl_add(self.rc, wp, L_VEL)
        self.robot.move_itpl_run(self.rc, L_ACC, rb.MoveITPLOption.Intended)
        self.wait_move()
        self.get_logger().info(f'[UNLOAD] {location} reached')
        return True

    # --- UNLOAD 시퀀스 ---

    def arm_command_cb(self, request, response):
        if request.action.upper() != 'UNLOAD':
            response.success = False
            response.slot = -1
            response.object_id = -1
            response.message = f'unknown action: {request.action}'
            return response

        with self._busy_lock:
            if self._busy:
                response.success = False
                response.slot = -1
                response.object_id = -1
                response.message = 'busy'
                return response
            self._busy = True

        try:
            result = self.sequence_unload(request.object_id, request.location)
            response.success = result['success']
            response.slot = result['slot']
            response.object_id = result['object_id']
            response.message = result['message']
        except Exception as e:
            self.get_logger().error(f'[UNLOAD] exception: {e}')
            response.success = False
            response.slot = -1
            response.object_id = -1
            response.message = str(e)
        finally:
            with self._busy_lock:
                self._busy = False

        return response

    def sequence_unload(self, object_id, location):
        self.get_logger().info(f'[UNLOAD START] object_id={object_id}, location={location}')

        # 1. 슬롯 확인
        res = self.call_cargo('FIND_OBJECT', object_id=object_id)
        if not res or not res.success:
            self.get_logger().error(f'[UNLOAD] object_id={object_id} not found in cargo')
            return {'success': False, 'slot': -1, 'object_id': -1, 'message': 'object not found'}
        slot = res.slot
        self.get_logger().info(f'[CARGO] object found: slot={slot}')

        # 2. 초기화
        self.call_gripper(False)
        self.go_home()

        # 3. 슬롯으로 ITPL 이동
        if not self.move_to_slot(slot):
            self.go_home()
            return {'success': False, 'slot': -1, 'object_id': -1, 'message': 'move to slot failed'}

        # 4. Z 하강
        self.robot.move_l_rel(
            self.rc, np.array([0.0, 0.0, Z_DOWN_MM, 0.0, 0.0, 0.0]),
            L_VEL, L_ACC, rb.ReferenceFrame.Tool)
        self.wait_move()

        # 5. 그리퍼 grip
        if not self.call_gripper(True):
            self.get_logger().error('[UNLOAD] grip failed')
            self.robot.move_l_rel(
                self.rc, np.array([0.0, 0.0, -100.0, 0.0, 0.0, 0.0]),
                L_VEL, L_ACC, rb.ReferenceFrame.Tool)
            self.wait_move()
            self.go_home()
            return {'success': False, 'slot': -1, 'object_id': -1, 'message': 'grip failed'}

        # 6. Z 상승
        self.robot.move_l_rel(
            self.rc, np.array([0.0, 0.0, Z_UP_MM, 0.0, 0.0, 0.0]),
            L_VEL, L_ACC, rb.ReferenceFrame.Tool)
        self.wait_move()

        # 7. 홈 복귀
        self.go_home()

        # 8. 목적지로 ITPL 이동
        if not self.move_to_destination(location):
            self.go_home()
            return {'success': False, 'slot': -1, 'object_id': -1, 'message': 'move to destination failed'}

        # 9. Z 하강 → open → Z 상승
        self.robot.move_l_rel(
            self.rc, np.array([0.0, 0.0, Z_DOWN_MM, 0.0, 0.0, 0.0]),
            L_VEL, L_ACC, rb.ReferenceFrame.Tool)
        self.wait_move()
        self.call_gripper(False)
        self.robot.move_l_rel(
            self.rc, np.array([0.0, 0.0, Z_UP_MM, 0.0, 0.0, 0.0]),
            L_VEL, L_ACC, rb.ReferenceFrame.Tool)
        self.wait_move()

        # 10. 홈 복귀
        self.go_home()

        # 11. 카고 기록 삭제
        self.call_cargo('CLEAR', slot=slot)
        self.get_logger().info(f'[UNLOAD DONE] object_id={object_id}, slot={slot}, location={location}')
        return {'success': True, 'slot': slot, 'object_id': object_id, 'message': 'unload success'}


def main(args=None):
    rclpy.init(args=args)
    node = UnloadNode()
    executor = MultiThreadedExecutor()
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()