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

# 슬롯별 웨이포인트 (joint, degree) - 실측 후 채워넣기
# 첫 번째 포인트는 반드시 HOME_JOINT_DEG와 동일해야 역순 복귀 시 홈에 도달함
SLOT_WAYPOINTS = {
    1: [
        np.array([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0]),
        np.array([-90.0, -20.81, 107.71, 0.0, 93.11, 0.0]),
        np.array([-35.0, -20.81, 107.71, 0.0, 93.11, 0.0]),
        np.array([ 53.60, 23.71, 15.87, 3.85, 130.79, 0.0]),
        np.array([ 67.77, 1.24, 49.43, 4.35, 119.99, -19.94]),
    ],
    2: [
        np.array([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0]),
        np.array([-90.0, -20.81, 107.71, 0.0, 93.11, 0.0]),
        np.array([-145.0, -20.81, 107.71, 0.0, 93.11, 0.0]),
        np.array([-220.0, -11.96, 57.40, 0.0, 100.40, 0.0]),
        np.array([-250.0, -11.96, 57.40, 0.0, 100.40, 0.0]),
        np.array([-265.26, 18.68, 28.51, -2.23, 125.87, 3.60])
    ],
    3: [
        np.array([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0]),
        np.array([-90.0, -20.81, 107.71, 0.0, 93.11, 0.0]),
        np.array([-145.0, -20.81, 107.71, 0.0, 93.11, 0.0]),
        np.array([-220.0, -11.96, 57.40, 0.0, 100.40, 0.0]),
        np.array([-250.0, -11.96, 57.40, 0.0, 100.40, 0.0]),
        np.array([-253.19, 22.98, 22.45, -4.08, 128.11, 14.39]),
        ],
    4: [
        np.array([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0]),
        np.array([-90.0, -20.81, 107.71, 0.0, 93.11, 0.0]),
        np.array([-145.0, -20.81, 107.71, 0.0, 93.11, 0.0]),
        np.array([-220.0, -11.96, 57.40, 0.0, 100.40, 0.0]),
        np.array([-250.0, -11.96, 57.40, 0.0, 100.40, 0.0]),
        np.array([-233.56, 1.26, 52.33, -18.50, 98.90, 28.90]),
        ],
    5: [
        np.array([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0]),
        np.array([-90.0, -20.81, 107.71, 0.0, 93.11, 0.0]),
        np.array([-145.0, -20.81, 107.71, 0.0, 93.11, 0.0]),
        np.array([-220.0, -11.96, 57.40, 0.0, 100.40, 0.0]),
        np.array([-243.08, 9.11, 40.45, 0.0, 130.43, 26.93]),
    ],
    6: [
        np.array([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0]),
        np.array([-90.0, -20.81, 107.71, 0.0, 93.11, 0.0]),
        np.array([-145.0, -20.81, 107.71, 0.0, 93.11, 0.0]),
        np.array([-220.0, -11.96, 57.40, 0.0, 100.40, 0.0]),
        np.array([-260.92, 2.93, 47.34, 0, 129.73, 9.09]),
    ],
}

# 인덱스 0~5: 내려놓는 순서에 따라 사용
DELIVERY_WAYPOINTS = {
    0: [
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),  # 실측 후 채우기
    ],
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
        waypoints = SLOT_WAYPOINTS.get(slot)
        if waypoints is None:
            self.get_logger().error(f'[UNLOAD] no waypoints for slot={slot}')
            return False
        for wp in waypoints:
            self.robot.move_j(self.rc, wp, J_VEL, J_ACC)
            self.wait_move()
        self.get_logger().info(f'[UNLOAD] slot={slot} reached')
        return True

    def return_from_slot(self, slot):
        waypoints = SLOT_WAYPOINTS.get(slot)
        if waypoints is None:
            self.get_logger().error(f'[UNLOAD] no waypoints for slot={slot}')
            return False
        for wp in reversed(waypoints):
            self.robot.move_j(self.rc, wp, J_VEL, J_ACC)
            self.wait_move()
        self.get_logger().info(f'[UNLOAD] returned from slot={slot}')
        return True

    def move_to_delivery(self, delivery_idx):
        waypoints = DELIVERY_WAYPOINTS.get(delivery_idx)
        if waypoints is None:
            self.get_logger().error(f'[UNLOAD] no waypoints for delivery_idx={delivery_idx}')
            return False
        for wp in waypoints:
            self.robot.move_j(self.rc, wp, J_VEL, J_ACC)
            self.wait_move()
        self.get_logger().info(f'[UNLOAD] delivery position {delivery_idx} reached')
        return True

    def return_from_delivery(self, delivery_idx):
        waypoints = DELIVERY_WAYPOINTS.get(delivery_idx)
        if waypoints is None:
            self.get_logger().error(f'[UNLOAD] no waypoints for delivery_idx={delivery_idx}')
            return False
        for wp in reversed(waypoints):
            self.robot.move_j(self.rc, wp, J_VEL, J_ACC)
            self.wait_move()
        self.get_logger().info(f'[UNLOAD] returned from delivery position {delivery_idx}')
        return True

    # --- UNLOAD 시퀀스 ---

    def arm_command_cb(self, request, response):
        if request.action.upper() != 'UNLOAD':
            response.success = False
            response.slots = []
            response.object_ids = []
            response.message = f'unknown action: {request.action}'
            return response

        with self._busy_lock:
            if self._busy:
                response.success = False
                response.slots = []
                response.object_ids = []
                response.message = 'busy'
                return response
            self._busy = True

        try:
            results = self.sequence_unload_multi(list(request.object_ids))
            success_all = all(r['success'] for r in results)
            response.success = success_all
            response.slots = [r['slot'] for r in results]
            response.object_ids = [r['object_id'] for r in results]
            response.message = ', '.join(r['message'] for r in results)
        except Exception as e:
            self.get_logger().error(f'[UNLOAD] exception: {e}')
            response.success = False
            response.slots = []
            response.object_ids = []
            response.message = str(e)
        finally:
            with self._busy_lock:
                self._busy = False

        return response

    def sequence_unload_multi(self, object_ids):
        results = []
        for idx, object_id in enumerate(object_ids):
            result = self.sequence_unload(object_id, idx)
            results.append(result)
            if not result['success']:
                self.get_logger().error(f'[UNLOAD] failed at object_id={object_id}, stopping')
                break
        return results

    def sequence_unload(self, object_id, delivery_idx):
        self.get_logger().info(f'[UNLOAD START] object_id={object_id}, delivery_idx={delivery_idx}')

        # 1. 슬롯 확인
        res = self.call_cargo('FIND_OBJECT', object_id=object_id)
        if not res or not res.success:
            self.get_logger().error(f'[UNLOAD] object_id={object_id} not found in cargo')
            return {'success': False, 'slot': -1, 'object_id': object_id, 'message': f'object not found: {object_id}'}
        slot = res.slot
        self.get_logger().info(f'[CARGO] object found: slot={slot}')

        # 2. 초기화
        self.call_gripper(False)
        self.go_home()

        # 3. 웨이포인트 순서대로 슬롯으로 이동
        if not self.move_to_slot(slot):
            self.go_home()
            return {'success': False, 'slot': -1, 'object_id': object_id, 'message': 'move to slot failed'}

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
            self.return_from_slot(slot)
            return {'success': False, 'slot': -1, 'object_id': object_id, 'message': 'grip failed'}

        # 6. Z 상승
        self.robot.move_l_rel(
            self.rc, np.array([0.0, 0.0, Z_UP_MM, 0.0, 0.0, 0.0]),
            L_VEL, L_ACC, rb.ReferenceFrame.Tool)
        self.wait_move()

        # 7. 웨이포인트 역순으로 홈 복귀
        self.return_from_slot(slot)

        # 8. 배달 위치로 이동 (delivery_idx번째 자리)
        if not self.move_to_delivery(delivery_idx):
            self.go_home()
            return {'success': False, 'slot': -1, 'object_id': object_id, 'message': 'move to delivery failed'}

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

        # 10. 웨이포인트 역순으로 홈 복귀
        self.return_from_delivery(delivery_idx)

        # 11. 카고 기록 삭제
        self.call_cargo('CLEAR', slot=slot)
        self.get_logger().info(f'[UNLOAD DONE] object_id={object_id}, slot={slot}, delivery_idx={delivery_idx}')
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