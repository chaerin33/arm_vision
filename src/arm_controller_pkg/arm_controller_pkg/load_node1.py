import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from arm_interfaces.srv import ArmCommand, Cargo, GetTargetPose
from std_srvs.srv import Trigger
import rbpodo as rb
import numpy as np
import time
import threading


ROBOT_IP = "10.0.2.8"

HOME_JOINT_DEG = np.array([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0])

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

CAM_X_OFF = -51.0
CAM_Y_OFF = 32.0
Z_DOWN_MM = 20.0
Z_UP_MM = -20.0
Z_OFFSET = -85.0
Z_MARGIN = 20.0
J_VEL, J_ACC = 255, 255
L_VEL, L_ACC = 500, 800

MATERIAL_NAMES = {
    1: "2x2_red",
    2: "2x2_green",
    3: "2x2_blue",
    4: "2x2_yellow",
    5: "4x2_red",
    6: "4x2_green",
    7: "4x2_blue",
    8: "4x2_yellow",
}


class LoadNode(Node):
    def __init__(self):
        super().__init__('load_node')
        self.cbg = ReentrantCallbackGroup()

        try:
            self.robot = rb.Cobot(ROBOT_IP)
            self.rc = rb.ResponseCollector()
            self.robot.set_operation_mode(self.rc, rb.OperationMode.Real)
            self.robot.set_speed_bar(self.rc, 1.0)
            self.get_logger().info('[LOAD] robot connected')
        except Exception as e:
            self.get_logger().error(f'[LOAD] robot connection error: {e}')

        self.vision_client = self.create_client(
            GetTargetPose, '/get_target_pose', callback_group=self.cbg)
        self.gripper_open_client = self.create_client(
            Trigger, '/gripper/open', callback_group=self.cbg)
        self.gripper_grip_client = self.create_client(
            Trigger, '/gripper/grip', callback_group=self.cbg)
        self.cargo_client = self.create_client(
            Cargo, '/cargo', callback_group=self.cbg)
        self.srv = self.create_service(
            ArmCommand, '/arm_command', self.arm_command_cb, callback_group=self.cbg)

        self._busy_lock = threading.Lock()
        self._busy = False

        self.get_logger().info('[LOAD] load_node started')

    # --- 서비스 호출 헬퍼 ---

    def call_service(self, client, request, timeout=10.0):
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout)
        if future.done():
            return future.result()
        self.get_logger().error(f'[LOAD] service timeout: {client.srv_name}')
        return None

    def call_vision(self, target_color, retries=3):
        for i in range(retries):
            req = GetTargetPose.Request()
            req.target_color = target_color
            req.target_size = ""
            res = self.call_service(self.vision_client, req)
            if res and res.success:
                return res
            self.get_logger().warn(f'[LOAD] vision retry {i+1}/{retries}')
            time.sleep(0.5)
        return None

    def call_gripper(self, grip: bool):
        client = self.gripper_grip_client if grip else self.gripper_open_client
        req = Trigger.Request()
        res = self.call_service(client, req, timeout=6.0)
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
        self.robot.wait_for_move_finished(self.rc, timeout=10.0)

    def wait_move(self):
        self.robot.wait_for_move_finished(self.rc, timeout=10.0)

    def move_to_slot(self, slot):
        waypoints = SLOT_WAYPOINTS.get(slot)
        if waypoints is None:
            self.get_logger().error(f'[LOAD] no waypoints for slot={slot}')
            return False
        for wp in waypoints:
            self.robot.move_j(self.rc, wp, J_VEL, J_ACC)
            self.wait_move()
        self.get_logger().info(f'[LOAD] slot={slot} reached')
        return True

    def return_from_slot(self, slot):
        waypoints = SLOT_WAYPOINTS.get(slot)
        if waypoints is None:
            self.get_logger().error(f'[LOAD] no waypoints for slot={slot}')
            return False
        for wp in reversed(waypoints):
            self.robot.move_j(self.rc, wp, J_VEL, J_ACC)
            self.wait_move()
        self.get_logger().info(f'[LOAD] returned from slot={slot}')
        return True

    # --- LOAD 시퀀스 ---

    def arm_command_cb(self, request, response):
        if request.action.upper() != 'LOAD':
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
            results = self.sequence_load_multi(list(request.object_ids))
            success_all = all(r['success'] for r in results)
            response.success = success_all
            response.slots = [r['slot'] for r in results]
            response.object_ids = [r['object_id'] for r in results]
            response.message = ', '.join(r['message'] for r in results)
        except Exception as e:
            self.get_logger().error(f'[LOAD] exception: {e}')
            response.success = False
            response.slots = []
            response.object_ids = []
            response.message = str(e)
        finally:
            with self._busy_lock:
                self._busy = False

        return response

    def sequence_load_multi(self, object_ids):
        results = []
        for object_id in object_ids:
            result = self.sequence_load(object_id)
            results.append(result)
            if not result['success']:
                self.get_logger().error(f'[LOAD] failed at object_id={object_id}, stopping')
                break
        return results

    def sequence_load(self, object_id):
        target_color = MATERIAL_NAMES.get(object_id)
        if not target_color:
            self.get_logger().error(f'[LOAD] unknown object_id: {object_id}')
            return {'success': False, 'slot': -1, 'object_id': object_id, 'message': f'unknown object_id={object_id}'}

        self.get_logger().info(f'[LOAD START] object_id={object_id}, target={target_color}')

        # 1. 빈 슬롯 확인
        res = self.call_cargo('FIND_EMPTY', object_id=object_id)
        if not res or not res.success:
            self.get_logger().error('[LOAD] no empty slot')
            return {'success': False, 'slot': -1, 'object_id': object_id, 'message': 'no empty slot'}
        slot = res.slot
        self.get_logger().info(f'[CARGO] empty slot: {slot}')

        # 2. 초기화
        self.call_gripper(False)
        self.go_home()

        # 3. YAW 보정
        p = self.call_vision(target_color)
        if not p:
            self.get_logger().error('[LOAD] vision failed at YAW step')
            return {'success': False, 'slot': -1, 'object_id': object_id, 'message': 'vision failed at YAW'}

        if abs(p.yaw) >= 0.01:
            target_j = HOME_JOINT_DEG.copy()
            target_j[5] += p.yaw
            self.robot.move_j(self.rc, target_j, J_VEL, J_ACC)
            self.wait_move()
            time.sleep(0.5)

        # 4. XY 이동 (YAW 보정 후 재측정)
        p = self.call_vision(target_color)
        if not p:
            self.get_logger().error('[LOAD] vision failed at XY step')
            self.go_home()
            return {'success': False, 'slot': -1, 'object_id': object_id, 'message': 'vision failed at XY'}

        dx = -(p.x * 1000.0) + CAM_Y_OFF
        dy = (p.y * 1000.0) + CAM_X_OFF
        self.robot.move_l_rel(
            self.rc, np.array([dy, dx, 0.0, 0.0, 0.0, 0.0]),
            L_VEL, L_ACC, rb.ReferenceFrame.Tool)
        self.wait_move()
        time.sleep(0.5)

        # 5. Z 하강 (XY 보정 후 재측정)
        p = self.call_vision(target_color)
        if not p:
            self.get_logger().error('[LOAD] vision failed at Z step')
            self.go_home()
            return {'success': False, 'slot': -1, 'object_id': object_id, 'message': 'vision failed at Z'}

        z_move = (p.z * 1000.0) + Z_OFFSET
        self.robot.move_l_rel(
            self.rc, np.array([0.0, 0.0, z_move - Z_MARGIN, 0.0, 0.0, 0.0]),
            L_VEL, L_ACC, rb.ReferenceFrame.Tool)
        self.wait_move()
        self.robot.move_l_rel(
            self.rc, np.array([0.0, 0.0, Z_MARGIN, 0.0, 0.0, 0.0]),
            L_VEL, L_ACC, rb.ReferenceFrame.Tool)
        self.wait_move()
        time.sleep(0.5)

        # 6. 그리퍼 grip
        if not self.call_gripper(True):
            self.get_logger().error('[LOAD] grip failed')
            self.robot.move_l_rel(
                self.rc, np.array([0.0, 0.0, -100.0, 0.0, 0.0, 0.0]),
                L_VEL, L_ACC, rb.ReferenceFrame.Tool)
            self.wait_move()
            self.go_home()
            return {'success': False, 'slot': -1, 'object_id': object_id, 'message': 'grip failed'}

        # 7. Z 상승 
        self.robot.move_l_rel(
            self.rc, np.array([0.0, 0.0, -50.0, 0.0, 0.0, 0.0]),
            L_VEL, L_ACC, rb.ReferenceFrame.Tool)
        self.wait_move()

        # 8. 웨이포인트 순서대로 슬롯으로 이동
        if not self.move_to_slot(slot):
            self.get_logger().error(f'[LOAD] move to slot failed')
            self.go_home()
            return {'success': False, 'slot': -1, 'object_id': object_id, 'message': 'move to slot failed'}

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
        self.return_from_slot(slot)

        # 11. 카고 기록
        self.call_cargo('SET', slot=slot, object_id=object_id)
        self.get_logger().info(f'[LOAD DONE] object_id={object_id}, slot={slot}')
        return {'success': True, 'slot': slot, 'object_id': object_id, 'message': 'load success'}


def main(args=None):
    rclpy.init(args=args)
    node = LoadNode()
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