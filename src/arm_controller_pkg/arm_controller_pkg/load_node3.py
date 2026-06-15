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
        np.array([53.60, 23.71, 15.87, 3.85, 130.79, 0.0]),
        np.array([67.77, 1.24, 49.43, 4.35, 119.99, -19.94]),
    ],
    2: [
        np.array([-90.0, 0.0, 90.0, 0.0, 90.0, 0.0]),
        np.array([-90.0, -20.81, 107.71, 0.0, 93.11, 0.0]),
        np.array([-145.0, -20.81, 107.71, 0.0, 93.11, 0.0]),
        np.array([-220.0, -11.96, 57.40, 0.0, 100.40, 0.0]),
        np.array([-250.0, -11.96, 57.40, 0.0, 100.40, 0.0]),
        np.array([-265.26, 18.68, 28.51, -2.23, 125.87, 3.60]),
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
        np.array([-246.43, 19.80, 27.31, 1.84, 120.90, 24.45]),
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
        np.array([-260.92, 2.93, 47.34, 0.0, 129.73, 9.09]),
    ],
}

CAM_X_OFF = -51.0
CAM_Y_OFF = 32.0
Z_DOWN_MM = 20.0
Z_UP_MM = -20.0
Z_OFFSET = -85.0
Z_MARGIN = 40.0
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

        self.robot = None
        self.rc = None
        self.robot_ready = False

        try:
            self.robot = rb.Cobot(ROBOT_IP)
            self.rc = rb.ResponseCollector()
            self.robot.set_operation_mode(self.rc, rb.OperationMode.Real)
            self.robot.set_speed_bar(self.rc, 1.0)
            self.robot_ready = True
            self.get_logger().info('[LOAD] robot connected')
        except Exception as e:
            self.robot = None
            self.rc = None
            self.robot_ready = False
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

    # --- 상태 확인 헬퍼 ---

    def is_robot_ready(self):
        if not self.robot_ready or self.robot is None or self.rc is None:
            self.get_logger().error('[LOAD] robot is not connected')
            return False
        return True

    # --- 서비스 호출 헬퍼 ---

    def call_service(self, client, request, timeout=10.0):
        """Call a ROS2 service from inside callbacks without nested spinning.
        This node is expected to run under MultiThreadedExecutor with a
        ReentrantCallbackGroup. The current callback thread waits on an Event,
        while another executor thread can process the service response.
        """
        try:
            if not client.wait_for_service(timeout_sec=1.0):
                self.get_logger().error(f'[LOAD] service unavailable: {client.srv_name}')
                return None

            future = client.call_async(request)
            done_event = threading.Event()
            future.add_done_callback(lambda _: done_event.set())

            if not done_event.wait(timeout=timeout):
                self.get_logger().error(f'[LOAD] service timeout: {client.srv_name}')
                return None

            return future.result()
        except Exception as e:
            self.get_logger().error(f'[LOAD] service call failed: {client.srv_name}: {e}')
            return None

    def call_vision(self, target_color, retries=3):
        for i in range(retries):
            req = GetTargetPose.Request()
            req.target_color = target_color
            req.target_size = ""
            res = self.call_service(self.vision_client, req)
            if res and res.success:
                return res
            self.get_logger().warn(f'[LOAD] vision retry {i + 1}/{retries}')
            time.sleep(0.5)
        return None

    def call_gripper(self, grip: bool):
        client = self.gripper_grip_client if grip else self.gripper_open_client
        req = Trigger.Request()
        res = self.call_service(client, req, timeout=6.0)
        action_name = 'grip' if grip else 'open'
        if res and res.success:
            self.get_logger().info(f'[GRIPPER] {action_name}')
            return True
        self.get_logger().error(f'[GRIPPER] {action_name} failed')
        return False

    def call_cargo(self, action, slot=0, object_id=0):
        req = Cargo.Request()
        req.action = action
        req.slot = slot
        req.object_id = object_id
        return self.call_service(self.cargo_client, req)

    # --- 로봇 이동 헬퍼 ---

    def wait_move(self, timeout=10.0, label='move'):
        if not self.is_robot_ready():
            return False
        try:
            result = self.robot.wait_for_move_finished(self.rc, timeout=timeout)
            if result is False:
                self.get_logger().error(f'[LOAD] {label} wait returned False')
                return False
            return True
        except Exception as e:
            self.get_logger().error(f'[LOAD] {label} wait failed: {e}')
            return False
    def move_j_checked(self, joints_deg, label='move_j', timeout=10.0):
        if not self.is_robot_ready():
            return False
        try:
            self.robot.move_j(self.rc, joints_deg, J_VEL, J_ACC)
        except Exception as e:
            self.get_logger().error(f'[LOAD] {label} command failed: {e}')
            return False
        return self.wait_move(timeout=timeout, label=label)
    def move_l_rel_checked(self, delta, label='move_l_rel', timeout=10.0):
        if not self.is_robot_ready():
            return False
        try:
            self.robot.move_l_rel(
                self.rc,
                np.array(delta, dtype=float),
                L_VEL,
                L_ACC,
                rb.ReferenceFrame.Tool,
            )
        except Exception as e:
            self.get_logger().error(f'[LOAD] {label} command failed: {e}')
            return False
        return self.wait_move(timeout=timeout, label=label)
    def go_home(self):
        return self.move_j_checked(HOME_JOINT_DEG, label='go_home')

    def move_to_slot(self, slot):
        waypoints = SLOT_WAYPOINTS.get(slot)
        if waypoints is None:
            self.get_logger().error(f'[LOAD] no waypoints for slot={slot}')
            return False

        # 정방향 첫 번째 waypoint는 HOME_JOINT_DEG라서 스킵한다.
        move_waypoints = waypoints[1:]

        for idx, wp in enumerate(move_waypoints, start=2):
            if not self.move_j_checked(wp, label=f'move_to_slot({slot}) wp{idx}'):
                return False

        self.get_logger().info(f'[LOAD] slot={slot} reached')
        return True
    def return_from_slot(self, slot):
        waypoints = SLOT_WAYPOINTS.get(slot)
        if waypoints is None:
            self.get_logger().error(f'[LOAD] no waypoints for slot={slot}')
            return False

        # 역방향 첫 번째 waypoint는 방금 도착했던 슬롯 최종 자세라서 스킵한다.
        return_waypoints = list(reversed(waypoints))[1:]

        for idx, wp in enumerate(return_waypoints, start=2):
            if not self.move_j_checked(wp, label=f'return_from_slot({slot}) wp{idx}'):
                return False

        self.get_logger().info(f'[LOAD] returned from slot={slot}')
        return True
    def arm_command_cb(self, request, response):
        response.slots = []
        response.object_ids = []

        if request.action.upper() != 'LOAD':
            response.success = False
            response.message = f'unknown action: {request.action}'
            return response

        if not self.is_robot_ready():
            response.success = False
            response.message = 'robot not connected'
            return response

        with self._busy_lock:
            if self._busy:
                response.success = False
                response.message = 'busy'
                return response
            self._busy = True

        try:
            results = self.sequence_load_multi(list(request.object_ids))
            success_all = bool(results) and all(r['success'] for r in results)
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
        if not self.is_robot_ready():
            return {
                'success': False,
                'slot': -1,
                'object_id': object_id,
                'message': 'robot not connected',
            }

        target_color = MATERIAL_NAMES.get(object_id)
        if not target_color:
            self.get_logger().error(f'[LOAD] unknown object_id: {object_id}')
            return {
                'success': False,
                'slot': -1,
                'object_id': object_id,
                'message': f'unknown object_id={object_id}',
            }

        vision_target = str(object_id)

        self.get_logger().info(f'[LOAD START] object_id={object_id}, target={target_color}')

        # 1. 빈 슬롯 확인
        res = self.call_cargo('FIND_EMPTY', object_id=object_id)
        if not res or not res.success:
            self.get_logger().error('[LOAD] no empty slot')
            return {
                'success': False,
                'slot': -1,
                'object_id': object_id,
                'message': 'no empty slot',
            }
        slot = res.slot
        self.get_logger().info(f'[CARGO] empty slot: {slot}')

        # 2. 초기화
        if not self.call_gripper(False):
            return {
                'success': False,
                'slot': -1,
                'object_id': object_id,
                'message': 'initial gripper open failed',
            }

        if not self.go_home():
            return {
                'success': False,
                'slot': -1,
                'object_id': object_id,
                'message': 'go_home failed',
            }

        # 3. HOME 포즈에서 1회만 측정 (YAW / XY / Z 모두 이 값으로 처리)
        p = self.call_vision(vision_target)
        if not p:
            self.get_logger().error('[LOAD] vision failed')
            self.go_home()
            return {
                'success': False,
                'slot': -1,
                'object_id': object_id,
                'message': 'vision failed',
            }

        # 4. YAW + XY 동시 이동
        #    move_l_rel(Tool)의 병진 성분은 이동 시작(HOME) 프레임 기준으로 적용되므로,
        #    HOME에서 측정한 dx/dy를 그대로 쓰면서 yaw(rz)를 함께 넣으면
        #    끝점은 물체 중심, 자세는 yaw 정렬 상태가 된다. (재측정/보정 불필요)
        dx = -(p.x * 1000.0) + CAM_Y_OFF
        dy = (p.y * 1000.0) + CAM_X_OFF
        #    NOTE: p.yaw 단위는 deg. 실제로 손목이 반대로 돌거나 단위가 rad이면
        #          아래 rz 항(p.yaw)을 -p.yaw 또는 np.radians(p.yaw)로 조정할 것.
        if not self.move_l_rel_checked(
            [dy, dx, 0.0, 0.0, 0.0, p.yaw],
            label='yaw+xy correction',
        ):
            self.go_home()
            return {
                'success': False,
                'slot': -1,
                'object_id': object_id,
                'message': 'yaw+xy correction failed',
            }
        time.sleep(0.5)

        # 5. Z 하강 (HOME에서 측정한 p.z 사용)
        z_move = (p.z * 1000.0) + Z_OFFSET
        if not self.move_l_rel_checked(
            [0.0, 0.0, z_move - Z_MARGIN, 0.0, 0.0, 0.0],
            label='z approach',
        ):
            self.go_home()
            return {
                'success': False,
                'slot': -1,
                'object_id': object_id,
                'message': 'z approach failed',
            }

        if not self.move_l_rel_checked(
            [0.0, 0.0, Z_MARGIN, 0.0, 0.0, 0.0],
            label='z final approach',
        ):
            self.go_home()
            return {
                'success': False,
                'slot': -1,
                'object_id': object_id,
                'message': 'z final approach failed',
            }
        time.sleep(0.5)

        # 6. 그리퍼 grip
        if not self.call_gripper(True):
            self.get_logger().error('[LOAD] grip failed')
            self.move_l_rel_checked(
                [0.0, 0.0, -100.0, 0.0, 0.0, 0.0],
                label='retreat after grip failure',
            )
            self.go_home()
            return {
                'success': False,
                'slot': -1,
                'object_id': object_id,
                'message': 'grip failed',
            }

        # 7. Z 상승
        if not self.move_l_rel_checked(
            [0.0, 0.0, -50.0, 0.0, 0.0, 0.0],
            label='lift after grip',
        ):
            self.go_home()
            return {
                'success': False,
                'slot': -1,
                'object_id': object_id,
                'message': 'lift after grip failed',
            }

        # 8. 웨이포인트 순서대로 슬롯으로 이동
        if not self.move_to_slot(slot):
            self.get_logger().error('[LOAD] move to slot failed')
            self.go_home()
            return {
                'success': False,
                'slot': -1,
                'object_id': object_id,
                'message': 'move to slot failed',
            }

        # 9. Z 하강 -> open -> Z 상승
        if not self.move_l_rel_checked(
            [0.0, 0.0, Z_DOWN_MM, 0.0, 0.0, 0.0],
            label='place z down',
        ):
            self.go_home()
            return {
                'success': False,
                'slot': slot,
                'object_id': object_id,
                'message': 'place z down failed',
            }

        if not self.call_gripper(False):
            self.get_logger().error('[LOAD] final gripper open failed')
            self.move_l_rel_checked(
                [0.0, 0.0, Z_UP_MM, 0.0, 0.0, 0.0],
                label='retreat after open failure',
            )
            self.go_home()
            return {
                'success': False,
                'slot': slot,
                'object_id': object_id,
                'message': 'final gripper open failed',
            }

        if not self.move_l_rel_checked(
            [0.0, 0.0, Z_UP_MM, 0.0, 0.0, 0.0],
            label='place z up',
        ):
            self.go_home()
            return {
                'success': False,
                'slot': slot,
                'object_id': object_id,
                'message': 'place z up failed',
            }

        # 10. 웨이포인트 역순으로 홈 복귀
        if not self.return_from_slot(slot):
            return {
                'success': False,
                'slot': slot,
                'object_id': object_id,
                'message': 'return from slot failed',
            }

        # 11. 카고 기록
        res = self.call_cargo('SET', slot=slot, object_id=object_id)
        if not res or not res.success:
            self.get_logger().error('[LOAD] cargo SET failed')
            return {
                'success': False,
                'slot': slot,
                'object_id': object_id,
                'message': 'loaded physically but cargo SET failed',
            }

        self.get_logger().info(f'[LOAD DONE] object_id={object_id}, slot={slot}')
        return {
            'success': True,
            'slot': slot,
            'object_id': object_id,
            'message': 'load success',
        }

def main(args=None):
    rclpy.init(args=args)
    node = LoadNode()
    executor = MultiThreadedExecutor(num_threads=4)
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
