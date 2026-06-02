import rclpy
from rclpy.node import Node
from arm_interfaces.srv import GetTargetPose
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO
import cv2
import time


class VisionNode(Node):
    def __init__(self):
        super().__init__('vision_node')
        self.srv = self.create_service(GetTargetPose, '/get_target_pose', self.get_pose_cb)

        MODEL_DET = "/home/scr/robocup/arm_ws/src/vision_pkg/vision_pkg/models/best.pt"
        MODEL_SEG = "/home/scr/robocup/arm_ws/src/vision_pkg/vision_pkg/models/best_old.pt"
        self.model_det = YOLO(MODEL_DET)
        self.model_seg = YOLO(MODEL_SEG)

        self.get_logger().info(f'[VISION] detect model task: {self.model_det.task}')
        self.get_logger().info(f'[VISION] segment model task: {self.model_seg.task}')

        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        self.intrinsics = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()

        self.latest_color = None
        self.latest_depth = None

        self.create_timer(0.033, self.visualize_cb)
        self.get_logger().info('[VISION] vision_node started (ensemble mode)')

    def calculate_yaw(self, rect):
        (cx, cy), (w, h), angle = rect
        if w < h:
            yaw = angle
        else:
            yaw = angle + 90.0
        if yaw > 90:
            yaw -= 180
        if yaw < -90:
            yaw += 180
        return yaw

    def get_valid_depth(self, depth_frame, u, v, search_radius=10):
        z = depth_frame.get_distance(u, v)
        if z > 0:
            return z
        for r in range(1, search_radius + 1):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    nu, nv = u + dx, v + dy
                    if 0 <= nu < 640 and 0 <= nv < 480:
                        z = depth_frame.get_distance(nu, nv)
                        if z > 0:
                            return z
        return 0.0

    def match_segment(self, res_seg, u, v):
        best_mask_pts = None
        min_dist = float('inf')
        if res_seg.masks is None or res_seg.boxes is None:
            return None
        for j, seg_box in enumerate(res_seg.boxes):
            s_xyxy = seg_box.xyxy[0].cpu().numpy()
            s_u = int((s_xyxy[0] + s_xyxy[2]) / 2)
            s_v = int((s_xyxy[1] + s_xyxy[3]) / 2)
            dist = ((u - s_u) ** 2 + (v - s_v) ** 2) ** 0.5
            if dist < 40 and dist < min_dist:
                min_dist = dist
                if len(res_seg.masks.xy) > j:
                    best_mask_pts = np.int32(res_seg.masks.xy[j])
        return best_mask_pts

    def calc_yaw_from_mask(self, mask_pts):
        if mask_pts is None or len(mask_pts) < 3:
            return 0.0
        M = cv2.moments(mask_pts)
        if M["m00"] == 0:
            return 0.0
        rect = cv2.minAreaRect(mask_pts)
        return self.calculate_yaw(rect)

    def visualize_cb(self):
        try:
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
            aligned = self.align.process(frames)
            self.latest_depth = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()

            if not color_frame or not self.latest_depth:
                return

            self.latest_color = np.asanyarray(color_frame.get_data())
            res_det = self.model_det(self.latest_color, verbose=False)[0]
            res_seg = self.model_seg(self.latest_color, verbose=False)[0]

            display_img = res_det.plot()
            cv2.circle(display_img, (320, 240), 5, (0, 0, 255), -1)

            if res_det.boxes is not None:
                for box in res_det.boxes:
                    xyxy = box.xyxy[0].cpu().numpy()
                    u = int((xyxy[0] + xyxy[2]) / 2)
                    v = int((xyxy[1] + xyxy[3]) / 2)

                    mask_pts = self.match_segment(res_seg, u, v)
                    yaw = self.calc_yaw_from_mask(mask_pts)

                    z = self.latest_depth.get_distance(u, v)
                    if z > 0:
                        x, y, _ = rs.rs2_deproject_pixel_to_point(self.intrinsics, [u, v], z)
                        cv2.putText(display_img, f"X:{x*1000:.1f} Y:{y*1000:.1f}",
                                    (u - 60, v + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(display_img, f"Yaw:{yaw:.1f}",
                                    (u - 60, v + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

            cv2.imshow("Vision Node", display_img)
            cv2.waitKey(1)
        except Exception:
            pass

    def get_pose_cb(self, request, response):
        target = request.target_color.lower()

        if target.startswith("count_"):
            search_color = target.replace("count_", "")
            start_time = time.time()
            max_count = 0
            while time.time() - start_time < 0.5:
                try:
                    frames = self.pipeline.wait_for_frames(timeout_ms=500)
                    aligned = self.align.process(frames)
                    color_f = aligned.get_color_frame()
                    if not color_f:
                        continue
                    img = np.asanyarray(color_f.get_data())
                    results = self.model_det(img, verbose=False)[0]
                    if results.boxes is not None:
                        count = sum(
                            1 for box in results.boxes
                            if search_color in results.names[int(box.cls[0])].lower()
                        )
                        max_count = max(max_count, count)
                except Exception:
                    pass
            response.success = True
            response.x = float(max_count)
            response.y = 0.0
            response.z = 0.0
            response.yaw = 0.0
            return response

        self.get_logger().info(f'[VISION] target: {target}')

        samples = []
        start_time = time.time()

        while time.time() - start_time < 1.2:
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=500)
                aligned = self.align.process(frames)
                depth_f = aligned.get_depth_frame()
                color_f = aligned.get_color_frame()

                if not color_f or not depth_f:
                    continue

                img = np.asanyarray(color_f.get_data())
                res_det = self.model_det(img, verbose=False)[0]
                res_seg = self.model_seg(img, verbose=False)[0]

                if res_det.boxes is None:
                    continue

                frame_targets = []
                for box in res_det.boxes:
                    cls_name = res_det.names[int(box.cls[0])].lower()
                    if target not in cls_name:
                        continue

                    xyxy = box.xyxy[0].cpu().numpy()
                    u = int((xyxy[0] + xyxy[2]) / 2)
                    v = int((xyxy[1] + xyxy[3]) / 2)

                    mask_pts = self.match_segment(res_seg, u, v)
                    yaw = self.calc_yaw_from_mask(mask_pts)

                    z = self.get_valid_depth(depth_f, u, v)
                    if z > 0:
                        dist = ((u - 320) ** 2 + (v - 240) ** 2) ** 0.5
                        frame_targets.append({'u': u, 'v': v, 'z': z, 'yaw': yaw, 'dist': dist})

                if frame_targets:
                    max_z = max(t['z'] for t in frame_targets)
                    ground_targets = [t for t in frame_targets if abs(max_z - t['z']) < 0.015]
                    if ground_targets:
                        best = min(ground_targets, key=lambda t: t['dist'])
                        x, y, z = rs.rs2_deproject_pixel_to_point(
                            self.intrinsics, [best['u'], best['v']], best['z'])
                        samples.append([x, y, z, best['yaw']])

                time.sleep(0.01)
            except Exception:
                continue

        if len(samples) < 5:
            self.get_logger().error('[VISION] recognition failed: not enough frames')
            response.success = False
            return response

        samples = np.array(samples)
        median = np.median(samples, axis=0)

        response.success = True
        response.x = float(median[0])
        response.y = float(median[1])
        response.z = float(median[2])
        response.yaw = float(median[3])

        self.get_logger().info(
            f'[VISION] result: x={response.x*1000:.1f} y={response.y*1000:.1f} yaw={response.yaw:.1f}')
        return response


def main(args=None):
    rclpy.init(args=args)
    node = VisionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()