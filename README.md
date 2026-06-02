# ARM Controller & Vision - ROS2 Package

RoboCup SML 대회용 AMR 탑재 로봇팔 제어 시스템. AMR이 목적지에 도착하면 Master로부터 명령을 받아 물체를 집어 적재함에 싣거나, 적재함에서 꺼내 목적지에 내려놓는다.

---

## 패키지 구조

**arm_interfaces** — 커스텀 srv 정의 (CMake)
**arm_controller_pkg** — 로봇팔 제어 노드 (Python)
**vision_pkg** — 카메라 + YOLO 인식 노드 (Python)

---

## 노드 설명

**cargo_manager_node** — 적재함 슬롯 상태 관리. `/cargo` 서비스로 슬롯 조회/수정. slot 1은 완제품, slot 2~6은 부품 전용.

**gripper_node** — 아두이노 시리얼 통신으로 그리퍼 제어. `/gripper_control` 서비스.

**load_node** — 물체를 집어 슬롯에 적재. `/arm_command` 수신 후 vision → 이동 → grip → 슬롯 적재 순서로 동작.

**unload_node** — 슬롯에서 물체를 꺼내 목적지에 하역. `/arm_command` 수신 후 슬롯 이동 → grip → 목적지 이동 → open 순서로 동작.

**manual_command_node** — 터미널에서 직접 load/unload 명령 입력. AMR 없이 단독 테스트용.

**vision_node** — RealSense + YOLO 앙상블로 물체 6D 포즈 반환. `/get_target_pose` 서비스.

---

## 서비스 구조

```
Master / manual_command_node
        │ /arm_command
        ▼
  load_node / unload_node
        ├─ /get_target_pose ──► vision_node
        ├─ /gripper_control ──► gripper_node
        └─ /cargo ───────────► cargo_manager_node
```

---

## 주의사항

- 로봇 IP, 그리퍼 시리얼 포트, 모델 파일 경로는 환경에 맞게 수정 필요
- `SLOT_ITPL`, `DESTINATION_ITPL` 좌표는 실제 로봇에서 측정 후 채워넣어야 함
