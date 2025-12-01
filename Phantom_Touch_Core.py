import cv2
import numpy as np
from pythonosc import udp_client

# ==========================================
# [설정] 승민이(사운드)에게 보낼 주소
# ==========================================
OSC_IP = "127.0.0.1"  # 테스트할 땐 내 컴퓨터 주소 (로컬)
OSC_PORT = 8000
ADDRESS = "/omo/shadow"

# 카메라 설정 (웹캠: 0 또는 1 / 아이폰 연속성: 0 또는 1)
CAM_ID = 0
WIDTH, HEIGHT = 1280, 720  

# 그림자 인식 민감도 (조명에 따라 조절: 0~255)
# 숫자가 클수록 '덜 어두운' 것도 그림자로 인식함
THRESHOLD_VAL = 80 
MIN_AREA = 1000  # 너무 작은 노이즈 무시

# ==========================================
# [초기화] 변수 세팅
# ==========================================
client = udp_client.SimpleUDPClient(OSC_IP, OSC_PORT)
# [초기화] 카메라 미리 다 열어두기 (속도 향상)
caps = []
for i in [0, 1]:
    c = cv2.VideoCapture(i)
    c.set(3, WIDTH)
    c.set(4, HEIGHT)
    c.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    caps.append(c)
    print(f"--- Camera {i} Initialized ---")

cur_cam_idx = 0

# ==========================================
# [커스텀 효과] 여기에 원하는 효과를 추가하세요
# ==========================================
TARGET_ZOOM = 1.0
CURRENT_ZOOM = 1.0
TARGET_ZOOM = 1.0
CURRENT_ZOOM = 1.0
VIEW_MODE = 0 # 0: Normal, 1: Threshold (Debug)
SHOW_HUD = True # 텍스트 표시 여부

def apply_zoom(img, zoom_val):
    if zoom_val > 1.0:
        h, w = img.shape[:2]
        cx, cy = w // 2, h // 2
        rw = int(w / (2 * zoom_val))
        rh = int(h / (2 * zoom_val))
        img = img[cy-rh:cy+rh, cx-rw:cx+rw]
        img = cv2.resize(img, (w, h))
    return img

    return img

    return img

# 카메라별 설정 저장 (0번, 1번 각각 독립적)
cam_configs = [
    {'points': [], 'matrix': None, 'is_calibrated': False}, # Cam 0
    {'points': [], 'matrix': None, 'is_calibrated': False}  # Cam 1
]

target_w, target_h = 1920, 1080 # 펴졌을 때 내부 연산 크기 (FHD)

# 마우스 클릭 콜백 함수
def click_event(event, x, y, flags, params):
    global cam_configs
    cfg = cam_configs[cur_cam_idx]

    if event == cv2.EVENT_LBUTTONDOWN:
        if len(cfg['points']) < 4:
            cfg['points'].append((x, y))
            print(f"[Cam {cur_cam_idx}] Point {len(cfg['points'])}: {x}, {y}")
            
            # 4개 다 찍으면 변환 행렬 계산
            if len(cfg['points']) == 4:
                pts1 = np.float32(cfg['points'])
                pts2 = np.float32([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]])
                cfg['matrix'] = cv2.getPerspectiveTransform(pts1, pts2)
                cfg['is_calibrated'] = True
                print(f"--- [Cam {cur_cam_idx}] Calibration Complete! ---")

cv2.namedWindow("Phantom Touch")
cv2.setMouseCallback("Phantom Touch", click_event)

print(f"시스템 시작... 화면의 네 모서리(좌상->우상->우하->좌하)를 클릭하세요.")

# ==========================================
# [메인 루프]
# ==========================================
while True:
    # 현재 선택된 카메라에서 읽기
    ret, frame = caps[cur_cam_idx].read()
    if not ret: break

    # [스무스 줌] 목표값으로 부드럽게 이동 (Interpolation)
    CURRENT_ZOOM += (TARGET_ZOOM - CURRENT_ZOOM) * 0.1
    
    # [줌 적용] 원본 프레임에 바로 적용
    frame = apply_zoom(frame, CURRENT_ZOOM)

    # 현재 카메라 설정 가져오기
    cfg = cam_configs[cur_cam_idx]

    # [1] 보정 전: 점 찍는 화면
    if not cfg['is_calibrated']:
        if VIEW_MODE == 1:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (21, 21), 0)
            _, thresh_temp = cv2.threshold(blur, THRESHOLD_VAL, 255, cv2.THRESH_BINARY_INV)
            display_img = cv2.cvtColor(thresh_temp, cv2.COLOR_GRAY2BGR)
        else:
            display_img = frame.copy()

        for pt in cfg['points']:
            cv2.circle(display_img, pt, 5, (255, 255, 255), -1)
        
        status_msg = f"Cam {cur_cam_idx} Calibration Mode"

    # [2] 보정 후: 워핑 & 센싱
    else:
        warped = cv2.warpPerspective(frame, cfg['matrix'], (target_w, target_h))
        
        # 전처리
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0)
        _, thresh = cv2.threshold(blur, THRESHOLD_VAL, 255, cv2.THRESH_BINARY_INV)

        # 컨투어 찾기
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area > MIN_AREA:
                topmost = tuple(c[c[:, :, 1].argmin()][0])
                tx, ty = topmost

                norm_x = tx / target_w
                norm_y = 1.0 - (ty / target_h)
                norm_z = min(area / (target_w * target_h * 0.6), 1.0)

                # OSC 전송 (메인 카메라 0번일 때만)
                if cur_cam_idx == 0:
                    client.send_message(ADDRESS, [norm_x, norm_y, norm_z])

                # 시각화
                cv2.drawContours(warped, [c], -1, (255, 255, 255), 2)
                cv2.circle(warped, topmost, 15, (255, 255, 255), -1)
                if SHOW_HUD:
                    cv2.putText(warped, f"X:{norm_x:.2f} Y:{norm_y:.2f} Size:{norm_z:.2f}", 
                                (tx+20, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # 화면 출력 설정
        if VIEW_MODE == 0:
            display_img = warped
        else:
            display_img = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
            if contours and area > MIN_AREA:
                cv2.drawContours(display_img, [c], -1, (255, 255, 255), 2)
                cv2.circle(display_img, topmost, 15, (255, 255, 255), -1)
        
        status_msg = f"Cam {cur_cam_idx} Tracking Mode"

    # [공통] 상태 텍스트 표시
    if SHOW_HUD:
        osc_status = "ON" if (cur_cam_idx == 0 and cfg['is_calibrated']) else "OFF"
        info_txt = f"{status_msg} | T:{THRESHOLD_VAL} | Z:x{CURRENT_ZOOM:.1f} | OSC:{osc_status} | Mode:{'RGB' if VIEW_MODE==0 else 'BW'}"
        cv2.putText(display_img, info_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.imshow("Phantom Touch", display_img)

    # 키보드 제어
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('r'): # 리셋 (현재 카메라)
        cam_configs[cur_cam_idx]['points'] = []
        cam_configs[cur_cam_idx]['is_calibrated'] = False
        print(f"--- Reset Calibration (Cam {cur_cam_idx}) ---")
    elif key == ord('['):
        THRESHOLD_VAL = max(0, THRESHOLD_VAL - 5)
        print(f"Threshold: {THRESHOLD_VAL}")
    elif key == ord(']'):
        THRESHOLD_VAL = min(255, THRESHOLD_VAL + 5)
        print(f"Threshold: {THRESHOLD_VAL}")
    elif key == ord('1'): # 카메라 1번
        cur_cam_idx = 0
        print("Switched to Camera 0")
    elif key == ord('2'): # 카메라 2번
        cur_cam_idx = 1
        print("Switched to Camera 1")
    elif key == ord('h'): # HUD 토글
        SHOW_HUD = not SHOW_HUD
        print(f"HUD Visible: {SHOW_HUD}")
    elif key == ord('-'): # 줌 아웃
        TARGET_ZOOM = max(1.0, TARGET_ZOOM - 0.2)
    elif key == ord('='): # 줌 인
        TARGET_ZOOM = min(5.0, TARGET_ZOOM + 0.2)
    elif key == ord('0'): # 줌 초기화
        TARGET_ZOOM = 1.0
        print("Zoom Reset")
    elif key == 9: # Tab 키: 뷰 모드 변경
        VIEW_MODE = 1 - VIEW_MODE
        print(f"View Mode: {VIEW_MODE}")

# 종료 시 모든 카메라 해제
for c in caps:
    c.release()
cv2.destroyAllWindows()