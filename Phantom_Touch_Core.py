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
cap = cv2.VideoCapture(CAM_ID)
cap.set(3, WIDTH)
cap.set(4, HEIGHT)
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

points = [] # 클릭한 4개 좌표 저장용
is_calibrated = False
matrix = None
target_w, target_h = 1920, 1080 # 펴졌을 때 내부 연산 크기 (FHD)

# 마우스 클릭 콜백 함수
def click_event(event, x, y, flags, params):
    global points, is_calibrated, matrix
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(points) < 4:
            points.append((x, y))
            print(f"Point {len(points)}: {x}, {y}")
            
            # 4개 다 찍으면 변환 행렬 계산
            if len(points) == 4:
                pts1 = np.float32(points)
                pts2 = np.float32([[0, 0], [target_w, 0], [target_w, target_h], [0, target_h]])
                matrix = cv2.getPerspectiveTransform(pts1, pts2)
                is_calibrated = True
                print("--- Calibration Complete! Tracking Started ---")

cv2.namedWindow("Phantom Touch")
cv2.setMouseCallback("Phantom Touch", click_event)

print(f"시스템 시작... 화면의 네 모서리(좌상->우상->우하->좌하)를 클릭하세요.")

# ==========================================
# [메인 루프]
# ==========================================
while True:
    ret, frame = cap.read()
    if not ret: break

    # [1] 보정 전: 점 찍는 화면 보여주기
    if not is_calibrated:
        for pt in points:
            cv2.circle(frame, pt, 5, (255, 255, 255), -1)
        cv2.imshow("Phantom Touch", frame)

    # [2] 보정 후: 펴진 화면에서 그림자 추적
    else:
        # 화면 펴기 (Warp Perspective)
        warped = cv2.warpPerspective(frame, matrix, (target_w, target_h))
        
        # 전처리 (그림자만 따기)
        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (21, 21), 0) # 뭉개기
        _, thresh = cv2.threshold(blur, THRESHOLD_VAL, 255, cv2.THRESH_BINARY_INV)

        # 컨투어 찾기
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # 가장 큰 덩어리 찾기
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)

            if area > MIN_AREA:
                # ★ 핵심: 가장 상단점 (Top-most) 찾기
                topmost = tuple(c[c[:, :, 1].argmin()][0])
                tx, ty = topmost

                # 데이터 정규화 (0.0 ~ 1.0)
                norm_x = tx / target_w
                norm_y = 1.0 - (ty / target_h) # Y축 뒤집기 (위가 1.0)
                norm_z = min(area / (target_w * target_h * 0.6), 1.0) # 면적

                # OSC 전송
                client.send_message(ADDRESS, [norm_x, norm_y, norm_z])

                # 시각화 (빨간 점 & 텍스트)
                cv2.drawContours(warped, [c], -1, (255, 255, 255), 2)
                cv2.circle(warped, topmost, 15, (255, 255, 255), -1)
                cv2.putText(warped, f"X:{norm_x:.2f} Y:{norm_y:.2f} Size:{norm_z:.2f}", 
                            (tx+20, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 화면 출력 (펴진 화면 & 흑백 화면)
        cv2.putText(warped, f"Threshold: {THRESHOLD_VAL}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow("Phantom Touch", warped)
        cv2.imshow("Debug View", thresh) # 그림자가 잘 따지는지 확인용

    # 키보드 제어
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    if key == ord('r'): # 리셋 (다시 점 찍기)
        points = []
        is_calibrated = False
        print("--- Reset Calibration ---")
    elif key == ord('['):
        THRESHOLD_VAL = max(0, THRESHOLD_VAL - 5)
        print(f"Threshold: {THRESHOLD_VAL}")
    elif key == ord(']'):
        THRESHOLD_VAL = min(255, THRESHOLD_VAL + 5)
        print(f"Threshold: {THRESHOLD_VAL}")

cap.release()
cv2.destroyAllWindows()