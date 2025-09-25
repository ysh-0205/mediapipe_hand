import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time

# 화면 크기 가져오기
SCREEN_WIDTH, SCREEN_HEIGHT = pyautogui.size()

# 변수 정의
SMOOTHING = 0.2
CLICK_THRESHOLD = 0.03  # 손가락 거리 임계값 (0~1 기준)
CLICK_COOLDOWN = 1.0    # 클릭 간 최소 시간 (초)

# Mediapipe Hands 초기화
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils  # 랜드마크 시각화용

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# 웹캠 열기
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("카메라를 열 수 없습니다.")

smooth_x, smooth_y = 0, 0
last_click_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임을 가져올 수 없습니다.")
        break

    h, w = frame.shape[:2]

    # 좌우 반전 (거울 효과)
    frame = cv2.flip(frame, 1)

    # BGR -> RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 손 인식
    results = hands.process(frame_rgb)

    click_display = False  # 클릭 텍스트 표시 여부

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]

        landmarks = hand_landmarks.landmark

        # 검지 끝(8), 엄지 끝(4)
        index_finger_tip = landmarks[8]
        thumb_tip = landmarks[4]

        # 화면 좌표로 변환 (좌우 반전 처리됨)
        mouse_x = int(index_finger_tip.x * SCREEN_WIDTH)
        mouse_y = int(index_finger_tip.y * SCREEN_HEIGHT)

        # 스무딩 적용
        smooth_x = smooth_x * (1 - SMOOTHING) + mouse_x * SMOOTHING
        smooth_y = smooth_y * (1 - SMOOTHING) + mouse_y * SMOOTHING

        pyautogui.moveTo(int(smooth_x), int(smooth_y))

        # 엄지와 검지 거리 계산
        dist = np.sqrt(
            (index_finger_tip.x - thumb_tip.x) ** 2 +
            (index_finger_tip.y - thumb_tip.y) ** 2
        )

        current_time = time.time()
        if dist < CLICK_THRESHOLD and (current_time - last_click_time) > CLICK_COOLDOWN:
            pyautogui.click()
            last_click_time = current_time
            click_display = True  # 클릭 텍스트 표시

        # 랜드마크 시각화
        mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    else:
        cv2.putText(frame, "No Hand Detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    if click_display:
        cv2.putText(frame, "CLICK!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 4)

    cv2.imshow("Hand Gesture Mouse Control", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
