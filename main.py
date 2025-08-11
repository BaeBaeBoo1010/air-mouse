import time
import math
import numpy as np
import cv2
import mediapipe as mp
import pyautogui
from pygrabber.dshow_graph import FilterGraph
from pynput.mouse import Controller as MouseController, Button
from collections import deque

# ========== One-Euro Filter ==========
class OneEuroFilter:
    def __init__(self, freq=60.0, min_cutoff=1.5, beta=0.007, d_cutoff=1.0):
        self.freq, self.min_cutoff, self.beta, self.d_cutoff = freq, min_cutoff, beta, d_cutoff
        self.x_prev = self.dx_prev = self.t_prev = None

    def _alpha(self, cutoff):
        tau = 1.0 / (2 * math.pi * cutoff)
        te = 1.0 / self.freq
        return 1.0 / (1.0 + tau / te)

    def __call__(self, x, t):
        if self.t_prev is None:
            self.t_prev, self.x_prev = t, x
            return x
        dt = t - self.t_prev
        if dt <= 0:
            return x
        self.freq = 1.0 / dt
        dx = (x - self.x_prev) * self.freq
        a_d = self._alpha(self.d_cutoff)
        dx_hat = a_d * dx + (1 - a_d) * (self.dx_prev if self.dx_prev is not None else dx)
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff)
        x_hat = a * x + (1 - a) * self.x_prev
        self.x_prev, self.dx_prev, self.t_prev = x_hat, dx_hat, t
        return x_hat

# ========== Tính góc giữa 2 vector ==========
def angle_between(v1, v2):
    v1_u = v1 / (np.linalg.norm(v1) + 1e-9)
    v2_u = v2 / (np.linalg.norm(v2) + 1e-9)
    dot = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    return math.acos(dot)

# ========== Chọn camera trên Windows ==========
def select_camera():
    fg = FilterGraph()
    cams = fg.get_input_devices()
    if not cams:
        print("No cameras detected!")
        exit(1)

    print("Available cameras:")
    for i, name in enumerate(cams):
        print(f"{i}: {name}")

    selected_cam = 0
    try:
        user_input = input(f"Select camera (default {selected_cam}): ")
        if user_input.strip() != '':
            idx = int(user_input)
            if 0 <= idx < len(cams):
                selected_cam = idx
            else:
                print(f"Invalid selection, using default {selected_cam}")
    except Exception:
        print(f"Invalid input, using default {selected_cam}")

    return selected_cam

def main():
    mouse = MouseController()
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7, min_tracking_confidence=0.6)
    screen_w, screen_h = pyautogui.size()

    PINCH_ANGLE_THRESH = 0.4
    HAND_TIMEOUT = 0.3
    SCROLL_PIX_PER_LINE = 100

    scrolling = False
    prev_scroll_y = None
    scroll_accum = 0.0

    filter_x = OneEuroFilter(freq=30)
    filter_y = OneEuroFilter(freq=30)
    filter_scroll = OneEuroFilter(freq=30)

    PATH_BUF_SIZE = 5
    path_buf = deque(maxlen=PATH_BUF_SIZE)

    left_clicking = False
    right_clicking = False
    last_hand_time = 0

    selected_cam = select_camera()
    cap = cv2.VideoCapture(selected_cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    prev_frame_time = 0
    FPS_LIMIT = 30

    try:
        while True:
            current_time = time.time()
            if current_time - prev_frame_time < 1.0 / FPS_LIMIT:
                continue
            prev_frame_time = current_time

            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)

            # Resize và padding giữ tỉ lệ khung hình
            h, w, _ = frame.shape
            scale_w = screen_w / w
            scale_h = screen_h / h
            scale = min(scale_w, scale_h)
            new_w, new_h = int(w * scale), int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
            bg = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            x_offset = (screen_w - new_w) // 2
            y_offset = (screen_h - new_h) // 2
            bg[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = frame
            frame = bg.copy()

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = hands.process(rgb)

            hand_seen = False
            tgt_x = tgt_y = None
            draw_pts = []

            if res.multi_hand_landmarks and res.multi_handedness:
                for lmks, handed in zip(res.multi_hand_landmarks, res.multi_handedness):
                    label = handed.classification[0].label  # 'Right' hoặc 'Left'

                    def lm_to_np(lm):
                        return np.array([lm.x, lm.y])

                    thumb = lm_to_np(lmks.landmark[4])
                    index = lm_to_np(lmks.landmark[8])
                    middle = lm_to_np(lmks.landmark[12])
                    ring = lm_to_np(lmks.landmark[16])

                    if label == 'Right':
                        tgt_x = index[0] * screen_w
                        tgt_y = index[1] * screen_h
                        draw_pts.append(lmks.landmark[8])
                        hand_seen = True
                    else:
                        vec_thumb_index = index - thumb
                        vec_thumb_middle = middle - thumb
                        vec_thumb_ring = ring - thumb

                        dist_thumb_index = np.linalg.norm(vec_thumb_index)
                        dist_thumb_middle = np.linalg.norm(vec_thumb_middle)
                        dist_thumb_ring = np.linalg.norm(vec_thumb_ring)

                        # Click trái
                        if dist_thumb_index < 0.06 and not scrolling:
                            if not left_clicking:
                                mouse.press(Button.left)
                                left_clicking = True
                        elif left_clicking:
                            mouse.release(Button.left)
                            left_clicking = False

                        # Click phải
                        if dist_thumb_middle < 0.06 and not scrolling:
                            if not right_clicking:
                                mouse.press(Button.right)
                                right_clicking = True
                        elif right_clicking:
                            mouse.release(Button.right)
                            right_clicking = False

                        # Scroll
                        if dist_thumb_ring < 0.06:
                            if not scrolling:
                                scrolling = True
                                prev_scroll_y = None
                                scroll_accum = 0.0
                        else:
                            if scrolling:
                                scrolling = False
                                prev_scroll_y = None
                                scroll_accum = 0.0

                        draw_pts.extend([lmks.landmark[4], lmks.landmark[8], lmks.landmark[12], lmks.landmark[16]])

            if hand_seen and tgt_x is not None:
                last_hand_time = time.time()
                t_now = time.time()
                sm_x = filter_x(tgt_x, t_now)
                sm_y = filter_y(tgt_y, t_now)

                if scrolling:
                    if prev_scroll_y is None:
                        prev_scroll_y = sm_y
                        scroll_accum = 0.0
                    else:
                        delta = prev_scroll_y - sm_y
                        scroll_accum += delta * 4

                        filtered_scroll = filter_scroll(scroll_accum, t_now)

                        lines = int(filtered_scroll / SCROLL_PIX_PER_LINE)
                        if lines != 0:
                            mouse.scroll(0, lines)
                            scroll_accum -= lines * SCROLL_PIX_PER_LINE
                            prev_scroll_y = sm_y
                else:
                    path_buf.append((sm_x, sm_y))

                    if len(path_buf) >= 2:
                        t_vals = np.arange(len(path_buf))
                        xs = np.array([p[0] for p in path_buf])
                        ys = np.array([p[1] for p in path_buf])

                        A = np.vstack([t_vals, np.ones_like(t_vals)]).T
                        m_x, b_x = np.linalg.lstsq(A, xs, rcond=None)[0]
                        m_y, b_y = np.linalg.lstsq(A, ys, rcond=None)[0]

                        next_t = t_vals[-1] + 1
                        pred_x = m_x * next_t + b_x
                        pred_y = m_y * next_t + b_y
                    else:
                        pred_x, pred_y = sm_x, sm_y

                    pred_x = max(0, min(screen_w - 1, pred_x))
                    pred_y = max(0, min(screen_h - 1, pred_y))

                    mouse.position = (int(pred_x), int(pred_y))

                    # Vẽ chấm đỏ vị trí dự đoán chuột
                    cv2.circle(frame,
                               (int(pred_x / screen_w * new_w + x_offset),
                                int(pred_y / screen_h * new_h + y_offset)),
                               6, (0, 0, 255), -1)

            # Reset filter khi không thấy tay phải lâu
            elif time.time() - last_hand_time > HAND_TIMEOUT:
                filter_x.t_prev = filter_y.t_prev = filter_scroll.t_prev = None
                path_buf.clear()
                scrolling = False
                prev_scroll_y = None
                scroll_accum = 0.0

            # Vẽ điểm landmark
            for tip in draw_pts:
                cv2.circle(frame,
                           (int(tip.x * new_w + x_offset), int(tip.y * new_h + y_offset)),
                           8, (0, 255, 0), -1)

            # Vẽ đường dự đoán chuột nếu không cuộn
            if len(path_buf) >= 2 and not scrolling:
                for i in range(len(path_buf) - 1):
                    x1 = int(path_buf[i][0] / screen_w * new_w + x_offset)
                    y1 = int(path_buf[i][1] / screen_h * new_h + y_offset)
                    x2 = int(path_buf[i + 1][0] / screen_w * new_w + x_offset)
                    y2 = int(path_buf[i + 1][1] / screen_h * new_h + y_offset)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 200, 0), 1)

            # Hiển thị trạng thái click/scroll
            status_y = 40
            if left_clicking:
                cv2.putText(frame, "Left Click", (30, status_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                status_y += 40
            if right_clicking:
                cv2.putText(frame, "Right Click", (30, status_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                status_y += 40
            if scrolling:
                cv2.putText(frame, "Scrolling", (30, status_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            cv2.imshow('AI Mouse Control (Improved Scroll)', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
