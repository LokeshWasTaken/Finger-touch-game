import cv2
import mediapipe as mp
import numpy as np
import random
import math
import time

WIDTH, HEIGHT = 1640, 720      # <-- CHANGE DIMENSIONS
BALL_RADIUS = 15
HIT_COOLDOWN = 0.4

# HAND TRACKER

class HandTracker:
    def __init__(self):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        self.drawer = mp.solutions.drawing_utils

    def get_index_finger_tip(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.hands.process(rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            self.drawer.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)

            tip = hand.landmark[8]
            x = int(tip.x * WIDTH)
            y = int(tip.y * HEIGHT)
            return (x, y)

        return None


# BALL OBJECT

class Ball:
    def __init__(self):
        self.respawn()

    def respawn(self):
        self.x = random.randint(BALL_RADIUS, WIDTH - BALL_RADIUS)
        self.y = random.randint(BALL_RADIUS, HEIGHT - BALL_RADIUS)

    def draw(self, frame):
        cv2.circle(frame, (self.x, self.y), BALL_RADIUS, (0, 0, 255), -1)

# GAME CLASS

class FingerTouchGame:
    def __init__(self):
        self.cap = cv2.VideoCapture(0)  # DroidCam IP
        self.cap.set(3, WIDTH)
        self.cap.set(4, HEIGHT)

        self.hand_tracker = HandTracker()
        self.ball = Ball()
        self.score = 0
        self.last_hit_time = 0

    def check_collision(self, finger):
        dist = math.hypot(finger[0] - self.ball.x, finger[1] - self.ball.y)
        return dist < BALL_RADIUS

    def draw_ui(self, frame):
        cv2.putText(
            frame,
            f"Score: {self.score}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (255, 255, 255),
            3
        )

    def run(self):
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            frame = cv2.resize(frame, (WIDTH, HEIGHT))

            finger = self.hand_tracker.get_index_finger_tip(frame)
            self.ball.draw(frame)

            if finger:
                cv2.circle(frame, finger, 12, (0, 255, 0), -1)

                current_time = time.time()
                if self.check_collision(finger):
                    if current_time - self.last_hit_time > HIT_COOLDOWN:
                        self.score += 1
                        self.ball.respawn()
                        self.last_hit_time = current_time

            self.draw_ui(frame)
            cv2.imshow("Finger Touch Game (20:9 Optimized)", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        print(f"\nYou scored {self.score} points")

        self.cap.release()
        cv2.destroyAllWindows()

# ENTRY POINT

if __name__ == "__main__":
    FingerTouchGame().run()
