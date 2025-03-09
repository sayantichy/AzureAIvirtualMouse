import time
import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import requests

from dotenv import load_dotenv
import os

load_dotenv()



# Azure Computer Vision API credentials
AZURE_ENDPOINT = "https://avm.cognitiveservices.azure.com/"
api_key = os.getenv("AZURE_KEY")

HEADERS = {'Ocp-Apim-Subscription-Key': api_key, 'Content-Type': 'application/octet-stream'}




# Initialize MediaPipe Hands model and drawing utils
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.8, min_tracking_confidence=0.8)
mp_draw = mp.solutions.drawing_utils

# Screen dimensions and sensitivity
screen_width, screen_height = pyautogui.size()
sensitivity = 1.5  # Adjust cursor speed

# Initialize state variables
hover_start_time = None
previous_position = None
dragging = False

# Colors for visual feedback
CLICK_COLOR = (0, 255, 0)  # Green
DRAG_COLOR = (0, 0, 255)  # Red
COPY_COLOR = (255, 255, 0)  # Yellow
PASTE_COLOR = (0, 255, 255)  # Cyan

def calculate_distance(point1, point2):
    """Calculates Euclidean distance between two hand landmarks."""
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)

def move_mouse_smooth(hand_landmarks, previous_position, sensitivity):
    """Moves the mouse cursor smoothly based on index finger position."""
    x = hand_landmarks.landmark[8].x  # Index finger tip x
    y = hand_landmarks.landmark[8].y  # Index finger tip y
    mouse_x = int(x * screen_width * sensitivity)
    mouse_y = int(y * screen_height * sensitivity)

    # Smooth cursor movement using weighted average
    if previous_position:
        mouse_x = int((mouse_x * 0.7) + (previous_position[0] * 0.3))
        mouse_y = int((mouse_y * 0.7) + (previous_position[1] * 0.3))

    pyautogui.moveTo(mouse_x, mouse_y)
    return mouse_x, mouse_y

def detect_click(hand_landmarks, frame):
    """Detects a pinch gesture (thumb + index finger) for left-click."""
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    if calculate_distance(thumb_tip, index_tip) < 0.05:
        pyautogui.click()
        cv2.circle(frame, (50, 50), 20, CLICK_COLOR, -1)  # Visual feedback

def toggle_drag(hand_landmarks, dragging, frame):
    """Toggles drag-and-drop on a pinch gesture (thumb + index finger)."""
    thumb_tip = hand_landmarks.landmark[4]
    index_tip = hand_landmarks.landmark[8]
    if calculate_distance(thumb_tip, index_tip) < 0.05:
        if not dragging:
            pyautogui.mouseDown()
            dragging = True
        else:
            pyautogui.mouseUp()
            dragging = False
        cv2.circle(frame, (100, 50), 20, DRAG_COLOR, -1)  # Visual feedback
    return dragging

def copy_text(hand_landmarks, frame):
    """
    Detects a pinch gesture (index + middle finger) for copying.
    Trigger a 'Ctrl+C' command if the gesture is detected.
    """
    index_tip = hand_landmarks.landmark[8]  # Index finger tip
    middle_tip = hand_landmarks.landmark[12]  # Middle finger tip

    # Check pinch distance for index and middle fingers
    if calculate_distance(index_tip, middle_tip) < 0.05:
        pyautogui.hotkey('ctrl', 'c')  # Trigger copy action
        cv2.circle(frame, (150, 50), 20, COPY_COLOR, -1)  # Visual feedback
        time.sleep(0.5)  # Prevent multiple triggers



def paste_text(hand_landmarks, frame):
    """Detects a specific gesture (thumb + pinky pinch) for pasting."""
    thumb_tip = hand_landmarks.landmark[4]
    pinky_tip = hand_landmarks.landmark[20]
    if calculate_distance(thumb_tip, pinky_tip) < 0.05:
        pyautogui.hotkey('ctrl', 'v')  # Simulate Paste (Ctrl+V)
        cv2.circle(frame, (200, 50), 20, PASTE_COLOR, -1)  # Visual feedback
        time.sleep(1)  # Prevent rapid repetition

def hover_click(hand_landmarks, hover_start_time, frame):
    """Simulates a click if the index finger hovers in the same position for 1 second."""
    index_tip = hand_landmarks.landmark[8]

    if not hover_start_time:
        hover_start_time = time.time()
    elif time.time() - hover_start_time > 1:  # Hover for 1 second
        pyautogui.click()
        hover_start_time = None
        cv2.circle(frame, (250, 50), 20, CLICK_COLOR, -1)  # Visual feedback
    return hover_start_time

def draw_feedback(frame, landmarks):
    """Draws visual feedback for landmarks."""
    for landmark in landmarks.landmark:
        x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
        cv2.circle(frame, (x, y), 5, (255, 0, 0), -1)  # Blue dots for landmarks

# Main program
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip frame for a mirrored view
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    # Detect hands
    if result.multi_hand_landmarks:
        for landmarks in result.multi_hand_landmarks:
            # Cursor movement with smoothing
            previous_position = move_mouse_smooth(landmarks, previous_position, sensitivity)

            # Gesture detection
            detect_click(landmarks, frame)
            dragging = toggle_drag(landmarks, dragging, frame)
            copy_text(landmarks, frame)
            paste_text(landmarks, frame)
            hover_start_time = hover_click(landmarks, hover_start_time, frame)

            # Draw landmarks and feedback
            mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)
            draw_feedback(frame, landmarks)

    # Display the frame
    cv2.imshow("Enhanced AI Virtual Mouse", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
