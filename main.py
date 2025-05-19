import cv2
import numpy as np
import mediapipe as mp
import os

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)

cap = cv2.VideoCapture(0)
canvas = np.zeros((480, 640, 3), dtype=np.uint8)
draw_color = (0, 0, 255)  # Default color: red
brush_thickness = 5

undo_stack = []
redo_stack = []
previous_point = None
status_message = ""

# Define color palette rectangles
colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
color_names = ["Red", "Green", "Blue", "Yellow", "Magenta", "Cyan"]
palette_rects = [(i * 60 + 10, 10, 50, 30) for i in range(len(colors))]

def recognize_and_draw_shapes():
    global canvas
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    new_canvas = np.zeros_like(canvas)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 100:
            continue

        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        shape = "Unidentified"

        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.8 <= aspect_ratio <= 1.2:
                shape = "Square"
            else:
                shape = "Rectangle"
        elif len(approx) > 4:
            perimeter = cv2.arcLength(contour, True)
            if perimeter == 0:
                continue
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if circularity > 0.7:
                shape = "Circle"

        if shape == "Circle":
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(new_canvas, center, radius, draw_color, -1)
        else:
            cv2.drawContours(new_canvas, [approx], 0, draw_color, -1)

    canvas = new_canvas
    print("Shapes recognized and redrawn.")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    for (x, y, w, h), color in zip(palette_rects, colors):
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, -1)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 255), 2)

    fingers_up = []
    ix, iy = 0, 0
    tx, ty = 0, 0

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = hand_landmarks.landmark
            index_finger_tip = landmarks[8]
            thumb_tip = landmarks[4]

            h, w, _ = frame.shape
            ix, iy = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            tx, ty = int(thumb_tip.x * w), int(thumb_tip.y * h)

            for i, (x, y, width, height) in enumerate(palette_rects):
                if x <= ix <= x + width and y <= iy <= y + height:
                    draw_color = colors[i]
                    status_message = f"Color set to {color_names[i]}"

            fingers_up = [
                landmarks[8].y < landmarks[6].y,
                landmarks[12].y < landmarks[10].y,
                landmarks[16].y < landmarks[14].y,
                landmarks[20].y < landmarks[18].y,
            ]

            if fingers_up == [True, False, False, False]:
                if previous_point is None:
                    previous_point = (ix, iy)
                else:
                    undo_stack.append(canvas.copy())
                    redo_stack.clear()
                    cv2.line(canvas, previous_point, (ix, iy), draw_color, brush_thickness)
                    previous_point = (ix, iy)
            elif abs(ix - tx) < 40 and abs(iy - ty) < 40:
                cv2.circle(canvas, (ix, iy), 20, (0, 0, 0), -1)
                previous_point = None
            else:
                previous_point = None

            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    preview_canvas = canvas.copy()
    if previous_point and fingers_up == [True, False, False, False]:
        cv2.line(preview_canvas, previous_point, (ix, iy), draw_color, brush_thickness)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break
    elif key == ord('z'):
        if undo_stack:
            redo_stack.append(canvas.copy())
            canvas = undo_stack.pop()
            status_message = "Undo performed."
    elif key == ord('y'):
        if redo_stack:
            undo_stack.append(canvas.copy())
            canvas = redo_stack.pop()
            status_message = "Redo performed."
    elif key == ord('s'):
        recognize_and_draw_shapes()
        status_message = "Shape recognition complete."
    elif key == ord('w'):
        if not os.path.exists("output"):
            os.makedirs("output")
        filename = f"output/drawing_{len(os.listdir('output')) + 1}.png"
        cv2.imwrite(filename, canvas)
        status_message = f"Drawing saved as '{filename}'"
        print(status_message)

    combined = cv2.addWeighted(frame, 0.5, preview_canvas, 0.5, 0)
    cv2.putText(combined, f"Color: {draw_color}", (10, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, draw_color, 2)
    cv2.putText(combined, status_message, (10, 470),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)

    cv2.imshow("Gesture Drawing", combined)

cap.release()
cv2.destroyAllWindows()
# Note: The code above is a complete implementation of the drawing application with gesture recognition and shape detection.