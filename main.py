import cv2
import mediapipe as mp
import classifier
import image_picker
import utils

mp_face = mp.solutions.face_mesh
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

face_mesh = mp_face.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.6
)

cap = cv2.VideoCapture(0)
show_landmarks = True

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_result = face_mesh.process(rgb)
    hand_result = hands.process(rgb)
    face_lm = face_result.multi_face_landmarks[0] if face_result.multi_face_landmarks else None
    hand_lm = hand_result.multi_hand_landmarks[0] if hand_result.multi_hand_landmarks else None
    label = classifier.classify(face_lm, hand_lm)

    # draw label on frame
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 4)
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)

    if show_landmarks:
        if face_result.multi_face_landmarks:
            for lm in face_result.multi_face_landmarks:
                mp_drawing.draw_landmarks(frame, lm, mp_face.FACEMESH_TESSELATION)
        if hand_result.multi_hand_landmarks:
            for lm in hand_result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

    reaction_img = image_picker.get_image(label)
    frame = utils.overlay(frame, reaction_img)

    cv2.imshow("The Goat", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    if key == ord('d'):
        show_landmarks = not show_landmarks

cap.release()
cv2.destroyAllWindows()