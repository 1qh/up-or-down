import cv2
import pandas as pd
from mediapipe.python.solutions import drawing_utils as draw, pose

engine = pose.Pose()
cap = cv2.VideoCapture(0)
lm = []

while len(lm) < 100:
    ret, frame = cap.read()
    if ret:
        res = engine.process(frame)
        if res.pose_landmarks:
            draw.draw_landmarks(frame, res.pose_landmarks, pose.POSE_CONNECTIONS)
            lm.append(sum([[i.x, i.y, i.z, i.visibility] for i in res.pose_landmarks.landmark], []))
        cv2.imshow('.', frame)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()

pd.DataFrame(lm).to_csv('test.csv')