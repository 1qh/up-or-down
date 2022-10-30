import torch
device = 'cuda' if torch.cuda.is_available else 'cpu'
import cv2
from mediapipe.python.solutions import drawing_utils as draw, pose

from model import MyLSTM
md = MyLSTM(
    input_size=132,
    hidden_size=20,
    num_class=2
).to(device)
md.load_state_dict(torch.load('best.pt'))
md.eval()

engine = pose.Pose()
cap = cv2.VideoCapture(0)
lm = []
label = str()

while True:
    ret, frame = cap.read()
    if ret:
        res = engine.process(frame)
        if res.pose_landmarks:
            draw.draw_landmarks(frame, res.pose_landmarks, pose.POSE_CONNECTIONS)
            lm.append(sum([[i.x, i.y, i.z, i.visibility] for i in res.pose_landmarks.landmark], []))
            if len(lm) == 10:
                X = torch.as_tensor([lm], device=device)
                with torch.inference_mode():
                    y = torch.argmax(md(X), 1).item()
                    if y == 0:
                        label = 'Hands down'
                    elif y == 1:
                        label = 'Hands up'
                    lm = []
        cv2.putText(frame, label, (10, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255))
        cv2.imshow('.', frame)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    else:
        break
cap.release()
cv2.destroyAllWindows()