import cv2

path = "data/1_0015_00.mp4"

cap = cv2.VideoCapture(path)

frame_count = 0
frames = []

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
    frame_count += 1

cap.release()

print(f"Total frames read = {frame_count}")
print(f"Single frame shape = {frames[0].shape}")