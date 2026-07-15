import cv2

for i in range(10):
    cap = cv2.VideoCapture(i)

    if cap.isOpened():
        ret, frame = cap.read()

        if ret:
            print(f"Camera Index {i}: Working ({frame.shape})")
        else:
            print(f"Camera Index {i}: Opened but no frame")

        cap.release()