import cv2
import numpy as np

#Use CSRT tracker 
tracker = cv2.legacy.TrackerCSRT_create()

#  Kalman Filter setup
kf = cv2.KalmanFilter(4, 2)

kf.measurementMatrix = np.array([[1, 0, 0, 0],
                                 [0, 1, 0, 0]], np.float32)

kf.transitionMatrix = np.array([[1, 0, 1, 0],
                                [0, 1, 0, 1],
                                [0, 0, 1, 0],
                                [0, 0, 0, 1]], np.float32)

kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03

#  Open webcam
cap = cv2.VideoCapture(0)

ret, frame = cap.read()
if not ret:
    print("Error opening camera")
    exit()

#  Select object
bbox = cv2.selectROI("Select Object", frame, False)
cv2.destroyWindow("Select Object")

tracker.init(frame, bbox)

#  Initialize Kalman state
x, y, w, h = [int(v) for v in bbox]
cx = x + w/2
cy = y + h/2

kf.statePre = np.array([[np.float32(cx)],
                        [np.float32(cy)],
                        [0],
                        [0]], np.float32)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    #  STEP 1: Predict future position
    predicted = kf.predict()
    pred_x, pred_y = int(predicted[0][0]), int(predicted[1][0])

    #  STEP 2: Update tracker
    success, bbox = tracker.update(frame)

    if success:
        x, y, w, h = [int(v) for v in bbox]

        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        # Current actual center
        cx = x + w/2
        cy = y + h/2

        # Measurement for Kalman
        measurement = np.array([[np.float32(cx)],
                                [np.float32(cy)]])

        #  STEP 3: Correct prediction with real value
        kf.correct(measurement)

        # Draw actual position (RED)
        cv2.circle(frame, (int(cx), int(cy)), 4, (0, 0, 255), -1)

        # Draw predicted position (BLUE)
        cv2.circle(frame, (pred_x, pred_y), 6, (255, 0, 0), -1)

        # Draw line between actual and predicted
        cv2.line(frame,
                 (int(cx), int(cy)),
                 (pred_x, pred_y),
                 (255, 255, 0), 2)

        cv2.putText(frame, "Red=Actual | Blue=Predicted",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 255, 255), 2)

    else:
        cv2.putText(frame, "Tracking Failure",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 0, 255), 2)

    cv2.imshow("Tracking + Prediction", frame)

    # Press ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()