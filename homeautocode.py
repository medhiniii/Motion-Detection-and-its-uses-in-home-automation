# Importing necessary libraries
import cv2
import pandas as pd
from datetime import datetime

# Assigning our static background as None for initial frames
static_back = None

# List to store motion status
motion_list = [None, None]

# List to capture time when movement occurs
motion_time = []

# Initializing DataFrame to store timestamps of motion
df = pd.DataFrame(columns=["Initial", "Final"])

# Start capturing video from webcam
video = cv2.VideoCapture(0)

while True:
    # Read frame from video
    check, frame = video.read()
    motion = 0

    # Convert color frame to grayscale and blur it
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    # Set the first frame as static background
    if static_back is None:
        static_back = gray
        continue

    # Compute absolute difference between static background and current frame
    diff_frame = cv2.absdiff(static_back, gray)

    # Apply threshold to get binary image and then dilate it
    thresh_frame = cv2.threshold(diff_frame, 30, 255, cv2.THRESH_BINARY)[1]
    thresh_frame = cv2.dilate(thresh_frame, None, iterations=2)

    # Find contours of moving object
    contours, _ = cv2.findContours(thresh_frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if cv2.contourArea(contour) < 10000:
            continue
        motion = 1
        (x, y, w, h) = cv2.boundingRect(contour)
        # Draw rectangle around moving object
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)

    # Update motion list
    motion_list.append(motion)
    motion_list = motion_list[-2:]

    # Record motion start time
    if motion_list[-1] == 1 and motion_list[-2] == 0:
        motion_time.append(datetime.now())

    # Record motion end time
    if motion_list[-1] == 0 and motion_list[-2] == 1:
        motion_time.append(datetime.now())

    # Display different video frames
    cv2.imshow("Gray Frame", gray)
    cv2.imshow("Difference Frame", diff_frame)
    cv2.imshow("Threshold Frame", thresh_frame)
    cv2.imshow("Color Frame", frame)

    # Press 'm' to exit
    key = cv2.waitKey(1)
    if key == ord('m'):
        if motion == 1:
            motion_time.append(datetime.now())
        break

# Add motion timestamps to DataFrame
for i in range(0, len(motion_time), 2):
    df = df.append({"Initial": motion_time[i], "Final": motion_time[i + 1]}, ignore_index=True)

# Save timestamps to CSV file
df.to_csv("MovementsTimeFile.csv")

# Release video capture and close all windows
video.release()
cv2.destroyAllWindows()
