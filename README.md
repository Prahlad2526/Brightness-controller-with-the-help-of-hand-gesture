# Brightness-controller-with-the-help-of-hand-gesture
i made this project by using some of the libraries of python like ; openCv , numpy , mediapipe , screen_brightness_control
# Import necessary libraries
import mediapipe as mp
import cv2
import numpy as np
import screen_brightness_control as abc
from math import hypot

# Open a video capture object for the default camera (0)
cap = cv2.VideoCapture(0)

# Initialize MediaPipe Hands module
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

# Main loop for real-time hand tracking and brightness adjustment
while True:
    # Read a frame from the camera
    a, img = cap.read()

    # Convert the frame to RGB format (required by MediaPipe)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Process the frame with the MediaPipe Hands module to detect hand landmarks
    results = hands.process(imgRGB)

    # List to store hand landmark coordinates
    lmList = []

    # Check if hands are detected in the frame
    if results.multi_hand_landmarks:
        # Iterate through detected hands
        for handLandmark in results.multi_hand_landmarks:
            # Enumerate through landmarks of each hand
            for lmId, lm in enumerate(handLandmark.landmark):
                # Get the height and width of the frame
                h, w, _ = img.shape
                # Calculate the pixel coordinates of the landmark
                cx, cy = int(lm.x * w), int(lm.y * h)
                # Append landmark information to the list
                lmList.append([lmId, cx, cy])
                # Draw landmarks and connections on the frame
                mpDraw.draw_landmarks(img, handLandmark, mpHands.HAND_CONNECTIONS)
        
        # Print the list of detected landmarks
        print(lmList)

    # Check if landmarks are detected
    if lmList != []:
        # Extract coordinates of specific landmarks for brightness adjustment
        x1, y1 = lmList[4][1], lmList[4][2]
        x2, y2 = lmList[8][1], lmList[8][2]

        # Draw circles and a line connecting the specified landmarks
        cv2.circle(img, (x1, y1), 4, (255, 0, 0), cv2.FILLED)
        cv2.circle(img, (x2, y2), 4, (255, 0, 0), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)

        # Calculate the Euclidean distance between the two landmarks
        length = hypot(x2 - x1, y2 - y1)

        # Interpolate the distance to adjust brightness in the range [15, 220] to [0, 100]
        bright = np.interp(length, [15, 220], [0, 100])

        # Print brightness and distance for debugging
        print(bright, length)

        # Adjust screen brightness using the screen_brightness_control library
        abc.set_brightness(int(bright))

    # Display the processed frame
    cv2.imshow('Image', img)

    # Check for the 'q' key press to exit the loop
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
