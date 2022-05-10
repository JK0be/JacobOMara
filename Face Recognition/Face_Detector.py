from ast import Break
from multiprocessing.connection import wait
import cv2
from cv2 import COLOR_BGR2GRAY

# Load Pre-Trained data from open CV
preTrainedData = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Choose an Image to Detect Face
#img = cv2.imread("TEST.jpg")
webcam = cv2.VideoCapture(0)

while True:

    frameReadSuccessful, frame = webcam.read()

    grayscaled_frame = cv2.cvtColor(frame, COLOR_BGR2GRAY)

    faceCoordinates = preTrainedData.detectMultiScale(grayscaled_frame)

    for (x, y, w, h) in faceCoordinates:
         cv2.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), 2)

    cv2.imshow("TEST", frame)
    key = cv2.waitKey(1)
    
    if key == 81 or key == 113:
        break

webcam.release()
print("Code Executed Correctly")


# # Convert to Greyscale
# grayscaled_img = cv2.cvtColor(img, COLOR_BGR2GRAY)

# # Detect faces in image with Pre-Trained Data
# face_coordinates = preTrainedData.detectMultiScale(grayscaled_img)

# # Drawing a Rectangle 
# for (x, y, w, h) in face_coordinates:
#     cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)

# # Showing the Image     
# cv2.imshow("IMAGESHOWER", img)
# cv2.waitKey()

# print("Code Completed successfully")