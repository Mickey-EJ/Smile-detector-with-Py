import cv2

# face and smile classifiers
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')

# Getting the webcam feed
webcam = cv2.VideoCapture(0)

while True:

    # Read current frame from webcam
    successful_frame_read, frame = webcam.read()

    # If there is an error, break
    if not successful_frame_read:
        break

    # Change to grayscale
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces first
    faces = face_detector.detectMultiScale(frame_grayscale)

    # run face detection
    for(x, y, w, h) in faces:

        # draw a rectangle around the face
        cv2.rectangle((frame), (x, y), (x+w, y+h), (0, 250, 0), 2)

        # getting the subframe using numpy Ndimensional array slicing
        the_face = frame[y:y+h, x:x+w]

        # grayscale again
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # detecting smiles
        smiles = smile_detector.detectMultiScale(
            face_grayscale, scaleFactor=1.7, minNeighbors=40)

        # uncomment these if you want to draw a rectangle around the smile as well
        #for (x_, y_, w_, h_) in smiles:

         #    cv2.rectangle((the_face), (x_, y_),
         #           (x_ + w_, y_ + h_), (250, 200, 0), 2)

        # Label as smiling
        if len(smiles) > 0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale=2,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    # showing the current frame
    cv2.imshow('detector window', frame)

    # Press Q to quit!
    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break


# Cleanup
webcam.release()
cv2.destroyAllWindows()


Print("Completed")
