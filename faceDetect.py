import cv2

trained_face_data = cv2.CascadeClassifier('./haarcascade_frontalface_default.xml')

webcam = cv2.VideoCapture(0)

while (True):

    sucess, frame = webcam.read() 
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    face_coordinates = trained_face_data.detectMultiScale(gray_img)

    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imshow('face detected', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break

print("Done")