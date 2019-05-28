import cv2

video = cv2.VideoCapture("http://cctv-dishub.sukoharjokab.go.id/zm/cgi-bin/nph-zms?mode=jpeg&monitor=9&scale=100&maxfps=15&buffer=1000&user=user&pass=user")
ok, frame = video.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

car_cascade = cv2.CascadeClassifier('cars1.xml')

cars = car_cascade.detectMultiScale(gray, 1.1, minNeighbors = 2)

id = 0

t = 10

while(True):
    ok, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.1, minNeighbors = 2)

    for (x,y,w,h) in cars: 
        if((y + h)/2 < 150 + t or (y + h)/2 < 150 - t):
            imcrop = frame[y:y+h+t, x:x+w+t]
            print("captured")
            cv2.imwrite('dataset/'+str(id)+'test.jpg', imcrop)
            id += 1
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.line(frame, (0, 150), (480, 150), (0,255,0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

cv2.destroyAllWindows()