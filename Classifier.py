from keras.models import load_model
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import cv2

# CNN init
classifier = Sequential()
# ConvL
classifier.add(Conv2D(32, (3,3), input_shape = (50,50,3), activation = 'relu'))
# PoolingL
classifier.add(MaxPooling2D(pool_size = (2,2)))
# ConvL
classifier.add(Conv2D(32, (3,3), input_shape = (50,50,3), activation = 'relu'))
# PoolingL
classifier.add(MaxPooling2D(pool_size = (2,2)))
# FlatteningL
classifier.add(Flatten())

# ANN
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 3, activation = 'sigmoid'))

# Compile CNN
classifier.compile(optimizer = 'adam', loss = 'mean_squared_error', metrics = ['accuracy'])

classifier.load_weights('classifier.h5')

video = cv2.VideoCapture("http://cctv-dishub.sukoharjokab.go.id/zm/cgi-bin/nph-zms?mode=jpeg&monitor=9&scale=100&maxfps=15&buffer=1000&user=user&pass=user")
ok, frame = video.read()
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

car_cascade = cv2.CascadeClassifier('cars1.xml')

cars = car_cascade.detectMultiScale(gray, 1.1, minNeighbors = 2)

id = 0
t = 100

while(True):
    ok, frame = video.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    cars = car_cascade.detectMultiScale(gray, 1.1, minNeighbors = 2)

    for (x,y,w,h) in cars: 
        if((y + h)/2 < 150 + t and (y + h)/2 > 150 - t):
            imcrop = frame[y:y+h+t, x:x+w+t]
            imcrop = cv2.resize(imcrop, (50,50))
            imcrop = imcrop.reshape([-1, 50, 50, 3])
            result = classifier.predict(imcrop) 
            y_classes = result.argmax(axis=-1)
            print(y_classes)
            if(y_classes == 0):
                cv2.putText(frame, "Car", ((x + w) / 2, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
                print("Car")
            elif(y_classes == 1):
                cv2.putText(frame, "Motor", ((x + w) / 2, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
                print("Motor")
            elif(y_classes == 2):
                cv2.putText(frame, "Truck", ((x + w) / 2, y - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,0,255), 2)
                print("Truck")
            id += 1
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)

    cv2.line(frame, (0, 150), (480, 150), (0,255,0), 2)

    cv2.imshow("Video", frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
      break

cv2.destroyAllWindows()



print(result)
