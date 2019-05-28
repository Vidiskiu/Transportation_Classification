from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

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

#=================
# Preprocessing
#=================

from keras.preprocessing.image import ImageDataGenerator
training_datagen = ImageDataGenerator(rescale = 1./255, shear_range = 0.2, zoom_range = 0.2, horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = training_datagen.flow_from_directory('dataset/train', target_size = (50,50), batch_size = 32, class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('dataset/test', target_size = (50,50), batch_size = 32, class_mode = 'categorical')

classifier.fit_generator(training_set, steps_per_epoch = 100, epochs = 10, validation_data = test_set, validation_steps = 10)

# serialize model to JSON
classifier_json = classifier.to_json()
with open("classifier.json","w") as json_file:
    json_file.write(classifier_json)

# serialize weights
classifier.save_weights("classifier.h5")
print("Saved")