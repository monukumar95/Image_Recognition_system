# -*- coding: utf-8 -*-
"""
Created on Wed Aug 28 14:01:46 2019

@author: dell
"""

# Part 1 - Building the CNN
# Importing the Keras libraries and package

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN (object of cnn)

classifier = Sequential()

# step1 - convolution
classifier.add(Convolution2D(32, 3, 3, input_shape = (64,64,3), activation = 'relu'))

# step 2 pooling map this is reduceing the size of feature map

classifier.add(MaxPooling2D(pool_size = (2,2)))

#Adding second convolution layer for reduce the accurecy between train and test
classifier.add(Convolution2D(32, 3, 3, activation = 'relu'))

classifier.add(MaxPooling2D(pool_size = (2,2)))

# step 3 - Flattening
classifier.add(Flatten())

#step 4 - Fully connection with NN
classifier.add(Dense(output_dim = 128,activation = 'relu'))
classifier.add(Dense(output_dim = 1,activation = 'sigmoid'))

#compailing the cnn

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fiting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator 

train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('dataset/training_set',
                                                 target_size=(64, 64),
                                                 batch_size=32,
                                                 class_mode='binary')

test_set = test_datagen.flow_from_directory( 'dataset/test_set',
                                            target_size=(64, 64),
                                            batch_size=32,
                                            class_mode='binary')

classifier.fit_generator(training_set,
                    steps_per_epoch=8000,
                    epochs=25,
                    validation_data=test_set,
                    validation_steps=200)

# Part 3 Making new predictions on single image
import numpy as np
from keras.preprocessing import image
test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',target_size =(64,64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image,axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] == 1:
    prediction = 'dog'
else:
    prediction = 'cat'
    
