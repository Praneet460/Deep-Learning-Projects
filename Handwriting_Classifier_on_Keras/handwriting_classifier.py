
import keras
import tensorflow as tf
import utils

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten





pixel_width = 28
pixel_height = 28
pixel_depth = 1
number_of_classes = 10
batch_size = 32
epochs = 10
from keras.datasets import mnist

input_shape = (pixel_width, pixel_height, pixel_depth)


#  creating the tuple to divide the dataset into training and testing dataset
(features_train, labels_train), (features_test, labels_test) = mnist.load_data()
# reshape eact of the sample with the pixel depth of 1 because the image is greyscale
features_train = features_train.reshape(features_train.shape[0],pixel_width,pixel_height,pixel_depth)
features_test = features_test.reshape(features_test.shape[0],pixel_width,pixel_height,pixel_depth)
# CNN require these data that's why we did it
features_train = features_train.astype('float32')
features_test = features_test.astype('float32')
# convert every element into percentage
features_train /= 255
features_test /= 255
labels_train = keras.utils.to_categorical(labels_train,number_of_classes)
labels_test = keras.utils.to_categorical(labels_test, number_of_classes)



model = Sequential()
# we use 32 features i.e. 32 filtered images
# uses 3 by 3 grid and pass it over the image
# activate the relu i.e. convert negative values to 0
# set the shape of input images i.e. 28 by 28 by 1 stored in a tuple named input_shape
model.add(Conv2D(32, kernel_size=(3,3), activation = 'relu', input_shape = input_shape))
# print(model.output_shape)
# set the pool size to 2 by 2
model.add(MaxPooling2D(pool_size = (2,2)))
# randomize our data
# rate of droupout = 25%
model.add(Dropout(0.25))
# stack the nodes in the single column
# have 5,048 nodes
model.add(Flatten())
# convert into 128 nodes
model.add(Dense(128, activation = 'relu'))
# print(model.output_shape)
# convert in 10 nodes
model.add(Dense(number_of_classes, activation = 'softmax'))



# compile our model
model.compile(loss= keras.losses.categorical_crossentropy,
           optimizer = keras.optimizers.Adadelta(),
           metrics = ['accuracy'])
 # train our model
model.fit(features_train, labels_train,
        batch_size = batch_size,
        epochs = epochs,
        verbose = 1,
        validation_data = (features_test, labels_test))

score = model.evaluate(features_test, labels_test, verbose = 0)
# save the model with extension h5

model.save('/Users/Praneet/Downloads/handwriting_model.h5')
