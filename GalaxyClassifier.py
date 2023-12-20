import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

#load data
database = np.load('galaxydata.npz')
input_data, labels = database['data'], database['labels']

#print data to see it's shape
print(input_data.shape) #as we see images are 128x128 RGB
print(labels.shape)

#split for training and testing
x_train, x_valid, y_train, y_valid = train_test_split(input_data, labels, test_size=0.20, stratify=labels, shuffle=True, random_state=222)

#load images and rescale
data_generator = ImageDataGenerator(rescale=1./255)

#create training and testing data iterators
training_iterator = data_generator.flow(x_train, y_train,batch_size=5)
validation_iterator = data_generator.flow(x_valid, y_valid, batch_size=5)

#Create model
model = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(128,128,3)))

#add convolutional layers and max pooling ones to optimize params
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2, 2), strides=(2,2)))
model.add(tf.keras.layers.Conv2D(8, 3, strides=2, activation="relu")) 
model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2,2), strides=(2,2)))
	
#add flatten layer to convert 2D matrix into 1D
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(16, activation="relu"))
model.add(tf.keras.layers.Dense(4, activation='softmax'))
print(model.summary())

#compile model
model.compile(
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
  loss=tf.keras.losses.CategoricalCrossentropy(), 
  metrics=
  [tf.keras.metrics.CategoricalAccuracy(),tf.keras.metrics.AUC()])

#Train Model
model.fit(
  training_iterator, 
  steps_per_epoch=len(x_train)/5,
  epochs=8,
  validation_data=validation_iterator,
  validation_steps=len(x_valid)/5,
  verbose=1)