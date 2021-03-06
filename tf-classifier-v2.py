from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from keras import regularizers
import matplotlib.pyplot as plt
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
from PIL import Image

#initialise cnn
cf = Sequential()

#Creating the method for model
#Step 1- Convolution
cf.add(Convolution2D(32, (3, 3), input_shape = (256, 256,3), activation = 'relu'))
#adding another layer
cf.add(Convolution2D(32, (3, 3), activation = 'relu'))
#Max pooling it
cf.add(MaxPooling2D(pool_size = (2, 2)))
#prevent overfitting
cf.add(Dropout(0.25))

cf.add(Convolution2D(64, (3, 3), input_shape = (256, 256,3), activation = 'relu'))
#adding another layer
cf.add(Convolution2D(64, (3, 3), activation = 'relu'))
#Max pooling it
cf.add(MaxPooling2D(pool_size = (2, 2)))
#prevent overfitting
cf.add(Dropout(0.25))

#Adding another layer
#cf.add(Convolution2D(32, (3, 3), activation = 'relu'))
#Pooling
#cf.add(MaxPooling2D(pool_size = (2, 2)))
#Adding another layer
#cf.add(Convolution2D(32, (3, 3), activation = 'relu'))
#Pooling
#cf.add(MaxPooling2D(pool_size = (2, 2)))
# reduce to linear array
cf.add(Flatten())
#full connection
cf.add(Dense(units = 128, activation = 'relu', kernel_regularizer=regularizers.l2(0.01)))
cf.add(Dropout(0.5))
#for output step
cf.add(Dense(units = 6, activation = 'softmax'))
#compile cnn
cf.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'] )


#data augmentation and fitting the cnn to the images
train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, 
                                   horizontal_flip=True)
test_datagen = ImageDataGenerator(rescale=1./255)
training_set = train_datagen.flow_from_directory('dataset-resized/training_set', 
                                                 target_size=(256, 256), batch_size=32, 
                                                 class_mode='categorical')
test_set = test_datagen.flow_from_directory('dataset-resized/test_set', target_size=(256, 256), 
                                            batch_size=32, class_mode='categorical')


#num of images in training-set/32, num of images in test-set/32
plot_compare = cf.fit_generator(training_set, steps_per_epoch=(2024/32), epochs=32, 
                                validation_data=test_set, validation_steps=(503/32))
                                #, callbacks = [checkpointer])


cf.save('cnn-model-V2')

#plot data on graphs
plt.plot(plot_compare.history['loss'])
plt.plot(plot_compare.history['val_loss'])
plt.title('Model loss V2')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper right')
plt.show()

plt.plot(plot_compare.history['accuracy'])
plt.plot(plot_compare.history['val_accuracy'])
plt.title('Model accuracy V2')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='lower right')
plt.show()