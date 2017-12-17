from sklearn.datasets import load_files       

from keras import applications  
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential

import pandas as pd
import numpy as np

train_dir = 'data_gen/train'
val_dir = 'data_gen/val'
test_dir = 'data/test'

img_width = 224
img_height = 224
batch_size = 8

## Bottleneck features
def save_bottleneck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG19 network
    model = applications.ResNet50(include_top=False, weights='imagenet', input_tensor=None, input_shape=None, pooling=None, classes=133)

    generator = datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

    bottleneck_features_train = model.predict_generator(
        generator, len(generator.filenames) // batch_size)
    np.save('bottleneck_features_train.npy', bottleneck_features_train)
    
    train_labels = generator.classes
    train_labels = to_categorical(train_labels, len(generator.filenames))

    generator = datagen.flow_from_directory(
        val_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(
        generator, len(generator.filenames) // batch_size)
    np.save('bottleneck_features_validation.npy', bottleneck_features_validation)

    validation_labels = generator.classes
    validation_labels = to_categorical(validation_labels, len(generator.filenames))

    return train_labels, validation_labels

## Create bottleneck features
train_labels, validation_labels = save_bottleneck_features()
train_data = np.load(open('bottleneck_features_train.npy'))
validation_data = np.load(open('bottleneck_features_validation.npy'))

## Model definition
model = Sequential()
model.add(Flatten(input_shape = train_data.shape[1:]))
model.add(Dense(133, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

## Training
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.hdf5', 
                               verbose=1, save_best_only=True)
epochs = 10

history = model.fit(train_data, train_labels,
                    epochs=epochs, batch_size=batch_size,
                    callbacks=[checkpointer], verbose=2)

## Testing
model.load_weights('saved_models/weights.best.hdf5')
