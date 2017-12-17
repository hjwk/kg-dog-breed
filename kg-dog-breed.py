from sklearn.datasets import load_files       

from keras import applications  
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint 

from keras.preprocessing import image                  
from tqdm import tqdm

import pandas as pd
import numpy as np

train_dir = 'data_gen/train'
val_dir = 'data_gen/validation'
test_dir = 'data/test'

img_width = 224
img_height = 224
batch_size = 16

def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)


## Bottleneck features
def save_bottleneck_features():
    datagen = ImageDataGenerator(preprocessing_function=applications.vgg16.preprocess_input)

    # build the network
    model = applications.VGG16(include_top=False, weights='imagenet')

    train_data = load_files(train_dir)
    train_tensors = path_to_tensor(train_data['filenames']).astype('float32')/255
    train_data = applications.vgg16.preprocess_input(train_data)

    generator = datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features_train = model.predict_generator(
        generator, len(generator.filenames) / batch_size)

    np.save('bottleneck_features/train.npy', bottleneck_features_train)

    generator = datagen.flow_from_directory(
        val_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features_validation = model.predict_generator(
        generator, len(generator.filenames) / batch_size)

    np.save('bottleneck_features/validation.npy', bottleneck_features_validation)

def load_labels(path):
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(
        path,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)
    labels = generator.classes
    labels = np_utils.to_categorical(labels, 120)
    return labels

## Create bottleneck features
save_bottleneck_features()
train_data = np.load('bottleneck_features/train.npy')
train_labels = load_labels('data_gen/train')
validation_data = np.load('bottleneck_features/validation.npy')
validation_labels = load_labels('data_gen/validation')

## Model definition
model = Sequential()
model.add(Flatten(input_shape = train_data.shape[1:]))
model.add(Dense(120, activation='softmax'))
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

## Training
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.hdf5', 
                               verbose=1, save_best_only=True)
epochs = 15
history = model.fit(train_data, train_labels,
                    validation_data=(validation_data, validation_labels),
                    epochs=epochs, batch_size=batch_size,
                    callbacks=[checkpointer], verbose=2)

## Testing
model.load_weights('saved_models/weights.best.hdf5')
