from sklearn.datasets import load_files       

from keras import applications  
from keras.utils import np_utils
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img  
from keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint

from keras.preprocessing import image                  
from tqdm import tqdm

from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True  

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

def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

## Bottleneck features
def save_bottleneck_features():
    #datagen = ImageDataGenerator(preprocessing_function=applications.vgg16.preprocess_input)

    # build the network
    model = applications.VGG16(include_top=False, weights='imagenet')

    train_files = load_files(train_dir)
    train_tensors = paths_to_tensor(train_files['filenames'])#.astype('float32')/255
    train_data = applications.vgg16.preprocess_input(train_tensors)

    '''generator = datagen.flow_from_directory(
        train_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)'''
        
    bottleneck_features_train = model.predict(
        train_data, batch_size=16)

    np.save('bottleneck_features/train.npy', bottleneck_features_train)

    val_files = load_files(val_dir)
    val_tensors = paths_to_tensor(val_files['filenames'])#.astype('float32')/255
    val_data = applications.vgg16.preprocess_input(val_tensors)
    
    #generator = datagen.flow_from_directory(
    #    val_dir,
    #    target_size=(img_width, img_height),
    #    batch_size=batch_size,
    #    class_mode=None,
    #    shuffle=False)

    bottleneck_features_validation = model.predict(
        val_data, batch_size=16)

    np.save('bottleneck_features/validation.npy', bottleneck_features_validation)

def load_labels(path):
    data = load_files(path)
    labels = np_utils.to_categorical(np.array(data['target']), 120)

    return labels

## Create bottleneck features
#save_bottleneck_features()

## Load bottleneck features
print('Loading training bottleneck features')
train_data = np.load('bottleneck_features/train.npy')
train_labels = load_labels('data_gen/train')

print('Loading validation bottleneck features')
validation_data = np.load('bottleneck_features/validation.npy')
validation_labels = load_labels('data_gen/validation')

## Model definition
print('Defining model')
model = Sequential()
model.add(GlobalAveragePooling2D(input_shape = train_data.shape[1:]))
model.add(Dense(512, activation='relu'))
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
