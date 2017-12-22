from sklearn.datasets import load_files       

from keras import applications
from keras.utils import np_utils
from keras.layers import Dropout, Flatten, Dense
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras.preprocessing import image                  

from tqdm import tqdm
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True  

import pandas as pd
import numpy as np
import glob
import os

train_dir = 'data_gen/train'
val_dir = 'data_gen/validation'
test_dir = 'data/test'

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

def generate_bottleneck_features():
    model = applications.ResNet50(include_top=False, weights='imagenet')

    train_files = load_files(train_dir)
    train_tensors = paths_to_tensor(train_files['filenames'])
    train_data = applications.resnet50.preprocess_input(train_tensors)
        
    bottleneck_features_train = model.predict(
        train_data, batch_size=16)

    np.save('bottleneck_features/train.npy', bottleneck_features_train)

    val_files = load_files(val_dir)
    val_tensors = paths_to_tensor(val_files['filenames'])
    val_data = applications.resnet50.preprocess_input(val_tensors)
    
    bottleneck_features_validation = model.predict(
        val_data, batch_size=16)

    np.save('bottleneck_features/validation.npy', bottleneck_features_validation)

def generate_bottleneck_features_test():
    # build the network
    model = applications.ResNet50(include_top=False, weights='imagenet')

    files = glob.glob('data/test/*.jpg')
    tensors = paths_to_tensor(files)
    data = applications.resnet50.preprocess_input(tensors)
        
    bottleneck_features = model.predict(
        data, batch_size=16)

    np.save('bottleneck_features/test.npy', bottleneck_features)

def load_labels(path):
    data = load_files(path)
    labels = np_utils.to_categorical(np.array(data['target']), 120)

    return labels

def extract_Resnet50(tensor):
	return ResNet50(weights='imagenet', include_top=False).predict(preprocess_input(tensor))

## Create bottleneck features
#generate_bottleneck_features()

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
model.add(Flatten(input_shape = train_data.shape[1:]))
model.add(Dense(120, activation='softmax'))
model.summary()
model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

## Training
checkpointer = ModelCheckpoint(filepath='saved_models/weights.best.hdf5', 
                               verbose=1, save_best_only=True)
epochs = 45
history = model.fit(train_data, train_labels,
                    validation_data=(validation_data, validation_labels),
                    epochs=epochs, batch_size=batch_size,
                    callbacks=[checkpointer], verbose=2)

## Testing
'''
model.load_weights('saved_models/weights.best.hdf5')

train_labels = np.array(pd.read_csv('data/labels.csv'))
classes, counts = np.unique(train_labels[:, 1], return_counts=True)

f = open('results.csv', 'w')
f.write('id')
for c in classes:
    f.write(',' + c)
f.write('\n')

#generate_bottleneck_features_test()
test = np.load('bottleneck_features/test.npy')
output = model.predict(test)
filenames = os.listdir('data/test')
for [o, name] in zip(output, filenames):
    f.write(name[:-4] + ',')
    o.tofile(f, sep=',', format='%.17f')
    f.write('\n')

f.close()
'''