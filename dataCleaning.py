import pandas as pd
import numpy as np

import os, shutil

original_train_dir = 'data/train'
test_dir = 'data/test

train_labels = np.array(pd.read_csv('data/labels.csv'))
classes, counts = np.unique(train_labels[:, 1], return_counts=True)

print("There are %d" % classes.size)

def mkdirIfNotExist(directory):
    if not os.path.exists(directory):
        os.mkdir(directory)
    return directory

base_dir = mkdirIfNotExist('./data_gen')
train_dir = mkdirIfNotExist(os.path.join(base_dir, 'train'))
validation_dir = mkdirIfNotExist(os.path.join(base_dir, 'validation'))
test_dir = mkdirIfNotExist(os.path.join(base_dir, 'test'))
for c in classes[:]:
    mkdirIfNotExist(os.path.join(train_dir, c))
    mkdirIfNotExist(os.path.join(validation_dir, c))

def copyIfNotExist(fnames, src_dir, dst_dir):
    nCopied = 0
    for fname in fnames:
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if not os.path.exists(dst):
            shutil.copyfile(src, dst)
            nCopied += 1
    if nCopied > 0:
        print("Copied %d to %s" % (nCopied, dst_dir))

# This will split available labeled data to train-validation sets
train_ratio = 0.7
for c in classes[:]:
    fnames = train_labels[train_labels[:, 1] == c][:, 0]
    fnames = ['{}.jpg'.format(name) for name in fnames]
    idx = int(len(fnames) * (1 - train_ratio))
    val_fnames = fnames[:idx]
    train_fnames = fnames[idx:]
    train_class_dir = os.path.join(train_dir, c)
    validation_class_dir = os.path.join(validation_dir, c)
    copyIfNotExist(train_fnames, original_train_dir, train_class_dir)
    copyIfNotExist(val_fnames, original_train_dir, validation_class_dir)