import numpy as np
import h5py
import cv2
import matplotlib.pyplot as plt
import gc; gc.enable() # memory is tight

from time import time
get_ipython().run_line_magic('matplotlib', 'inline')
from .env import unet
from keras.preprocessing.image import ImageDataGenerator
from data import *
from keras.callbacks import ModelCheckpoint

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD, Adadelta, Adamax,Nadam
from keras.callbacks import ModelCheckpoint

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.8 # 限定GPU的使用比例
session = tf.Session(config=config)

KTF.set_session(session)

# load data
def loadData(dataname = ' '):
    dataset = h5py.File(dataname, 'r')
    X_pre = dataset['X_pre'][:]
    Y_pre = dataset['Y_pre'][:]
    dataset.close()
    # normalization
    X_pre = X_pre.astype('float32')
    cv2.normalize(X_pre, X_pre,0, 1, cv2.NORM_MINMAX)
    Y_pre = Y_pre.astype('float32')
    cv2.normalize(Y_pre,  Y_pre ,0, 1, cv2.NORM_MINMAX)
    return X_pre, Y_pre

X_load, Y_load = loadData('training_dataset_gray.h5')
print (X_load.shape, Y_load.shape)

Y_show = np.append(Y_load, Y_load, axis = 3)
Y_show = np.append(Y_show, Y_load, axis = 3)

fig, axarr = plt.subplots(10, 2, figsize=(10, 50))
[idx.axis('off') for idx in axarr.flatten()]
for i in range(axarr.shape[0]):    
    axarr[i,0].imshow(X_load[i])
    axarr[i,1].imshow((Y_show[i]))

plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()

X_train = X_load#[0:20]
Y_train = Y_load#[0:20]
print (Y_train.shape)

X_dev, Y_dev = loadData('dev_dataset_gray.h5')
print (X_dev.shape, Y_dev.shape)

Y_dev_show = np.append(Y_dev, Y_dev, axis = 3)
Y_dev_show = np.append(Y_dev_show, Y_dev, axis = 3)

fig, axarr = plt.subplots(10, 2, figsize=(10, 50))
[idx.axis('off') for idx in axarr.flatten()]

for i in range(axarr.shape[0]):    
    axarr[i,0].imshow(X_dev[i])
    axarr[i,1].imshow(np.squeeze(Y_dev_show[i]))

plt.tight_layout(h_pad=0.1, w_pad=0.1)
plt.show()
model = unet(num_classes=1, learning_rate = 3e-4)
model.summary()
model_checkpoint = [ModelCheckpoint('callback_model.h5',verbose=1, monitor='val_loss',  mode='auto',save_best_only=True)]
history = model.fit(x = X_train, y = Y_train, batch_size = 5, callbacks = model_checkpoint,epochs=5, verbose=1, validation_split=0.0, validation_data=(X_dev, Y_dev), shuffle=True, steps_per_epoch=None, validation_steps=None)
model.save('test.h5')

result = model.evaluate(X_train, Y_train)
print ('Accuracy of training Set:',result[1])
print ('Loss of training set:', result[0])
print ()
result = model.evaluate(X_dev, Y_dev)
print ('Accuracy of Testing Set:',result[1])
print ('Loss of testing set:', result[0])

plt.plot( history.history['val_loss'], color = 'blue', label = 'dev_loss')
plt.plot( history.history['loss'], color = 'red', label = 'training_loss')
plt.legend( loc="upper right")
plt.ylabel('loss')
plt.xlabel('Epoch')
plt.savefig('loss.png')
plt.show()

plt.plot( history.history['val_acc'], color = 'blue', label = 'dev_acc')
plt.plot( history.history['acc'], color = 'red', label = 'training_acc')
plt.legend(loc="lower right")
plt.xlabel('Epoch')
plt.ylabel('accuracy')
plt.savefig('accuracy.png')
plt.show()
