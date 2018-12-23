import numpy as np
import h5py
import matplotlib.pyplot as plt
import gc; gc.enable() # memory is tight
import os

from .env import unet
from time import time

import tensorflow as tf
import keras.backend.tensorflow_backend as KTF
import cv2

from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization, Dropout
from keras.optimizers import Adam, SGD, Adadelta, Adamax,Nadam
from keras.callbacks import ModelCheckpoint
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras.models import load_model


get_ipython().run_line_magic('matplotlib', 'inline')

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # 限定GPU的使用比例
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

X_load, Y_load = loadData('training_dataset_gray_5.h5')

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

X_dev, Y_dev = loadData('dev_dataset_gray_1.h5')

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

model = load_model('15test.h5', compile=False)

model.summary()
model.compile(optimizer=Adam(lr=3e-4), loss='binary_crossentropy', metrics=['accuracy'])
model_checkpoint = [ModelCheckpoint('callback_model.h5',verbose=1, monitor='val_loss',  mode='auto',save_best_only=True)]
history = model.fit(x = X_train, y = Y_train, batch_size = 4, callbacks = model_checkpoint,epochs=10, verbose=1, validation_split=0.0, validation_data=(X_dev, Y_dev), shuffle=True, steps_per_epoch=None, validation_steps=None)

model.save('test.h5')

result = model.evaluate(X_train, Y_train)
print ('Accuracy of training Set:',result[1])
print ('Loss of training set:', result[0])
print ()
result = model.evaluate(X_dev, Y_dev)
print ('Accuracy of dev Set:',result[1])
print ('Loss of dev set:', result[0])

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

X_load1, Y_load1 = loadData('training_dataset_gray_1.h5')
print (X_load1.shape, Y_load1.shape)
result1 = model.evaluate(X_load1, Y_load1)
print ('Accuracy of training Set1:',result1[1])
print ('Loss of training set1:', result1[0])

X_load2, Y_load2 = loadData('training_dataset_gray_2.h5')
print (X_load2.shape, Y_load2.shape)
result2 = model.evaluate(X_load2, Y_load2)
print ('Accuracy of training Set2:',result2[1])
print ('Loss of training set2:', result2[0])

X_load3, Y_load3 = loadData('training_dataset_gray_3.h5')
print (X_load3.shape, Y_load3.shape)
result3 = model.evaluate(X_load3, Y_load3)
print ('Accuracy of training Set3:',result3[1])
print ('Loss of training set3:', result3[0])

X_load4, Y_load4 = loadData('training_dataset_gray_4.h5')
print (X_load4.shape, Y_load4.shape)
result4 = model.evaluate(X_load4, Y_load4)
print ('Accuracy of training Set4:',result4[1])
print ('Loss of training set4:', result4[0])

X_load5, Y_load5 = loadData('training_dataset_gray_5.h5')
print (X_load5.shape, Y_load5.shape)
result5 = model.evaluate(X_load5, Y_load5)
print ('Accuracy of training Set5:',result5[1])
print ('Loss of training set5:', result5[0])

X_load6, Y_load6 = loadData('training_dataset_gray_6.h5')
print (X_load6.shape, Y_load6.shape)
result6 = model.evaluate(X_load6, Y_load6)
print ('Accuracy of training Set6:',result6[1])
print ('Loss of training set6:', result6[0])

result = (np.array(result1)+np.array(result2)+np.array(result3)+np.array(result4)+np.array(result5)+np.array(result6))/6
print ('Accuracy of training set:',result[1])
print ('Loss of training set:', result[0])

