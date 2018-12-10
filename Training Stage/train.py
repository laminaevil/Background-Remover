import numpy as np
import h5py
import matplotlib.pyplot as plt
from model import unet
from time import time

# load data
dataset = h5py.File('dataset.h5', 'r')
X_pre = dataset['X_pre'][:]
Y_pre = dataset['Y_pre'][:]
dataset.close()

X_train = X_pre.astype('float32')
X_train = X_train / 255.0



