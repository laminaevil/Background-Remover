from keras.models import Model
# https://keras.io/models/model/
from keras.layers import Input, Conv2D, MaxPooling2D, Dropout, UpSampling2D, concatenate
# https://keras.io/layers/core/
from keras.optimizers import Adam


def unet(input_size = (256,256,3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, (3,3), activation = 'relu')(inputs)
    conv1 = Conv2D(64, 3, activation = 'relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(128, 3, activation = 'relu')(pool1)
    conv2 = Conv2D(128, 3, activation = 'relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    
    conv3 = Conv2D(256, 3, activation = 'relu')(pool2)
    conv3 = Conv2D(256, 3, activation = 'relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    
    conv4 = Conv2D(512, 3, activation = 'relu')(pool3)
    conv4 = Conv2D(512, 3, activation = 'relu')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation = 'relu')(pool4)
    conv5 = Conv2D(1024, 3, activation = 'relu')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation = 'relu')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([drop4,up6], axis = 3)
    conv6 = Conv2D(512, 3, activation = 'relu')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu')(conv6)

    up7 = Conv2D(256, 2, activation = 'relu')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([conv3,up7], axis = 3)
    conv7 = Conv2D(256, 3, activation = 'relu')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu')(conv7)

    up8 = Conv2D(128, 2, activation = 'relu')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([conv2,up8], axis = 3)
    conv8 = Conv2D(128, 3, activation = 'relu')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu')(conv8)

    up9 = Conv2D(64, 2, activation = 'relu')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([conv1,up9], axis = 3)
    conv9 = Conv2D(64, 3, activation = 'relu')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)

    model = Model(input = inputs, output = conv10)

    model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
    
    return model




