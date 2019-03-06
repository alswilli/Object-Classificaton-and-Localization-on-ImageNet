from keras.models import Model
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape, BatchNormalization, ELU, Reshape, Concatenate, Activation, Input, Lambda
from keras.regularizers import l2
from keras import backend as K


def build_model(n_classes, img_width=224, img_height=224, channels=3):
    if K.image_data_format() == 'channels_first':
        input_shape = (channels, img_width, img_height)
    else:
        input_shape = (img_width, img_height, channels)


    model = Sequential()
    #ADD CRAZYNET HERE
    l2_reg = 0.0
    x = Input(shape=input_shape)
    
    conv1 = Conv2D(32, (5,5), padding="same",  
        kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1')(x)
    conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1)
    conv1 = ELU(name='elu')(conv1)
    pool1 = MaxPooling2D(name='pool1')(conv1)

    conv2 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2')(pool1)
    conv2 = BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
    conv2 = ELU(name='elu2')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)

    conv3 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3')(pool2)
    conv3 = BatchNormalization(axis=3, momentum=0.99, name='bn3')(conv3)
    conv3 = ELU(name='elu3')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)

    conv4 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4')(pool3)
    conv4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
    conv4 = ELU(name='elu4')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)

    conv5 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5')(pool4)
    conv5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
    conv5 = ELU(name='elu5')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5)

    conv6 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6')(pool5)
    conv6 = BatchNormalization(axis=3, momentum=0.99, name='bn6')(conv6)
    conv6 = ELU(name='elu6')(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2), name='pool6')(conv6)

    conv7 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7')(pool6)
    conv7 = BatchNormalization(axis=3, momentum=0.99, name='bn7')(conv7)
    conv7 = ELU(name='elu7')(conv7)
    
    flat = Flatten()(conv7)
    dense = Dense(256,  activation='relu')(flat)
    drop = Dropout(0.1)(dense)

    output = Dense(n_classes, activation='softmax')(drop)
    # classes_softmax = Activation('softmax', name='classes_softmax')(classes_concat)
    
    model = Model(inputs=x, outputs=output)
    
    
    return model
