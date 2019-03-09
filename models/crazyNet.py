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
    
    layer_sizes = [48, 64, 64, 96, 96, 48, 48]

    conv1 = Conv2D(32, (5,5), padding="same",  
        kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1')(x)
    conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1)
    conv1 = ELU(name='elu')(conv1)
    pool1 = MaxPooling2D(name='pool1')(conv1)

    i = 2
    last_pool = pool1
    # for size in layer_sizes:
    #     conv = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv{0}'.format(i))(last_pool)
    #     conv = BatchNormalization(axis=3, momentum=0.99, name='bn{0}'.format(i))(conv)
    #     conv = ELU(name='elu{0}'.format(i))(conv)
    #     last_pool = MaxPooling2D(pool_size=(2, 2), name='pool{0}'.format(i))(conv)


    conv2 = Conv2D(layer_sizes[i-2], (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv{0}'.format(i))(pool1)
    conv2 = BatchNormalization(axis=3, momentum=0.99, name='bn{0}'.format(i))(conv2)
    conv2 = ELU(name='elu{0}'.format(i))(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool{0}'.format(i))(conv2)
    i+=1

    conv3 = Conv2D(layer_sizes[i-2], (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv{0}'.format(i))(pool2)
    conv3 = BatchNormalization(axis=3, momentum=0.99, name='bn{0}'.format(i))(conv3)
    conv3 = ELU(name='elu{0}'.format(i))(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool{0}'.format(i))(conv3)
    i+=1

    conv4 = Conv2D(layer_sizes[i-2], (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv{0}'.format(i))(pool3)
    conv4 = BatchNormalization(axis=3, momentum=0.99, name='bn{0}'.format(i))(conv4)
    conv4 = ELU(name='elu{0}'.format(i))(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool{0}'.format(i))(conv4)
    i+=1

    conv5 = Conv2D(layer_sizes[i-2], (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv{0}'.format(i))(pool4)
    conv5 = BatchNormalization(axis=3, momentum=0.99, name='bn{0}'.format(i))(conv5)
    conv5 = ELU(name='elu{0}'.format(i))(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name='pool{0}'.format(i))(conv5)
    i+=1

    conv6 = Conv2D(layer_sizes[i-2], (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv{0}'.format(i))(pool5)
    conv6 = BatchNormalization(axis=3, momentum=0.99, name='bn{0}'.format(i))(conv6)
    conv6 = ELU(name='elu{0}'.format(i))(conv6)
    pool6 = MaxPooling2D(pool_size=(2, 2), name='pool{0}'.format(i))(conv6)
    i+=1

    conv7 = Conv2D(layer_sizes[i-2], (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv{0}'.format(i))(pool6)
    conv7 = BatchNormalization(axis=3, momentum=0.99, name='bn{0}'.format(i))(conv7)
    conv7 = ELU(name='elu{0}'.format(i))(conv7)
    pool7 = MaxPooling2D(pool_size=(2, 2), name='pool{0}'.format(i))(conv7)
    i+=1

    conv8 = Conv2D(layer_sizes[i-2], (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv{0}'.format(i))(pool7)
    conv8 = BatchNormalization(axis=3, momentum=0.99, name='bn{0}'.format(i))(conv8)
    conv8 = ELU(name='elu{0}'.format(i))(conv8)
    # pool8 = MaxPooling2D(pool_size=(2, 2), name='pool{0}'.format(i))(conv8)
    i+=1

   

    conv_last = Conv2D(32, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv_last')(conv8)
    conv_last = BatchNormalization(axis=3, momentum=0.99, name='bn_last')(conv_last)
    conv_last = ELU(name='elu_last')(conv_last)
    
    flat = Flatten()(conv_last)
    dense = Dense(256,  activation='relu')(flat)
    drop = Dropout(0.1)(dense)

    output = Dense(n_classes, activation='softmax')(drop)
    # classes_softmax = Activation('softmax', name='classes_softmax')(classes_concat)
    
    model = Model(inputs=x, outputs=output)
    
    
    return model


def build_model2(n_classes, img_width=224, img_height=224, channels=3):
    if K.image_data_format() == 'channels_first':
        input_shape = (channels, img_width, img_height)
    else:
        input_shape = (img_width, img_height, channels)

    input_shape = (3, 224, 224)

    model = Sequential()
    #ADD CRAZYNET HERE
    l2_reg = 0.0
    # x = Input(shape=input_shape)
    
    layer_sizes = [48, 64, 64, 96, 48]

    model.add(Conv2D(32, (5,5), padding="same",  
        kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1', input_shape=input_shape))
    model.add(BatchNormalization(axis=3, momentum=0.99, name='bn1'))
    model.add(ELU(name='elu'))
    model.add(MaxPooling2D(name='pool1', data_format=K.image_data_format()))

    i = 2
    
    for size in layer_sizes:
        model.add(Conv2D(size, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv{0}'.format(i)))
        model.add(BatchNormalization(axis=3, momentum=0.99, name='bn{0}'.format(i)))
        model.add(ELU(name='elu{0}'.format(i)))
        if i%2==0:
            model.add(MaxPooling2D(pool_size=(2, 2), name='pool{0}'.format(i), data_format=K.image_data_format()))

        i+=1

   

    model.add(Conv2D(32, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv_last'))
    model.add(BatchNormalization(axis=3, momentum=0.99, name='bn_last'))
    model.add(ELU(name='elu_last'))
    
    model.add(Flatten())
    model.add(Dense(256,  activation='relu'))
    model.add(Dropout(0.1))

    model.add(Dense(n_classes, activation='softmax'))
    # classes_softmax = Activation('softmax', name='classes_softmax')(classes_concat)

    
    
    return model

def build_model3(n_classes, img_width=224, img_height=224, channels=3):
    if K.image_data_format() == 'channels_first':
        input_shape = (channels, img_width, img_height)
    else:
        input_shape = (img_width, img_height, channels)


    # model = Sequential()
    #ADD CRAZYNET HERE
    l2_reg = 0.0
    x = Input(shape=input_shape)
    
#    layer_sizes = [48, 64, 64, 80, 80, 96, 96, 112, 112, 128, 128, 128, 128, 112, 112, 96, 96, 80, 80, 64, 64, 48, 48]
    layer_sizes = [48, 64, 64, 48, 48]

    conv1 = Conv2D(32, (5,5), padding="same",  
        kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1')(x)
    conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1)
    conv1 = ELU(name='elu')(conv1)
    drop1 = Dropout(0.05)(conv1)
    pool1 = MaxPooling2D(name='pool1')(drop1)
    

    i = 2
    last_pool = pool1
    for size in layer_sizes:
        conv = Conv2D(size, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv{0}'.format(i))(last_pool)
        
        conv = BatchNormalization(axis=3, momentum=0.99, name='bn{0}'.format(i))(conv)
        
        conv = ELU(name='elu{0}'.format(i))(conv)
        
        last_pool = MaxPooling2D(pool_size=(2, 2), name='pool{0}'.format(i))(conv)
        
#        if(i%3 == 0):
#            conv = Dropout(0.25)(conv)
        
#        if i%1==0:
#            if i == 5:
#                
#            last_pool = MaxPooling2D(pool_size=(2, 2), name='pool{0}'.format(i))(conv)
            
        i+=1


    conv_last = Conv2D(32, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv_last')(last_pool)
    conv_last = BatchNormalization(axis=3, momentum=0.99, name='bn_last')(conv_last)
    conv_last = ELU(name='elu_last')(conv_last)
    
    flat = Flatten()(conv_last)
    dense = Dense(128,  activation='relu')(flat)
    drop = Dropout(0.15)(dense)

    dense = Dense(256,  activation='relu')(drop)
    drop = Dropout(0.15)(dense)

    dense = Dense(512,  activation='relu')(drop)
    drop = Dropout(0.15)(dense)
    
    dense = Dense(256,  activation='relu')(drop)
    drop = Dropout(0.25)(dense)

    dense = Dense(128,  activation='relu')(drop)
    drop = Dropout(0.25)(dense)

    output = Dense(n_classes, activation='softmax')(drop)
    # classes_softmax = Activation('softmax', name='classes_softmax')(classes_concat)
    
    model = Model(inputs=x, outputs=output)
    return model