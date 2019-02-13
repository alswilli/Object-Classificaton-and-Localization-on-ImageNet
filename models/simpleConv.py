from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape
from keras import backend as K


def build_model(img_width=224, img_height=224, channels=3):
    if K.image_data_format() == 'channels_first':
        input_shape = (channels, img_width, img_height)
    else:
        input_shape = (img_width, img_height, channels)
        
    model = Sequential()
    # model.add(Reshape(img_width*img_height*3, input_shape=input_shape))
    model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    # model.add(Dense(256))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.1))
    # model.add(Dense(256))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.25))

    model.add(Dense(5))
    model.add(Activation('softmax'))

    return model