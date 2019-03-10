import numpy as np
import warnings
from keras.layers import Input
from keras import layers
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalMaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import AveragePooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import BatchNormalization
from keras.models import Model
from keras.preprocessing import image
import keras.backend as K
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import preprocess_input
# from keras.applications.imagenet_utils import _obtain_input_shape
from keras.engine.topology import get_source_inputs

WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'

# def ResNet50(n_classes, img_width=224, img_height=224, channels=3
def build_model(n_classes, img_width=224, img_height=224, channels=3, include_top=True, pooling='avg'):
    
    if K.image_data_format() == 'channels_first':
        input_shape = (channels, img_width, img_height)
    else:
        input_shape = (img_width, img_height, channels)
    bn_axis = 3
    img_input = Input(shape=input_shape)
    x = ZeroPadding2D((3, 3))(img_input)
    x = Conv2D(64, (7, 7), strides=(2, 2), name='conv1')(x)
    x = BatchNormalization(axis=bn_axis, name='bn_conv1')(x)
    x = Activation('relu')(x)
    x = MaxPooling2D((3, 3), strides=(2, 2))(x)

    x = conv_block(x, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='b')
    x = identity_block(x, 3, [64, 64, 256], stage=2, block='c')

    x = conv_block(x, 3, [128, 128, 512], stage=3, block='a')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='b')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='c')
    x = identity_block(x, 3, [128, 128, 512], stage=3, block='d')

    x = conv_block(x, 3, [256, 256, 1024], stage=4, block='a')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='b')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='c')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='d')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='e')
    x = identity_block(x, 3, [256, 256, 1024], stage=4, block='f')

    x = conv_block(x, 3, [512, 512, 2048], stage=5, block='a')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='b')
    x = identity_block(x, 3, [512, 512, 2048], stage=5, block='c')

    x = AveragePooling2D((7, 7), name='avg_pool')(x)

    if include_top:
        x = Flatten()(x)
        x = Dense(n_classes, activation='softmax', name='fc1000')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Create model.
    model = Model(img_input, x, name='resnet50')

    return model

def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """conv_block is the block that has a conv layer at shortcut
    # Arguments
        input_tensor: input tensor
        kernel_size: defualt 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'..., current block label, used for generating layer names
    # Returns
        Output tensor for the block.
    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a')(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c')(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c')(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1')(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1')(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x





#    model = Sequential()
#    #ADD CRAZYNET HERE
#    l2_reg = 0.0
#    x = Input(shape=input_shape)
#    
#    conv1 = Conv2D(32, (5,5), padding="same",  
#        kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1')(x)
#    conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1)
#    conv1 = ELU(name='elu')(conv1)
#    pool1 = MaxPooling2D(name='pool1')(conv1)
#
#    conv2 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2')(pool1)
#    conv2 = BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
#    conv2 = ELU(name='elu2')(conv2)
#    pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(conv2)
#
#    conv3 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3')(pool2)
#    conv3 = BatchNormalization(axis=3, momentum=0.99, name='bn3')(conv3)
#    conv3 = ELU(name='elu3')(conv3)
#    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(conv3)
#    
#    conv4 = Conv2D(80, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4')(pool3)
#    conv4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
#    conv4 = ELU(name='elu4')(conv4)
#    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(conv4)
#    
#    conv5 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5')(pool4)
#    conv5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
#    conv5 = ELU(name='elu5')(conv5)
##    pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(conv5)
#
#    conv6 = Conv2D(64, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6')(conv5)
#    conv6 = BatchNormalization(axis=3, momentum=0.99, name='bn6')(conv6)
#    conv6 = ELU(name='elu6')(conv6)
#    pool6 = MaxPooling2D(pool_size=(2, 2), name='pool6')(conv6)
#
#    conv7 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7')(pool6)
#    conv7 = BatchNormalization(axis=3, momentum=0.99, name='bn7')(conv7)
#    conv7 = ELU(name='elu7')(conv7)
#    pool7 = MaxPooling2D(pool_size=(2, 2), name='pool7')(conv7)
#
#    conv8 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv8')(pool7)
#    conv8 = BatchNormalization(axis=3, momentum=0.99, name='bn8')(conv8)
#    conv8 = ELU(name='elu8')(conv8)
#    pool8 = MaxPooling2D(pool_size=(2, 2), name='pool8')(conv8)
#
#    conv9 = Conv2D(32, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv9')(pool8)
#    conv9 = BatchNormalization(axis=3, momentum=0.99, name='bn9')(conv9)
#    conv9 = ELU(name='elu9')(conv9)
#    
#    flat = Flatten()(conv9)
#    dense = Dense(256,  activation='relu')(flat)
#    drop = Dropout(0.1)(dense)
#
#    output = Dense(n_classes, activation='softmax')(drop)
#    # classes_softmax = Activation('softmax', name='classes_softmax')(classes_concat)
#    
#    model = Model(inputs=x, outputs=output)

#    model = tf.keras.models.Sequential()
#    model.add(tf.keras.layers.BatchNormalization(input_shape=input_shape))
#    model.add(tf.keras.layers.Conv2D(64, (5, 5), padding='same', activation='elu'))
#    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
#    model.add(tf.keras.layers.Dropout(0.25))
#
#    model.add(tf.keras.layers.BatchNormalization()
#    model.add(tf.keras.layers.Conv2D(128, (5, 5), padding='same', activation='elu'))
#    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))
#    model.add(tf.keras.layers.Dropout(0.25))
#
#    model.add(tf.keras.layers.BatchNormalization()
#    model.add(tf.keras.layers.Conv2D(256, (5, 5), padding='same', activation='elu'))
#    model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
#    model.add(tf.keras.layers.Dropout(0.25))
#
#    model.add(tf.keras.layers.Flatten())
#    model.add(tf.keras.layers.Dense(256))
#    model.add(tf.keras.layers.Activation('elu'))
#    model.add(tf.keras.layers.Dropout(0.5))
#    model.add(tf.keras.layers.Dense(n_classes))
#    model.add(tf.keras.layers.Activation('softmax'))

    # model = Sequential()
    # model.add(BatchNormalization(input_shape=input_shape))
    # model.add(Conv2D(64, (5, 5), padding='same', activation='elu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    # model.add(Dropout(0.25))

    # model.add(BatchNormalization(input_shape=input_shape))
    # model.add(Conv2D(128, (5, 5), padding='same', activation='elu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    # model.add(Dropout(0.25))

    # model.add(BatchNormalization(input_shape=input_shape))
    # model.add(Conv2D(256, (5, 5), padding='same', activation='elu'))
    # model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
    # model.add(Dropout(0.25))

    # model.add(Flatten())
    # model.add(Dense(256))
    # model.add(Activation('elu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(n_classes))
    # model.add(Activation('softmax'))
    
    
    # return model











    