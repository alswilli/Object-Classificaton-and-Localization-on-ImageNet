# from keras.models import Sequential # How layers interact (x -> y)
# from keras.layers import Dense      # Type of layer (what you feed your data into)

from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input
from keras import backend as K

def build_model():

    img_width, img_height = 224, 224

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    # model = Sequential()

    inputs = Input(shape = input_shape)



    conv1 = Conv2D(32, (3, 3)) (inputs)
    relu1 = Activation('relu') (conv1)
    mPool1 = MaxPooling2D(pool_size=(2, 2)) (relu1)

    conv2 = Conv2D(32, (3, 3)) (mPool1)
    relu2 = Activation('relu') (conv2)
    mPool2 = MaxPooling2D(pool_size=(2, 2)) (relu2)

    conv3 = Conv2D(64, (3, 3)) (mPool2)
    relu3 = Activation('relu') (conv3)
    mPool3 = MaxPooling2D(pool_size=(2, 2)) (relu3)

    #branch for the localization
    # outputs = Dense (4, name='BOX_PREDICTIONs')(mPool3)

    # branch for the classification
    flatten1 = Flatten() (mPool3)
    dense1 = Dense(64) (flatten1)
    relu4 = Activation('relu') (dense1)
    dropout1 = Dropout(0.5) (relu4)
    dense2 = Dense(5) (dropout1)

    sigmoid1 = Activation('sigmoid') (dense2)
    boxOutputs = Dense (4, name='BOX_PREDICTIONS')(dense2)
    totalOutputs = [sigmoid1, boxOutputs]

    model = Model(inputs=inputs, outputs=totalOutputs)

    # model.add(Conv2D(32, (3, 3), input_shape=input_shape))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(32, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(64, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Flatten())
    # model.add(Dense(64))
    # model.add(Activation('relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(5))
    # model.add(Activation('sigmoid'))


    # ##Fully Connected Layer 3 -- CLASSIFICATION HEAD
    # #Weights for FC Layer 3
    # w3_fc_1 = tf.Variable(tf.truncated_normal([nodes_fc2,output_classes], stddev=0.01))
    # #Bias for FC Layer 3b3_fc = tf.Variable( tf.constant(1.0, shape=[output_classes] ) )
    # b3_fc_1 = tf.Variable( tf.constant(1.0, shape=[output_classes] ) )
    # #Summing Matrix calculations and bias
    # y_pred_1 = tf.matmul(s_fc2, w3_fc_1) + b3_fc_1
    # #Applying RELU
    # print(y_pred_1)

    # ##Fully Connected Layer 3 -- REGRESSION HEAD
    # #Weights for FC Layer 3
    # w3_fc_2 = tf.Variable(tf.truncated_normal([nodes_fc2,output_locations], stddev=0.01))
    # #Bias for FC Layer 3b3_fc = tf.Variable( tf.constant(1.0, shape=[output_classes] ) )
    # b3_fc_2 = tf.Variable( tf.constant(1.0, shape=[output_locations] ) )
    # #Summing Matrix calculations and bias
    # y_pred_2 = tf.matmul(s_fc2, w3_fc_2) + b3_fc_2
    # #Applying RELU
    # print(y_pred_2)

    # #Defining Classification function
    # cross_entropy = tf.multiply(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_true_1,logits=y_pred_1)),10)

    # #Defining Regression Loss
    # regression_loss = tf.multiply(tf.reduce_mean(tf.square(y_pred_2 - y_true_2)),1.0)

    # #Defining total loss
    # final_loss = cross_entropy + regression_loss

    # #Defining objective
    # train = tf.train.AdamOptimizer(learning_rate=0.00001).minimize(final_loss)


    lossFunctionList = ['binary_crossentropy', 'mse']

    model.compile(loss=lossFunctionList,
                  optimizer='adam',
                  metrics=['accuracy'],
                  loss_weights = [0.9, 0.1])


    # model.compile(loss='binary_crossentropy',
    #               optimizer='adam',
    #               metrics=['accuracy'])

    # model = Sequential()
    # model.add(Dense(units=256, activation="softmax"))

    # # LATER: Specify input shape!!

    # #define metrics, loss function (maybe our own?), gradient descent
    # model.compile(loss = "categorical_crossentropy", metrics = ["accuracy"], optimizer = "adam")

    # # Train model
    model.fit(x_train, [y_train, y_train_boxes], epochs = 10, verbose=1)

    # model.fit(x_train, y_train, epochs = 1, verbose=1)

    # Validate

    # Evaluate

    return model