# # from keras.models import Sequential # How layers interact (x -> y)
# # from keras.layers import Dense      # Type of layer (what you feed your data into)

# from keras.models import Sequential
# from keras.layers import Conv2D, MaxPooling2D
# from keras.layers import Activation, Dropout, Flatten, Dense
# from keras import backend as K

# img_width, img_height = 224, 224

# if K.image_data_format() == 'channels_first':
#     input_shape = (3, img_width, img_height)
# else:
#     input_shape = (img_width, img_height, 3)

# model = Sequential()
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

# model.compile(loss='binary_crossentropy',
#               optimizer='adam',
#               metrics=['accuracy'])

# # model = Sequential()
# # model.add(Dense(units=256, activation="softmax"))

# # # LATER: Specify input shape!!

# # #define metrics, loss function (maybe our own?), gradient descent
# # model.compile(loss = "categorical_crossentropy", metrics = ["accuracy"], optimizer = "adam")

# # # Train model
# model.fit(x_train, y_train, epochs = 1, verbose=1)

# # Validate

# # Evaluate

import numpy as np
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation
from keras.regularizers import l2

from keras_layer_AnchorBoxes import AnchorBoxes

def build_model():

	aspect_ratios_global=[0.5, 1.0, 2.0]
	aspect_ratios_per_layer=None
	two_boxes_for_ar1=True
	l2_reg = 0
	img_height, img_width = 224, 224
	img_channels = 3
	n_predictor_layers = 4 # The number of predictor conv layers in the network
	n_boxes = [4] * n_predictor_layers #[4, 4, 4, 4]
	steps=None
	offsets=None
	limit_boxes=True
	variances=[1.0, 1.0, 1.0, 1.0]
	coords='centroids'
	normalize_coords=False

	min_scale=0.1
	max_scale=0.9
	scales=None
	variances = np.array(variances)

	n_classes = 5
	n_classes += 1 # Account for the background class.

	scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)
	aspect_ratios = [aspect_ratios_global] * n_predictor_layers

	steps = [None] * n_predictor_layers
	offsets = [None] * n_predictor_layers

	# Compute the number of boxes to be predicted per cell for each predictor layer.
	# We need this so that we know how many channels the predictor layers need to have.
	# if aspect_ratios_per_layer:
	#     n_boxes = []
	#     for ar in aspect_ratios_per_layer:
	#         if (1 in ar) & two_boxes_for_ar1:
	#             n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
	#         else:
	#             n_boxes.append(len(ar))
	# else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
	#     if (1 in aspect_ratios_global) & two_boxes_for_ar1:
	#         n_boxes = len(aspect_ratios_global) + 1
	#     else:
	#         n_boxes = len(aspect_ratios_global)
	#     n_boxes = [n_boxes] * n_predictor_layers

	x = Input(shape=(img_height, img_width, img_channels)) #shape of rgb image expected by the CNN

	# The following identity layer is only needed so that subsequent lambda layers can be optional.
	x1 = Lambda(lambda z: z,
				output_shape=(img_height, img_width, img_channels),
				name='idendity_layer')(x)
	# if not (subtract_mean is None):
	#     x1 = Lambda(lambda z: z - np.array(subtract_mean),
	#                output_shape=(img_height, img_width, img_channels),
	#                name='input_mean_normalization')(x1)
	# if not (divide_by_stddev is None):
	#     x1 = Lambda(lambda z: z / np.array(divide_by_stddev),
	#                output_shape=(img_height, img_width, img_channels),
	#                name='input_stddev_normalization')(x1)
	# if swap_channels and (img_channels == 3):
	#     x1 = Lambda(lambda z: z[...,::-1],
	#                output_shape=(img_height, img_width, img_channels),
	#                name='input_channel_swap')(x1)

	conv1 = Conv2D(32, (5, 5), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1')(x1)
	conv1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1) # Tensorflow uses filter format [filter_height, filter_width, in_channels, out_channels], hence axis = 3
	conv1 = ELU(name='elu1')(conv1)
	pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(conv1)

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

	# The next part is to add the convolutional predictor layers on top of the base network
	# that we defined above. Note that I use the term "base network" differently than the paper does.
	# To me, the base network is everything that is not convolutional predictor layers or anchor
	# box layers. In this case we'll have four predictor layers, but of course you could
	# easily rewrite this into an arbitrarily deep base network and add an arbitrary number of
	# predictor layers on top of the base network by simply following the pattern shown here.

	# Build the convolutional predictor layers on top of conv layers 4, 5, 6, and 7.
	# We build two predictor layers on top of each of these layers: One for class prediction (classification), one for box coordinate prediction (localization)
	# We precidt `n_classes` confidence values for each box, hence the `classes` predictors have depth `n_boxes * n_classes`
	# We predict 4 box coordinates for each box, hence the `boxes` predictors have depth `n_boxes * 4`
	# Output shape of `classes`: `(batch, height, width, n_boxes * n_classes)`
	classes4 = Conv2D(n_boxes[0] * n_classes, (3, 3), strides=(1, 1), padding="valid", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes4')(conv4)
	classes5 = Conv2D(n_boxes[1] * n_classes, (3, 3), strides=(1, 1), padding="valid", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes5')(conv5)
	classes6 = Conv2D(n_boxes[2] * n_classes, (3, 3), strides=(1, 1), padding="valid", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes6')(conv6)
	classes7 = Conv2D(n_boxes[3] * n_classes, (3, 3), strides=(1, 1), padding="valid", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='classes7')(conv7)
	# Output shape of `boxes`: `(batch, height, width, n_boxes * 4)`
	boxes4 = Conv2D(n_boxes[0] * 4, (3, 3), strides=(1, 1), padding="valid", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes4')(conv4)
	boxes5 = Conv2D(n_boxes[1] * 4, (3, 3), strides=(1, 1), padding="valid", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes5')(conv5)
	boxes6 = Conv2D(n_boxes[2] * 4, (3, 3), strides=(1, 1), padding="valid", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes6')(conv6)
	boxes7 = Conv2D(n_boxes[3] * 4, (3, 3), strides=(1, 1), padding="valid", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes7')(conv7)

	# Generate the anchor boxes
	# Output shape of `anchors`: `(batch, height, width, n_boxes, 8)`
	anchors4 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
						   two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0],
						   limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors4')(boxes4)
	anchors5 = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
						   two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
						   limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors5')(boxes5)
	anchors6 = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
						   two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2],
						   limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors6')(boxes6)
	anchors7 = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
						   two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3],
						   limit_boxes=limit_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors7')(boxes7)

	# Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
	# We want the classes isolated in the last axis to perform softmax on them
	classes4_reshaped = Reshape((-1, n_classes), name='classes4_reshape')(classes4)
	classes5_reshaped = Reshape((-1, n_classes), name='classes5_reshape')(classes5)
	classes6_reshaped = Reshape((-1, n_classes), name='classes6_reshape')(classes6)
	classes7_reshaped = Reshape((-1, n_classes), name='classes7_reshape')(classes7)
	# Reshape the box coordinate predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
	# We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
	boxes4_reshaped = Reshape((-1, 4), name='boxes4_reshape')(boxes4)
	boxes5_reshaped = Reshape((-1, 4), name='boxes5_reshape')(boxes5)
	boxes6_reshaped = Reshape((-1, 4), name='boxes6_reshape')(boxes6)
	boxes7_reshaped = Reshape((-1, 4), name='boxes7_reshape')(boxes7)
	# Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
	anchors4_reshaped = Reshape((-1, 8), name='anchors4_reshape')(anchors4)
	anchors5_reshaped = Reshape((-1, 8), name='anchors5_reshape')(anchors5)
	anchors6_reshaped = Reshape((-1, 8), name='anchors6_reshape')(anchors6)
	anchors7_reshaped = Reshape((-1, 8), name='anchors7_reshape')(anchors7)

	# Concatenate the predictions from the different layers and the assosciated anchor box tensors
	# Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
	# so we want to concatenate along axis 1
	# Output shape of `classes_merged`: (batch, n_boxes_total, n_classes)
	classes_concat = Concatenate(axis=1, name='classes_concat')([classes4_reshaped,
																 classes5_reshaped,
																 classes6_reshaped,
																 classes7_reshaped])

	# Output shape of `boxes_final`: (batch, n_boxes_total, 4)
	boxes_concat = Concatenate(axis=1, name='boxes_concat')([boxes4_reshaped,
															 boxes5_reshaped,
															 boxes6_reshaped,
															 boxes7_reshaped])

	# Output shape of `anchors_final`: (batch, n_boxes_total, 8)
	anchors_concat = Concatenate(axis=1, name='anchors_concat')([anchors4_reshaped,
																 anchors5_reshaped,
																 anchors6_reshaped,
																 anchors7_reshaped])

	# The box coordinate predictions will go into the loss function just the way they are,
	# but for the class predictions, we'll apply a softmax activation layer first
	classes_softmax = Activation('softmax', name='classes_softmax')(classes_concat)

	# Concatenate the class and box coordinate predictions and the anchors to one large predictions tensor
	# Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
	predictions = Concatenate(axis=2, name='predictions')([classes_softmax, boxes_concat, anchors_concat])

	model = Model(inputs=x, outputs=predictions)
	
	return model