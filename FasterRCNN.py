import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, TimeDistributed, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16

def rpn_layer(base_layers, num_anchors):
    """Create a convolution layer to predict object scores and bounding boxes."""
    x = Conv2D(512, (3, 3), padding='same', activation='relu', kernel_initializer='normal', name='rpn_conv1')(base_layers)
    x_class = Conv2D(num_anchors, (1, 1), activation='sigmoid', kernel_initializer='uniform', name='rpn_out_class')(x)
    x_regr = Conv2D(num_anchors * 4, (1, 1), activation='linear', kernel_initializer='zero', name='rpn_out_regress')(x)
    return [x_class, x_regr, base_layers]

def classifier_layer(base_layers, input_rois, num_rois, num_classes):
    """Define the classifier layer to classify object types and bounding boxes"""
    pooling_regions = 7
    x = TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(7, 7)), input_shape=(num_rois,7,7,512))(input_rois)

    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(4096, activation='relu'))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    x = TimeDistributed(Dense(4096, activation='relu'))(x)
    x = TimeDistributed(Dropout(0.5))(x)
    out_class = TimeDistributed(Dense(num_classes, activation='softmax', kernel_initializer='zero'), name='dense_class')(x)
    out_regr = TimeDistributed(Dense(num_classes * 4, activation='linear', kernel_initializer='zero'), name='dense_regress')(x)
    return [out_class, out_regr]

def get_faster_rcnn_model():
    num_anchors = 9
    num_rois = 32
    num_classes = 4  # Adjust based on your dataset (including background)
    pooling_regions = 7
    # Base network (VGG16)
    base_model = VGG16(weights='imagenet', include_top=False)
    base_layers = base_model.get_layer('block5_conv3').output

    # Define inputs
    input_shape = base_model.input_shape[1:]
    img_input = Input(shape=input_shape)
    roi_input = Input(shape=(num_rois, pooling_regions, pooling_regions, 512))

    # Create RPN
    rpn_classes, rpn_regressors, shared_layers = rpn_layer(base_layers, num_anchors)

    # Create classifier
    classifier_classes, classifier_regressors = classifier_layer(shared_layers, roi_input, num_rois, num_classes)

    # Creating models
    model_rpn = Model(inputs=img_input, outputs=[rpn_classes, rpn_regressors])
    model_classifier = Model(inputs=[img_input, roi_input], outputs=[classifier_classes, classifier_regressors])

    # Compile models
    model_rpn.compile(optimizer='adam', loss=['binary_crossentropy', 'mean_squared_error'], metrics=['accuracy'])
    model_classifier.compile(optimizer='adam', loss=['categorical_crossentropy', 'mean_squared_error'], metrics=['accuracy'])

    return model_rpn, model_classifier
