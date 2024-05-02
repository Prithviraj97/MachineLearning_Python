import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, TimeDistributed, Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint


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


def fit_and_evaluate(t_x, val_x, t_y, val_y, t_box, val_box, EPOCHS, BATCH_SIZE, class_w):
    # Assuming `get_faster_rcnn_model()` is a function that builds and compiles the Faster R-CNN model
    model_rpn, model_classifier = get_faster_rcnn_model()
    
    # Callbacks for training
    erlstp = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    model_checkpoint = ModelCheckpoint('faster_rcnn_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

    # Training the RPN
    print("Training RPN...")
    model_rpn.fit(t_x, [t_y, t_box], batch_size=BATCH_SIZE, epochs=EPOCHS,
                  validation_data=(val_x, [val_y, val_box]), 
                  class_weight=class_w, callbacks=[erlstp, reduce_lr, model_checkpoint], verbose=1)

    # Training the Classifier
    print("Training Classifier...")
    model_classifier.fit(t_x, [t_y, t_box], batch_size=BATCH_SIZE, epochs=EPOCHS,
                         validation_data=(val_x, [val_y, val_box]), 
                         class_weight=class_w, callbacks=[erlstp, reduce_lr, model_checkpoint], verbose=1)

    # Evaluate the model
    print("\nValidation Score: ", model_classifier.evaluate(val_x, [val_y, val_box]))
    return model_rpn.history, model_classifier.history

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

n_folds = 5
epochs = 100
batch_size = 32
weight = {'class_weight': 1}  # Define as needed based on your class imbalances

model_history = []

for i in range(n_folds):
    print("Training on Fold: ", i + 1)

    # Assuming your data and labels are split into features (X), labels for classification (Y), and bounding boxes (B)
    X_train, X_val, Y_train, Y_val, B_train, B_val = train_test_split(
        X, Y, B, test_size=0.2, random_state=i)  # Ensure you have a proper split for X, Y, and B

    X_train, Y_train, B_train = shuffle(X_train, Y_train, B_train, random_state=i)

    history_rpn, history_classifier = fit_and_evaluate(X_train, X_val, Y_train, Y_val, B_train, B_val, epochs, batch_size, weight)
    model_history.append((history_rpn, history_classifier))
    print("=======" * 12, end="\n\n")

