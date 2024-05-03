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

######################################################################################################
import tensorflow as tf

# Define paths to the COCO dataset files. Any dataset can be used as along as it follows the requirement of Faster RCNN model. Example: Pascal VOC.
train_image_dir = '/content/train2017'  
val_image_dir = '/content/val2017'      
annotations_file = '/content/annotations/instances_train2017.json'  

# Load and preprocess images using TensorFlow Dataset API
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_image_dir,
    labels='inferred',
    label_mode='int',
    image_size=(img_height, img_width), #use image.shape to find the height and width
    batch_size=batch_size
)

val_dataset = tf.keras.utils.image_dataset_from_directory(
    val_image_dir,
    labels='inferred',
    label_mode='int',
    image_size=(img_height, img_width),
    batch_size=batch_size
)


def fit_and_evaluate(train_dataset, val_dataset, EPOCHS, BATCH_SIZE, class_w):
    model_rpn, model_classifier = get_faster_rcnn_model() #this builds and compiles the Faster R-CNN model.
    
    # Callbacks for training
    erlstp = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)
    model_checkpoint = ModelCheckpoint('faster_rcnn_model.h5', monitor='val_loss', save_best_only=True, verbose=1)

    # Training the RPN
    print("Training RPN...")
    model_rpn.fit(train_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS,
                  validation_data=val_dataset, 
                  class_weight=class_w, callbacks=[erlstp, reduce_lr, model_checkpoint], verbose=1)

    # Training the Classifier
    print("Training Classifier...")
    model_classifier.fit(train_dataset, batch_size=BATCH_SIZE, epochs=EPOCHS,
                         validation_data=val_dataset, 
                         class_weight=class_w, callbacks=[erlstp, reduce_lr, model_checkpoint], verbose=1)

    # Evaluate the model
    print("\nValidation Score: ", model_classifier.evaluate(val_dataset))
    return model_rpn.history, model_classifier.history 


import tensorflow as tf

def roi_pooling(inputs, rois, pool_size):
  """
  Implements ROI Pooling operation for Faster R-CNN.

  Args:
      inputs: Feature map tensor (e.g., output from a convolutional layer).
      rois: Region of Interest proposals (bounding boxes) with shape [num_rois, 4].
              Each row represents [y_min, x_min, y_max, x_max] for a bounding box.
      pool_size: Size of the pooling window (e.g., 7 for 7x7 pooling).

  Returns:
      Pooled feature vectors: Tensor with shape [num_rois, pool_size, pool_size, channels].
  """

  # Number of ROIs
  num_rois = tf.shape(rois)[0]

  # Extract coordinates from ROIs (y_min, x_min, y_max, x_max)
  y_min, x_min, y_max, x_max = tf.split(rois, axis=1, num_split=4)

  # Normalize coordinates to lie between 0 and 1
  normalized_y_min = (y_min - tf.reduce_min(y_min)) / (tf.reduce_max(y_min) - tf.reduce_min(y_min))
  normalized_x_min = (x_min - tf.reduce_min(x_min)) / (tf.reduce_max(x_min) - tf.reduce_min(x_min))
  normalized_y_max = (y_max - tf.reduce_min(y_max)) / (tf.reduce_max(y_max) - tf.reduce_min(y_max))
  normalized_x_max = (x_max - tf.reduce_min(x_max)) / (tf.reduce_max(x_max) - tf.reduce_min(x_max))

  # Calculate ROI widths and heights based on normalized coordinates
  roi_width = normalized_x_max - normalized_x_min
  roi_height = normalized_y_max - normalized_y_min

  # Clip potential out-of-bound coordinates (ensure they stay within the feature map)
  roi_width = tf.clip_by_value(roi_width, clip_value_min=0.0, clip_value_max=1.0)
  roi_height = tf.clip_by_value(roi_height, clip_value_min=0.0, clip_value_max=1.0)

  # Calculate grid size within the ROI for pooling
  grid_x = tf.cast(tf.floor(pool_size / roi_width), tf.float32)
  grid_y = tf.cast(tf.floor(pool_size / roi_height), tf.float32)

  # Define a meshgrid of coordinates for each ROI (similar to feature map sampling)
  x_grid, y_grid = tf.meshgrid(tf.range(pool_size), tf.range(pool_size))
  x_grid = tf.expand_dims(x_grid, axis=0)
  y_grid = tf.expand_dims(y_grid, axis=0)
  x_grid = tf.tile(x_grid, [num_rois, 1, 1])
  y_grid = tf.tile(y_grid, [num_rois, 1, 1])

  # Normalize grid coordinates to lie within the ROI (between 0 and 1)
  normalized_x_grid = x_grid * roi_width + normalized_x_min
  normalized_y_grid = y_grid * roi_height + normalized_y_min

  # Sample features from the input feature map using normalized grid coordinates
  # (similar to bilinear interpolation for non-integer coordinates)
  pooled_features = tf.gather_nd(inputs, tf.cast(tf.stack([y_grid, x_grid], axis=-1), tf.int32))

  # Max pooling across the grid within each ROI
  pooled_features = tf.reduce_max(pooled_features, axis=[1, 2])

  return pooled_features



