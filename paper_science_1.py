import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np
import math


# The code is using the `tfds.load()` function from the TensorFlow Datasets library to load the MNIST
# dataset.
ds, ds_info = tfds.load(
    'mnist',
    as_supervised=True,  # returns `(img, label)` instead of dict(image=, ...)
    with_info=True,
)


# The lines `n_train = ds_info.splits['train'].num_examples` and `n_test =
# ds_info.splits['test'].num_examples` are retrieving the number of examples in the training and test
# splits of the MNIST dataset, respectively. These variables are used to keep track of the number of
# examples in each split for later use in the code.
n_train = ds_info.splits['train'].num_examples #60000 cases
n_test = ds_info.splits['test'].num_examples #10000 cases

# The line `dts_test, dts_train =  ds['test'], ds['train']` is assigning the test and train datasets
# from the loaded MNIST dataset to the variables `dts_test` and `dts_train`, respectively.
dts_test, dts_train =  ds['test'], ds['train']


# `names_class = ds_info.features['label'].names` is retrieving the names of the classes or labels in
# the MNIST dataset.
names_class = ds_info.features['label'].names


def normalize_img(image, label):
    """
    The function `normalize_img` takes an image and label as input, converts the image to float32 data
    type, and normalizes the pixel values by dividing them by 255.
    
    :param image: The image parameter is the input image that needs to be normalized. It is expected to
    be a tensor representing the image data
    :param label: The label parameter represents the label or class associated with the image. It could
    be a numerical value or a categorical value indicating the class or category to which the image
    belongs
    :return: the normalized image (beans) and the label.
    """
    beans = tf.cast(image, tf.float32)
    beans = beans / 255.
    return beans, label


# The lines `dts_test = dts_test.map(normalize_img)` and `dts_train = dts_train.map(normalize_img)`
# are applying the `normalize_img` function to each image in the test and train datasets,
# respectively.
dts_test = dts_test.map(normalize_img)
dts_train = dts_train.map(normalize_img)


# builder = tfds.builder('my_dataset')
# builder.info.splits['train'].num_examples  # 10_000
# builder.info.splits['train[:75%]'].num_examples  # 7_500 (also works with slices)
# builder.info.splits.keys()  # ['train', 'test']


# The line `dts_test, dts_train = dts_test.cache(), dts_train.cache()` is caching the test and train
# datasets. Caching a dataset means that the data will be stored in memory after it is loaded, which
# can improve the performance of training or evaluating a model on the dataset. By caching the
# datasets, subsequent operations on the datasets can be faster because the data is already loaded in
# memory.
dts_test, dts_train = dts_test.cache(), dts_train.cache()


# The line `dts_test, dts_train = dts_test.shuffle(10000), dts_train.shuffle(10000)` is shuffling the
for image, label in dts_train.take(1):
    break
image = image.numpy().reshape((28, 28))


# The code `plt.figure()` creates a new figure for plotting.
plt.figure()
plt.imshow(image, cmap=plt.cm.binary)
plt.colorbar()
plt.grid(False)
plt.show()


# The code is creating a figure with a size of 10x10 inches using `plt.figure(figsize=(10, 10))`.
plt.figure(figsize=(10, 10))
for i, (image, label) in enumerate(dts_train.take(25)):
    image = image.numpy().reshape((28, 28))
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image, cmap=plt.cm.binary)
    plt.xlabel(names_class[label])


# The code `beans_classification = tf.keras.Sequential([...])` is creating a sequential model using
# the Keras API in TensorFlow.
beans_classification = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(128, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax),
])


# The `beans_classification.compile()` function is used to configure the model for training. It
# specifies the optimizer, loss function, and metrics to be used during training.
beans_classification.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
)


# The code `dts_train = dts_train.repeat().shuffle(n_train).batch(32)` is preparing the training
# dataset for training the model.
dts_train = dts_train.repeat().shuffle(n_train).batch(32)
dts_test = dts_test.batch(32)


# The code `beans_logs = beans_classification.fit(dts_train, epochs=5,
# steps_per_epoch=math.ceil(n_train / 32))` is training the `beans_classification` model on the
# training dataset (`dts_train`) for 5 epochs.
beans_logs = beans_classification.fit(
    dts_train, 
    epochs=5, 
    steps_per_epoch=math.ceil(n_train / 32)
)

# Epoch 1/5
# 1875/1875 [==============================] - 15s 5ms/step - loss: 0.2271 - accuracy: 0.9327
# Epoch 2/5
# 1875/1875 [==============================] - 9s 5ms/step - loss: 0.0933 - accuracy: 0.9715
# Epoch 3/5
# 1875/1875 [==============================] - 9s 5ms/step - loss: 0.0673 - accuracy: 0.9791
# Epoch 4/5
# 1875/1875 [==============================] - 9s 5ms/step - loss: 0.0523 - accuracy: 0.9835
# Epoch 5/5
# 1875/1875 [==============================] - 7s 4ms/step - loss: 0.0428 - accuracy: 0.9859


# The code `plt.xlabel("#Y")` sets the label for the x-axis of the plot to "#Y", while
# `plt.ylabel("#X")` sets the label for the y-axis of the plot to "#X".
plt.xlabel("#Y")
plt.ylabel("#X")
plt.plot(beans_logs.history['loss'])


# The code is taking one batch of images and labels from the test dataset (`dts_test`) using the
# `take(1)` method. Then, it converts the image and label tensors to numpy arrays using the `numpy()`
# method. Finally, it passes the image array to the `beans_classification` model to make predictions
# on the test images. The predictions are stored in the `predictions` variable.
for image_test, label_test in dts_test.take(1):
    image_test = image_test.numpy()
    label_test = label_test.numpy()
    predictions = beans_classification(image_test)
    
    
def show_image(i, arr_predictions, label_values, images):
    """
    The function `show_image` displays an image along with its predicted label and the corresponding
    true label, highlighting the predicted label in blue if it matches the true label and in red
    otherwise.
    
    :param i: The index of the image to be shown
    :param arr_predictions: arr_predictions is a numpy array containing the predicted probabilities for
    each class. It has shape (num_classes,) where num_classes is the number of classes in the
    classification problem
    :param label_values: The `label_values` parameter is a list or array that contains the true labels
    or classes for each image in the `images` array. It should have the same length as the `images`
    array
    :param images: The `images` parameter is a list of images. Each image is represented as a numpy
    array
    """
    arr_predictions, label_value, img = arr_predictions[i], label_values[i], images[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    plt.imshow(img[...,0], cmap=plt.cm.binary)
    
    prediction_label = np.argmax(arr_predictions)
    
    if prediction_label == label_value:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(
        names_class[prediction_label],
        100*np.max(arr_predictions),
        names_class[label_value],
        color=color
    ))


def show_value_prediction(i, arr_predictions, label_value):
    """
    The function `show_value_prediction` is used to display the predicted value and the actual value of
    an image, along with a bar graph showing the probabilities of each class prediction.
    
    :param i: The index of the prediction and label value to display
    :param arr_predictions: arr_predictions is a list of predicted values for each class. It contains
    the predicted probabilities for each class in the range [0, 1]
    :param label_value: The true label value of the image
    """
    arr_predictions, label_value = arr_predictions[i], label_value[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    graphics = plt.bar(range(10), arr_predictions, color='#101010')
    plt.ylim([0, 1])
    label_prediction = np.argmax(arr_predictions)
    
    graphics[label_value].set_color('red')
    graphics[label_prediction].set_color('blue')
    
    rows = 5
    columns = 5
    nums_images = rows * columns
    plt.figure(figsize=(2*columns, 2*rows))
    for i in range(nums_images):
        plt.subplot(rows, 2*columns, 2*i+1)
        show_image(i, predictions, label_test, image_test)
        plt.subplot(rows, 2*columns, 2*i+1)
        show_value_prediction(i, predictions, label_value, image_test)