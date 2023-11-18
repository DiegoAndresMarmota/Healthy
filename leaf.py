import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
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


plt.xlabel("#Y")
plt.ylabel("#X")
plt.plot(beans_logs.history['loss'])

