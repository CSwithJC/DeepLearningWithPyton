"""Simple example of a Neural Net classifier for the MNIST dataset using Keras
"""
from keras.datasets import mnist
from keras import layers, models

# Load the MNIST dataset as test and training data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Dimensions of training data
print(train_images.shape)
print(len(train_labels))
print(train_labels)

# Dimensions of test data
print(test_images.shape)
print(len(test_labels))
print(test_labels)

# A Sequential model is just a linear stack of layers
network = models.Sequential()
# Dense = Densely connected (fully connected) neural layer
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
# Softmax = Returns the probability scores of belonging to each label
network.add(layers.Dense(10, activation='softmax'))  # 10 because there are 10 labels here

# COMPILATION STEP:
#
# Loss Function = How the network measures the performance on the training data, and how it will be able to
#                 steer itself in the right direction
# Optimizer = Mechanism through which the network will update itself based on the data it sees and its loss function
#
# Metrics to Monitor = What metric you want to optimize
#
network.compile(optimizer='rmsprop',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

