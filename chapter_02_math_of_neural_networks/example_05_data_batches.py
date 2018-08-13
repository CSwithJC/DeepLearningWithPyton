"""Examples of getting the data in batches.
"""
from keras.datasets import mnist

# Load the MNIST dataset as test and training data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# First batch of 128 digits
batch = train_images[:128]

# Second batch of 128 digits
batch = train_images[128:256]

# Nth batch of 128 digits
n = 4
batch = train_images[128 * n:128 * (n + 1)]

# When considering such a batch tensor, the first axis (axis 0) is called the batch axis or batch dimension.
