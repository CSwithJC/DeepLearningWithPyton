"""Examples of tensor manipulation in Numpy.
"""
from keras.datasets import mnist

# Load the MNIST dataset as test and training data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Select digits 10 to 100 (100 isn't included)
my_slice = train_images[10:100]
print(my_slice.shape)

# Equivalent slicing as before
my_slice = train_images[10:100, :, :]
print(my_slice.shape)

# Also equivalent slicing as before
my_slice = train_images[10:100, 0:28, 0:28]
print(my_slice.shape)

# Select 14x14 pixels in the bottom-right corner of all images
my_slice = train_images[:, 14:, 14:]
print(my_slice.shape)

# Crop the images to patches of 14x14 pixels centered in the middle:
my_slice = train_images[:, 7:-7, 7:-7]
print(my_slice.shape)

