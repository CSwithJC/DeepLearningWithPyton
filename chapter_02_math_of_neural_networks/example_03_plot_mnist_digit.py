"""Plot an MNIST digit using matplotlib
"""
import matplotlib.pyplot as plt
from keras.datasets import mnist

# Load the MNIST dataset as test and training data
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print('Dimensions of training image:')
print(train_images.ndim)

print('\nShape of training image:')
print(train_images.shape)

print('\nData type of training image:')
print(train_images.dtype)

digit = train_images[4]
plt.imshow(digit, cmap=plt.cm.binary)
plt.show()
