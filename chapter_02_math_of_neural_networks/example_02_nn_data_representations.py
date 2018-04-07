"""Different types of Data Representations for Neural Networks

NOTE: Tensors are basically multidimensional nd-arrays

"""
import numpy as np

# Scalars (0D Tensors)
# Scalar = a tensor that contains only one number
print('Scalar (0D Tensor)')
x = np.array(12)
print(x)
print(x.ndim)  # The number of axis of a tensor is also called its rank

# Vectors (1D Tensors)
# Vector = An Array of Numbers
print('\nVector (1D Tensor)')
x = np.array([12, 3, 6, 14])
print(x)
print(x.ndim)

# Matrices (2D Tensors)
# Matrix = an array of vectors
print('\nMatrix (2D Tensor)')
x = np.array([[5, 78, 2, 34, 0],
              [5, 79, 3, 35, 1],
              [6, 80, 4, 36, 2]])
print(x)
print(x.ndim)

# 3D Tensors
# Can be interpreted as a cube of numbers
print('\n3D Tensor')
x = np.array([[[5, 78, 2, 34, 0],
               [5, 79, 3, 35, 1],
               [6, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [5, 79, 3, 35, 1],
               [6, 80, 4, 36, 2]],
              [[5, 78, 2, 34, 0],
               [5, 79, 3, 35, 1],
               [6, 80, 4, 36, 2]]])
print(x)
print(x.ndim)
