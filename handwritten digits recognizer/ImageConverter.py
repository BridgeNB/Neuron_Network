__author__ = 'zhengyangqiao'
"""
The image converter will convert png or jpg to grey scale matrix
for handwritten digits recognition.
"""
import numpy as np
from PIL import Image

# set element in matrix to two digits
# float_formatter = lambda x: "%.1f" % x
# np.set_printoptions(formatter={'float_kind':float_formatter})
x = Image.open('handwritten_number.png', 'r')
x = x.convert('L') # make it greyscale
y = np.asarray(x.getdata(), dtype=np.int16).reshape((x.size[1], x.size[0]))

length = x.size[1] / 28
width = x.size[0] / 28
print(length, width)

cleanedMatrix = [[]]
cleanedMatrix = [[0 for i in range(0, 28)] for j in range(0, 28)]
for i in range(0 , 28):
    for j in range(0, 28):
        ii = i * length
        jj = j * width
        cleanedMatrix[i][j] = np.sum(y[ii : ii + length, jj : jj + width]) / (length * width)


# final = np.round(y, 2)
# print(y)
# print(cleanedMatrix)

print(cleanedMatrix)