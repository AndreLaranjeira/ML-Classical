# Library to hold functions related to image processing.

# Control variables:
IMG_W = 28

# Auxiliary functions:
def image_matrix(image_array, width = IMG_W):

    lines = []

    for i in range(0, len(image_array), width):
        lines.append(image_array[i:i+width])

    return lines
