# Library to hold functions related to image processing.

# Control variables:
IMG_L = 28
IMG_W = 28

# Function definitions:
def center_of_gravity(image_matrix, length = IMG_L, width = IMG_W):

    num_dots = 0
    x_sum = 0
    y_sum = 0

    for i in range(length):
        for j in range(width):
            if(image_matrix[i][j] != 0):
                num_dots += 1
                x_sum += j
                y_sum += i

    return (x_sum//num_dots, y_sum//num_dots)


def image_matrix(image_array, width = IMG_W):

    lines = []

    for i in range(0, len(image_array), width):
        lines.append(image_array[i:i+width])

    return lines
