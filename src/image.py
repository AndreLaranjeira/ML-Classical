# Library to hold functions related to image processing.

# Package imports
import cv2
import numpy as np

# Control variables:
GREYSCALE_THRESHOLD = 105
IMG_L = 28
IMG_W = 28

# Function definitions:
def center_of_gravity(image_matrix, length = IMG_L, width = IMG_W):

    num_dots = 0
    x_sum = 0
    y_sum = 0

    for i in range(length):
        for j in range(width):
            if(image_matrix[i][j] >= GREYSCALE_THRESHOLD):
                num_dots += 1
                x_sum += j
                y_sum += i

    return (x_sum//num_dots, y_sum//num_dots)

def image_matrix(image_array, width = IMG_W):

    lines = []

    for i in range(0, len(image_array), width):
        lines.append(image_array[i:i+width])

    return lines

# Codado por Victor André Gris Costa vvvvvvv
def get_contours(image):
    img = np.reshape(np.array(image, dtype=np.uint8), (28,28))
    #cv2.imshow('a', cv2.resize(img ,(400,400),interpolation=cv2.INTER_NEAREST))
    _, thresh = cv2.threshold(img, GREYSCALE_THRESHOLD, 255, 0)
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=lambda x:len(x), reverse=True)
    return contours, img

def get_euler_number(contours):
    return 2-len(contours)

def get_shape_area(contours):
    shape_area = cv2.contourArea(contours[0])
    # for cont in contours[1:]:
    #     shape_area -= cv2.contourArea(cont)
    return shape_area

erros = 0
def get_rectangularity(contours):
    global erros
    rect = cv2.minAreaRect(contours[0])
    _, dim, _ = rect
    rect_area = dim[0]*dim[1]
    shape_area = get_shape_area(contours)
    return shape_area/rect_area

def get_circularity(contours):
    perimeter = cv2.arcLength(contours[0],True)
    shape_area = get_shape_area(contours)
    return 4*np.pi*shape_area/(perimeter*perimeter)

def get_convexity(contours):
    perimeter = cv2.arcLength(contours[0],True)
    convex_perimeter = cv2.arcLength(cv2.convexHull(contours[0]), True)
    return convex_perimeter/perimeter

def get_elongation(contours):
    _, _, w, h = cv2.boundingRect(contours[0])
    return 1 - (w/h)

def get_solidity(contours):
    area = get_shape_area(contours)
    convex_hull_area = cv2.contourArea(cv2.convexHull(contours[0]))
    return area/convex_hull_area

def preprocess(image):
    contours, img = get_contours(image)
    euler = get_euler_number(contours)
    rectangularity = get_rectangularity(contours)
    circularity = get_circularity(contours)
    convexity = get_convexity(contours)
    elongation = get_elongation(contours)
    solidity = get_solidity(contours)
    #print(euler,rectangularity,circularity)
    return [euler, rectangularity, circularity, solidity, convexity], img, contours

def preprocess_many(images):
    features = []
    for image in images:
        data, img, contours = preprocess(image)
        features.append(data)
    return features
# Codado por Victor André Gris Costa ^^^^^^^
