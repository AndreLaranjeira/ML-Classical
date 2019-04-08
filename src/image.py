# Library to hold functions related to image processing.

# Package imports
import cv2
import numpy as np

# -----------------------------------
# Código feito por André Laranjeira.
# -----------------------------------

# Control variables:
GREYSCALE_THRESHOLD = 70
IMG_L = 28
IMG_W = 28

# Function definitions:
def ALI_measuments(image_matrix, length = IMG_L, width = IMG_W):

    a = 0
    b = 0
    c = 0

    for i in range(length):
        for j in range(width):
            if(image_matrix[i][j] >= GREYSCALE_THRESHOLD):
                a += j**2
                b += j*i
                c += i**2

    return a, 2*b, c

def image_matrix(image_array, width = IMG_W):

    lines = []

    for i in range(0, len(image_array), width):
        lines.append(image_array[i:i+width])

    return lines

# -----------------------------------
# Código feito por Victor Gris Costa.
# -----------------------------------

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

def get_rectangularity(contours):
    rect = cv2.minAreaRect(contours[0])
    _, dim, _ = rect
    rect_area = dim[0]*dim[1]
    shape_area = get_shape_area(contours)
    return shape_area/rect_area

def get_circularity(contours):
    perimeter = cv2.arcLength(contours[0],True)
    shape_area = get_shape_area(contours)
    return 4*np.pi*shape_area/(perimeter*perimeter)

# -----------------------------------
# Código feito por André Laranjeira.
# -----------------------------------

def get_convexity(contours):
    perimeter = cv2.arcLength(contours[0],True)
    convex_perimeter = cv2.arcLength(cv2.convexHull(contours[0]), True)
    return convex_perimeter/perimeter

def get_elongation(contours):
    rect = cv2.minAreaRect(contours[0])
    w = rect[1][0]
    h = rect[1][1]

    if w < h:
        aux = (w/h)

    else:
        aux = (h/w)

    return 1 - aux

def get_solidity(contours):
    area = get_shape_area(contours)
    convex_hull_area = cv2.contourArea(cv2.convexHull(contours[0]))
    return area/convex_hull_area

def get_ALI_angle(img):
    a, b, c = ALI_measuments(image_matrix(img))
    alpha = 0.5*np.arctan2(b, (a-c))

    if (2*(a-c)*np.cos(2*alpha) + 2*b*np.sin(2*alpha)) < 0:
        alpha += np.pi/2

    return alpha

# -----------------------------------
# Código feito por Victor Gris Costa.
# -----------------------------------

def preprocess(image):
    contours, img = get_contours(image)
    ALI_angle = get_ALI_angle(image)
    euler = (get_euler_number(contours) + 1)/2
    rectangularity = get_rectangularity(contours)
    convexity = get_convexity(contours)
    elongation = get_elongation(contours)
    solidity = get_solidity(contours)
    return [euler, rectangularity, solidity, elongation, convexity, ALI_angle]

def preprocess_many(images):
    features = []
    for image in images:
        data = preprocess(image)
        features.append(data)
    return np.array(features)
