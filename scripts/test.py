import numpy as np
import cv2


def find_drink(rgb_image, lower, upper):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=1)
    mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=1)
    mask = cv2.erode(mask, np.ones((5,5), np.uint8), iterations=1)
    mask = cv2.dilate(mask, np.ones((5,5), np.uint8), iterations=1)
    cnts, hie = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    maxc_i = np.argmax([cv2.contourArea(c) for c in cnts])
    c = cnts[maxc_i]
    x,y,w,h = cv2.boundingRect(c)
    x -= 5
    y -= 60
    w += 50
    h += 50
    cv2.rectangle(rgb_image, (x,y), (x+w,y+h), (0,0,255), 2)

    # Create a bounding box which only contains the upper half of the bottle
    # HSV thresholding the black nozzle
    nozzle_mask = np.zeros(rgb_image.shape[:2], np.uint8)
    nozzle_mask[y:y+h, x:x+w] = 255
    nozzle_mask = cv2.bitwise_and(rgb_image, rgb_image, mask=nozzle_mask)
    nozzle_hsv = cv2.cvtColor(nozzle_mask, cv2.COLOR_BGR2HSV)
    nozzle_mask = cv2.inRange(nozzle_hsv, np.array([0,1,0]), np.array([179,255,50]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    nozzle_mask = cv2.morphologyEx(nozzle_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    nozzle_mask = cv2.morphologyEx(nozzle_mask, cv2.MORPH_OPEN, kernel, iterations=5)
    cv2.imshow('nozzle_mask', cv2.resize(nozzle_mask, (1024,768)))
    
    nozzle_cnts, _ = cv2.findContours(nozzle_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    nozzle_c = max(nozzle_cnts, key=cv2.contourArea)
    (nozzle_x, nozzle_y), radius = cv2.minEnclosingCircle(nozzle_c)
    center = [int(nozzle_x), int(nozzle_y)]
    radius = int(radius)
    cv2.circle(rgb_image, center, radius, (255,0,0), 2)

    cv2.imshow('img', cv2.resize(rgb_image, (1024,768)))
    cv2.waitKey(0)

    return center


rLower = np.array([160,59,20])
rUpper = np.array([179,255,255])

gLower = np.array([70,69,20])
gUpper = np.array([83,255,255])

bLower = np.array([102,7,28])
bUpper = np.array([111,196,255])

rgb_image = cv2.imread('rgb.png')
border = [[389, 171], [1635, 875]]

mask = np.zeros(rgb_image.shape[:2], np.uint8)
mask[border[0][1]:border[1][1], border[0][0]:border[1][0]] = 255
rgb_image = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)

find_drink(rgb_image, bLower, bUpper)