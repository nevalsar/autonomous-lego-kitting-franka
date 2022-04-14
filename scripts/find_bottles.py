import cv2
import numpy as np

def define_borders(img):
    border = []
    
    def drawBorder(action, x, y, flags, param):
        if action == cv2.EVENT_LBUTTONDOWN:
           param.append([x,y])
        elif action == cv2.EVENT_LBUTTONUP:
            param.append([x,y])
        
    cv2.namedWindow('image')
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', drawBorder, border)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return border


def find_drink_old(rgb_image, lower, upper, border):
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
    cv2.rectangle(rgb_image, (x-30,y-60), (x+w+30,y+h), (0,0,255), 2)

    # Create a bounding box which only contains the upper half of the bottle
    # HSV thresholding the black nozzle
    nozzle_mask = np.zeros(rgb_image.shape[:2], np.uint8)
    nozzle_mask[y-60:y+h, x-30:x+w+30] = 255
    nozzle_mask = cv2.bitwise_and(rgb_image, rgb_image, mask=nozzle_mask)
    nozzle_hsv = cv2.cvtColor(nozzle_mask, cv2.COLOR_BGR2HSV)
    nozzle_mask = cv2.inRange(nozzle_hsv, np.array([0,1,0]), np.array([179,255,50]))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
    nozzle_mask = cv2.morphologyEx(nozzle_mask, cv2.MORPH_CLOSE, kernel, iterations=5)
    nozzle_mask = cv2.morphologyEx(nozzle_mask, cv2.MORPH_OPEN, kernel, iterations=5)
    nozzle_cnts, _ = cv2.findContours(nozzle_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    nozzle_c = max(nozzle_cnts, key=cv2.contourArea)
    (nozzle_x, nozzle_y), radius = cv2.minEnclosingCircle(nozzle_c)
    center = [int(nozzle_x), int(nozzle_y)]
    radius = int(radius)
    cv2.circle(rgb_image, center, radius, (255,0,0), 2)

    cv2.imshow('img', cv2.resize(rgb_image, (1024,768)))
    cv2.waitKey(0)


def find_drink(rgb_image, lower, upper, border):
    hsv = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    cv2.imshow('mask', cv2.resize(mask, (1024,768)))

    cnts, hie = cv2.findContours(mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    maxc_i = np.argmax([cv2.contourArea(c) for c in cnts])
    childc_i = hie.squeeze()[maxc_i][2]

    if childc_i == -1:
        c = cnts[maxc_i]

        bottle_mask = np.zeros(rgb_image.shape[:2], np.uint8)
        bottle_mask = cv2.drawContours(bottle_mask, cnts, maxc_i, (255,255,255), cv2.FILLED)
        white_bg = np.ones_like(rgb_image) * 255
        bottle_img = cv2.bitwise_and(white_bg, rgb_image, mask=bottle_mask) + cv2.bitwise_and(white_bg, white_bg, mask=~bottle_mask)
        cv2.imshow('bottle', cv2.resize(bottle_img, (1024,768)))

        bottle_hsv = cv2.cvtColor(bottle_img, cv2.COLOR_BGR2HSV)
        hole_mask = cv2.inRange(bottle_hsv, np.array([0,0,0]), np.array([179,100,130]))
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        hole_mask = cv2.morphologyEx(hole_mask, cv2.MORPH_OPEN, kernel, iterations=2)
        cv2.imshow('hole', cv2.resize(hole_mask, (1024,768)))

        hole_cnts, _ = cv2.findContours(hole_mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        if len(hole_cnts) > 0:
            c = max(hole_cnts, key=cv2.contourArea)
        
    else:
        c = cnts[childc_i]

    contour = np.array(c).squeeze()
    center_x, center_y = np.mean(contour, axis=0).astype(int)
    
    cv2.circle(rgb_image, (center_x, center_y), radius=1, color=(0,0,255), thickness=10)
    cv2.drawContours(rgb_image, cnts, maxc_i, (0,255,0), 2)

    cv2.imshow('img', cv2.resize(rgb_image, (1024,768)))
    cv2.waitKey(0)
    cv2.imwrite('blue.png', rgb_image)


if __name__ == '__main__':
    rgb_image = cv2.imread('rgb.png')

    # border = define_borders(rgb_image)
    border = [[562, 184], [1448, 876]]

    mask = np.zeros(rgb_image.shape[:2], np.uint8)
    mask[border[0][1]:border[1][1], border[0][0]:border[1][0]] = 255
    rgb_image = cv2.bitwise_and(rgb_image, rgb_image, mask=mask)

    rLower = np.array([160,59,20])
    rUpper = np.array([179,255,255])

    gLower = np.array([70,69,20])
    gUpper = np.array([83,255,255])

    bLower = np.array([102,7,28])
    bUpper = np.array([111,196,255])

    find_drink_old(rgb_image, gLower, gUpper, border)


# # Load image, grayscale, Otsu's threshold, and extract ROI
# image = cv2.imread('1.jpg')
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
# x,y,w,h = cv2.boundingRect(thresh)
# ROI = image[y:y+h, x:x+w]

# # Color segmentation on ROI
# hsv = cv2.cvtColor(ROI, cv2.COLOR_BGR2HSV)
# lower = np.array([0, 0, 152])
# upper = np.array([179, 255, 255])
# mask = cv2.inRange(hsv, lower, upper)

# # Crop left and right half of mask
# x, y, w, h = 0, 0, ROI.shape[1]//2, ROI.shape[0]
# left = mask[y:y+h, x:x+w]
# right = mask[y:y+h, x+w:x+w+w]

# # Count pixels
# left_pixels = cv2.countNonZero(left)
# right_pixels = cv2.countNonZero(right)

# print('Left pixels:', left_pixels)
# print('Right pixels:', right_pixels)

# cv2.imshow('mask', mask)
# cv2.imshow('thresh', thresh)
# cv2.imshow('ROI', ROI)
# cv2.imshow('left', left)
# cv2.imshow('right', right)
# cv2.waitKey()