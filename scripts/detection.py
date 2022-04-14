import cv2

import numpy as np

import imutils

###New imports
import argparse
from cv_bridge import CvBridge
from autolab_core import RigidTransform, Point
from perception import CameraIntrinsics
from utils import *
from sensor_msgs.msg import Image
from frankapy import FrankaArm

# cap = cv2.VideoCapture(0)

# cap.set(3, 640)
# cap.set(4, 480)

AZURE_KINECT_INTRINSICS = 'calib/azure_kinect.intr'
AZURE_KINECT_EXTRINSICS = 'calib/azure_kinect_overhead/azure_kinect_overhead_to_world.tf'

# cap.set(3, 100)
# cap.set(4, 100)
# count=1
# while count>0:
parser = argparse.ArgumentParser()
parser.add_argument('--intrinsics_file_path', type=str, default=AZURE_KINECT_INTRINSICS)
parser.add_argument('--extrinsics_file_path', type=str, default=AZURE_KINECT_EXTRINSICS) 
args = parser.parse_args()


cv_bridge = CvBridge()
azure_kinect_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
azure_kinect_to_world_transform = RigidTransform.load(args.extrinsics_file_path)    
print("hi")
# azure_kinect_rgb_image = get_azure_kinect_rgb_image(cv_bridge)
# print("hi")
# azure_kinect_depth_image = get_azure_kinect_depth_image(cv_bridge)
# print("hi")

while True:
    print("yo")
    # _, frame = cap.read()
    azure_kinect_rgb_image = None
    print("yo")
    rgb_image_msg = rospy.wait_for_message('/rgb/image_raw', Image)
    print("yo")
    try:
        rgb_cv_image = cv_bridge.imgmsg_to_cv2(rgb_image_msg)
    except CvBridgeError as e:
        print(e)
    print("yo")
    azure_kinect_rgb_image = rgb_cv_image
    # azure_kinect_rgb_image = get_azure_kinect_rgb_image(cv_bridge)
    azure_kinect_depth_image = get_azure_kinect_depth_image(cv_bridge)
    print("yo")
    cv2.imshow(azure_kinect_rgb_image)
    frame = azure_kinect_rgb_image
    frame = frame[0:450,275:850]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Red color ranges
    lower_red = np.array([129, 196, 61])
    upper_red = np.array([179, 255, 255])

    # Yellow color ranges
    lower_yellow = np.array([11, 172, 21])
    upper_yellow = np.array([70, 255, 255])

    # green color ranges
    lower_green = np.array([58, 52, 24])
    upper_green = np.array([95, 255, 148])

    # blue color ranges
    lower_blue = np.array([104, 197, 51])
    upper_blue = np.array([155, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red, upper_red)
    mask2 = cv2.inRange(hsv, lower_yellow, upper_yellow)
    mask3 = cv2.inRange(hsv, lower_green, upper_green)
    mask4 = cv2.inRange(hsv, lower_blue, upper_blue)

    # Finding Red contours
    cnts1 = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts1 = imutils.grab_contours(cnts1)

    # Finding yellow contours
    cnts2 = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts2 = imutils.grab_contours(cnts2)

    # Finding green contours
    cnts3 = cv2.findContours(mask3, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts3 = imutils.grab_contours(cnts3)

    # Finding blue contours
    cnts4 = cv2.findContours(mask4, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts4 = imutils.grab_contours(cnts4)

    all_blocks = []
    red_rect = []
    for c in cnts1:
        area1 = cv2.contourArea(c)
        if area1 > 50:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1)
            rect = list(cv2.minAreaRect(c))

            red_dict = {'color': 'red', 'center': (cx, cy), 'angle': rect[2]}
            all_blocks.append(red_dict)
            # cv2.putText(frame, "red", (cx-20, cy-20),cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

    for c in cnts2:
        area2 = cv2.contourArea(c)
        if area2 > 50:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1)
            rect = list(cv2.minAreaRect(c))

            yellow_dict = {'color': 'yellow',
                           'center': (cx, cy), 'angle': rect[2]}
            all_blocks.append(yellow_dict)
            # cv2.putText(frame, "yellow", (cx-20, cy-20),cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

    for c in cnts3:
        area3 = cv2.contourArea(c)
        if area3 > 50:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1)
            rect = list(cv2.minAreaRect(c))

            green_dict = {'color': 'green',
                          'center': (cx, cy), 'angle': rect[2]}
            all_blocks.append(green_dict)
            # cv2.putText(frame, "green", (cx-20, cy-20),cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

    for c in cnts4:
        area4 = cv2.contourArea(c)
        if area4 > 50:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1)
            rect = list(cv2.minAreaRect(c))

            blue_dict = {'color': 'blue', 'center': (cx, cy), 'angle': rect[2]}
            all_blocks.append(blue_dict)
            # cv2.putText(frame, "blue", (cx-20, cy-20),cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

    # count-=1
    print(all_blocks)
    print(len(all_blocks))

    cx, cy = all_blocks[0]
    get_obj_center_abhi()

    get_obj_center_abhi = get_obj_center_abhi(cx, cy, azure_kinect_depth_image, azure_kinect_intrinsics, azure_kinect_to_world_transform)
    print(get_obj_center_abhi)
    cv2.imshow("result", frame)

    k = cv2.waitKey(5)

    if k == 27:

        break


# cap.release()

cv2.destroyAllWindows()
