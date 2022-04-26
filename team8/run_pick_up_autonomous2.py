from frankapy import FrankaArm
import numpy as np
import argparse
import cv2
from cv_bridge import CvBridge
from autolab_core import RigidTransform, Point
from perception import CameraIntrinsics
from utils import *

import imutils


cap = cv2.VideoCapture(0)

cap.set(3, 640)
cap.set(4, 480)

# cap.set(3, 100)
# cap.set(4, 100)
count=1


if __name__ == '__main__':

    print('Starting robot')
    fa = FrankaArm()    

    print('Opening Grippers')
    #Open Gripper
    fa.open_gripper()
    print('Reset Pose')
    #Reset Pose
    fa.reset_pose() 
    print('Reset joints')
    #Reset Joints
    fa.reset_joints()

    
    all_blocks = []
    while count>0:

        _, frame = cap.read()

        frame = frame[0:450,275:850]
        
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Red color ranges
        lower_red = np.array([129, 196, 61])
        upper_red = np.array([179, 255, 255])

        # Yellow color ranges
        lower_yellow = np.array([11, 150, 175])
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
        mask4 = cv2.inRange(hsv, lower_blue, upper_red)

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

        count-=1
        print(all_blocks)
        print(len(all_blocks))

        cv2.imshow("result", frame)

        k = cv2.waitKey(5)

        if k == 27:

            break


    cap.release()

    cv2.destroyAllWindows()

    object_z_height = 0.019
    intermediate_pose_z_height = 0.19
    

    x_pos = all_blocks[0]['center'][0]
    y_pos = all_blocks[0]['center'][1]
    theta_deg = all_blocks[0]['angle']
    print(x_pos,y_pos, theta_deg)
    x_pos = 0.544 + (x_pos - 205)*(-0.1/159.0)
    x_pos += 0.02
    y_pos = -0.1133 + (y_pos - 317)*(-0.2266/81.0)
    print(x_pos,y_pos, theta_deg)
    print("Press enter to do stuff")
    input()
    
    object_center_pose = fa.get_pose()
    object_center_pose.translation = [x_pos, y_pos, object_z_height]

    theta = (theta_deg/180.0)*np.pi
    new_rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                          [-np.sin(theta), -np.cos(theta), 0],
                          [0, 0, -1]])
    object_center_pose.rotation = new_rotation


    intermediate_robot_pose = object_center_pose.copy()
    intermediate_robot_pose.translation = [x_pos, y_pos, intermediate_pose_z_height]

    #Move to intermediate robot pose
    fa.goto_pose(intermediate_robot_pose)

    fa.goto_pose(object_center_pose, 5, force_thresholds=[10, 10, 10, 10, 10, 10])

    #Close Gripper
    fa.goto_gripper(0.045, grasp=True, force=10.0)
    
    #Move to intermediate robot pose
    fa.goto_pose(intermediate_robot_pose)

    fa.goto_pose(object_center_pose, 5, force_thresholds=[10, 10, 20, 10, 10, 10])

    print('Opening Grippers')
    #Open Gripper
    fa.open_gripper()

    fa.goto_pose(intermediate_robot_pose)

    #Reset Pose
    fa.reset_pose() 
    #Reset Joints
    fa.reset_joints()