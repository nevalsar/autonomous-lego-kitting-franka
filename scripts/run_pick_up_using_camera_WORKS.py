from frankapy import FrankaArm
import numpy as np
import argparse
import cv2
from cv_bridge import CvBridge
from autolab_core import RigidTransform, Point
from perception import CameraIntrinsics
from utils import *
import imutils

AZURE_KINECT_INTRINSICS = 'calib/azure_kinect.intr'
AZURE_KINECT_EXTRINSICS = 'calib/azure_kinect_overhead/azure_kinect_overhead_to_world.tf'

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--intrinsics_file_path', type=str, default=AZURE_KINECT_INTRINSICS)
    parser.add_argument('--extrinsics_file_path', type=str, default=AZURE_KINECT_EXTRINSICS) 
    args = parser.parse_args()

    print('Starting robot')
    fa = FrankaArm()    

    print('Opening Grippers')
    #Open Gripper
    fa.open_gripper()

    #Reset Pose
    fa.reset_pose() 
    #Reset Joints
    fa.reset_joints()

    cv_bridge = CvBridge()
    azure_kinect_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
    azure_kinect_to_world_transform = RigidTransform.load(args.extrinsics_file_path)    
    while True:

        azure_kinect_rgb_image = get_azure_kinect_rgb_image(cv_bridge)
        azure_kinect_depth_image = get_azure_kinect_depth_image(cv_bridge)
        
        object_image_position = np.array([800, 800])
        frame = azure_kinect_rgb_image
        frame = frame[200:850,500:1300]

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
        # print(len(all_blocks))
        x_pos = 500+all_blocks[0]['center'][0]
        y_pos = 200+all_blocks[0]['center'][1]
        object_center_point_in_world = get_object_center_point_in_world(x_pos,
                                                                    y_pos,
                                                                    azure_kinect_depth_image, azure_kinect_intrinsics,
                                                                    azure_kinect_to_world_transform)
        xloc = object_center_point_in_world[0] + 0.01
        yloc = object_center_point_in_world[1] + 0.07
        print(xloc,yloc)
        # def onMouse(event, x, y, flags, param):
        #     if event == cv2.EVENT_LBUTTONDOWN:
        #         print('x = %d, y = %d'%(x, y))
        #         param[0] = x
        #         param[1] = y
            
        cv2.namedWindow('image')
        # cv2.imshow('image', azure_kinect_rgb_image)
        cv2.imshow('image', frame)
        # cv2.setMouseCallback('image', onMouse, object_image_position)
        break
        if (cv2.waitKey(5)==27): 
            break
    cv2.destroyAllWindows()
    for block in all_blocks:

        object_z_height = 0.021
        intermediate_pose_z_height = 0.19
        x_pos = 500+block['center'][0]
        y_pos = 200+block['center'][1]
        theta_deg = block['angle']
        print(x_pos,y_pos, theta_deg)
        # input()
        object_center_point_in_world = get_object_center_point_in_world(x_pos,
                                                                        y_pos,
                                                                        azure_kinect_depth_image, azure_kinect_intrinsics,
                                                                        azure_kinect_to_world_transform)
        # x_pos = 0.54399961
        # y_pos = -0.11331095
        xloc = object_center_point_in_world[0] + 0.01
        yloc = object_center_point_in_world[1] + 0.06
        object_center_pose = fa.get_pose()
        object_center_pose.translation = [xloc, yloc, object_z_height]
        print(xloc,yloc)
        print('press enter to continue')
        # input()
        theta = (theta_deg/180.0)*np.pi
        new_rotation = np.array([[np.cos(theta), -np.sin(theta), 0],
                            [-np.sin(theta), -np.cos(theta), 0],
                            [0, 0, -1]])
        object_center_pose.rotation = new_rotation

        intermediate_robot_pose = object_center_pose.copy()
        intermediate_robot_pose.translation = [xloc, yloc, intermediate_pose_z_height]

        #Move to intermediate robot pose
        fa.goto_pose(intermediate_robot_pose)

        fa.goto_pose(object_center_pose, 5, force_thresholds=[10, 10, 10, 10, 10, 10])
        print('press enter to continue')
        # input()
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