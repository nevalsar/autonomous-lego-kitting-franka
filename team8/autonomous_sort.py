from frankapy import FrankaArm
import numpy as np
import argparse
import cv2
from cv_bridge import CvBridge
from autolab_core import RigidTransform, Point
from perception import CameraIntrinsics
from utils import *
import imutils

AZURE_KINECT_INTRINSICS = "calib/azure_kinect.intr"
AZURE_KINECT_EXTRINSICS = (
    "calib/azure_kinect_overhead/azure_kinect_overhead_to_world.tf"
)

bin_positions = {}
bin_positions["red"] = [0.24583305, 0.30, 0.10]
bin_positions["yellow"] = [0.35583305, 0.30, 0.10]
bin_positions["blue"] = [0.46583305, 0.30, 0.10]
bin_positions["green"] = [0.57583305, 0.30, 0.10]

IMAGE_X_LOWER_BOUND = 625
IMAGE_X_UPPER_BOUND = 1150
IMAGE_Y_LOWER_BOUND = 475
IMAGE_Y_UPPER_BOUND = 825

INTERMEIDATE_POSE_Z = 0.19
PLANE_Z = 0.003
X_OFFSET_WORLD = 0.01
Y_OFFSET_WORLD = 0.06

# Red color ranges
lower_red = np.array([129, 196, 61])
upper_red = np.array([179, 255, 255])

# Yellow color ranges
lower_yellow = np.array([15, 170, 21])
upper_yellow = np.array([31, 255, 255])

# green color ranges
lower_green = np.array([58, 19, 24])
upper_green = np.array([83, 255, 148])

# blue color ranges
lower_blue = np.array([104, 197, 51])
upper_blue = np.array([155, 255, 255])


def updateBlocks() -> list:
    all_blocks = []

    azure_kinect_rgb_image = get_azure_kinect_rgb_image(cv_bridge)
    object_image_position = np.array([800, 800])
    frame = azure_kinect_rgb_image[
        IMAGE_Y_LOWER_BOUND:IMAGE_Y_UPPER_BOUND,
        IMAGE_X_LOWER_BOUND:IMAGE_X_UPPER_BOUND,
    ]

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

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

    for c in cnts1:
        area1 = cv2.contourArea(c)
        if area1 > 50:
            cv2.drawContours(frame, [c], -1, (0, 255, 0), 3)
            M = cv2.moments(c)
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cx, cy), 2, (255, 255, 255), -1)
            rect = list(cv2.minAreaRect(c))

            red_dict = {"color": "red", "center": (cx, cy), "angle": rect[2]}
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

            yellow_dict = {"color": "yellow", "center": (cx, cy), "angle": rect[2]}
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

            green_dict = {"color": "green", "center": (cx, cy), "angle": rect[2]}
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

            blue_dict = {"color": "blue", "center": (cx, cy), "angle": rect[2]}
            all_blocks.append(blue_dict)
            # cv2.putText(frame, "blue", (cx-20, cy-20),cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 3)

    cv2.namedWindow("image")
    cv2.imshow("image", frame)
    return all_blocks


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--intrinsics_file_path", type=str, default=AZURE_KINECT_INTRINSICS
    )
    parser.add_argument(
        "--extrinsics_file_path", type=str, default=AZURE_KINECT_EXTRINSICS
    )
    args = parser.parse_args()

    # print('Starting robot')
    fa = FrankaArm()

    print("Opening gripper")
    # Open Gripper
    fa.open_gripper()

    # Reset Pose
    fa.reset_pose()
    # Reset Joints
    fa.reset_joints()

    cv_bridge = CvBridge()
    azure_kinect_intrinsics = CameraIntrinsics.load(args.intrinsics_file_path)
    azure_kinect_to_world_transform = RigidTransform.load(args.extrinsics_file_path)
    
    while True:
        all_blocks = updateBlocks()

        # check first before starting pick up
        if cv2.waitKey(5) == 27:
            break

    azure_kinect_depth_image = get_azure_kinect_depth_image(cv_bridge)

    object_center_pose = fa.get_pose()
    bin_pose = fa.get_pose()

    for block in all_blocks:

        x_pos = IMAGE_X_LOWER_BOUND + block["center"][0]
        y_pos = IMAGE_Y_LOWER_BOUND + block["center"][1]
        theta_deg = block["angle"]
        # print(x_pos,y_pos, theta_deg)
        # input()
        object_center_point_in_world = get_object_center_point_in_world(
            x_pos,
            y_pos,
            azure_kinect_depth_image,
            azure_kinect_intrinsics,
            azure_kinect_to_world_transform,
        )

        xloc = object_center_point_in_world[0] + X_OFFSET_WORLD
        yloc = object_center_point_in_world[1] + Y_OFFSET_WORLD

        object_center_pose.translation = [xloc, yloc, PLANE_Z]
        theta = (theta_deg / 180.0) * np.pi
        new_rotation = np.array(
            [
                [np.cos(theta), -np.sin(theta), 0],
                [-np.sin(theta), -np.cos(theta), 0],
                [0, 0, -1],
            ]
        )
        object_center_pose.rotation = new_rotation

        intermediate_robot_pose = object_center_pose.copy()
        intermediate_robot_pose.translation = [xloc, yloc, INTERMEIDATE_POSE_Z]

        # Move to intermediate robot pose
        print("Moving to intermediate pose")
        fa.goto_pose(intermediate_robot_pose)

        print("Moving to object")
        fa.goto_pose(object_center_pose, 5, force_thresholds=[10, 10, 10, 10, 10, 10])

        # Close Gripper
        print("Closing gripper")
        fa.goto_gripper(0.045, grasp=True, force=10.0)

        # Move to intermediate robot pose
        print("Moving to intermediate pose")
        fa.goto_pose(intermediate_robot_pose)
        
        fa.reset_joints()

        # go to bin
        print("Moving to appropriate bin position")
        bin_pose.translation = bin_positions[block["color"]]
        fa.goto_pose(bin_pose, 5, force_thresholds=[10, 10, 10, 10, 10, 10])

        print("Opening gripper")
        # Open Gripper
        fa.open_gripper()

        # fa.goto_pose(intermediate_robot_pose)
        print("Resetting joints")
        fa.reset_joints()

    # Reset Pose
    fa.reset_pose()
    # Reset Joints
    fa.reset_joints()
    cv2.destroyAllWindows()
