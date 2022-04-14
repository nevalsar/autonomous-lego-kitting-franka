#!/usr/bin/env python

import rospy
from message_ui.msg import sent_recipe
from message_ui.msg import reply_msg
import time

cup_world = [0.45, 0, 0]  # TODO: record a fixed position

rLower = np.array([160,59,20])
rUpper = np.array([179,255,255])

gLower = np.array([70,69,20])
gUpper = np.array([83,255,255])

bLower = np.array([102,7,28])
bUpper = np.array([111,196,255])

class Test(object):
    def __init__(self):
        self.pub = rospy.Publisher("reply_msg", reply_msg, queue_size=1000)
        rospy.Subscriber('sent_recipe', sent_recipe, self.start_mixing_cb)
        self.status = reply_msg()
        rospy.on_shutdown(self.shutdownhook)

    
    def start_mixing_cb(self, msg):
        self.status.message = "Franka starts making your drink..."
        self.pub.publish(self.status)

        # Testing only. Should be replaced by calling franka
        for _ in range(msg.Red):
            center = find_drink(rgb_image, rLower, rUpper)


        for _ in range(msg.Blue):
            center = find_drink(rgb_image, bLower, bUpper)

        for _ in range(msg.Green):
            center = find_drink(rgb_image, gLower, gUpper)

        self.status.message = "Done. Enjoy!"
        self.pub.publish(self.status)

    
    def shutdownhook(self):
        # works better than the rospy.is_shut_down()
        global ctrl_c
        self.status.message = "Oops, it's time to close. Franka looks forward to serving you next time!"
        self.pub.publish(self.status)
        ctrl_c = True


if __name__ == "__main__":
    ctrl_c = False

    rospy.init_node('test_gui', anonymous=True)
    Test()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("Shutdown time! Stop the robot")