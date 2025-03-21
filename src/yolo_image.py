#! /usr/bin/env python

import time

import ros_numpy
import rospy
from sensor_msgs.msg import Image

from ultralytics import YOLO

class YoloInstanceSegmentation:
    def __init__(self):
        # Initialize node and load models
        rospy.init_node("ultralytics")
        self.detection_model = YOLO("yolo11m.pt")
        self.segmentation_model = YOLO("yolo11m-seg.pt")
        time.sleep(1)

        # ROS publishers
        self.det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)
        self.seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=5)

        # Subscribe to topics
        rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback)

        rospy.loginfo("YoloInstanceSegmentation node initialized")


    def image_callback(self, data):
        """Callback function to process image and publish annotated images."""
        array = ros_numpy.numpify(data)
        if self.det_image_pub.get_num_connections():
            det_result = self.detection_model(array)
            det_annotated = det_result[0].plot(show=False)
            self.det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))

        if self.seg_image_pub.get_num_connections():
            seg_result = self.segmentation_model(array)
            seg_annotated = seg_result[0].plot(show=False)
            self.seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="rgb8"))


if __name__ == "__main__":
    try:
        node = YoloInstanceSegmentation()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass