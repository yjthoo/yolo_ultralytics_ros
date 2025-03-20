#! /usr/bin/env python

import time

import numpy as np
import ros_numpy
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String
import cv2

from ultralytics import YOLO

rospy.init_node("ultralytics_depth")
time.sleep(1)

detection_model = YOLO("yolo11m.pt")
segmentation_model = YOLO("yolo11m-seg.pt")

classes_pub = rospy.Publisher("/ultralytics/detection/distance", String, queue_size=5)
det_image_pub = rospy.Publisher("/ultralytics/detection/image", Image, queue_size=5)
seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=5)

# camera intrinsics
PATH = "/home/humanist/pmb2_public_ws/src/aruco_detector/calibration/"
matrix_coefficients = np.load(PATH + "calibration_matrix.npy")
distortion_coefficients = np.load(PATH + "distortion_coefficients.npy")

fx = matrix_coefficients[0, 0]
fy = matrix_coefficients[1, 1]
cx_cam = matrix_coefficients[0, 2]
cy_cam = matrix_coefficients[1, 2]


def callback(data):
    """Callback function to process depth image and RGB image."""
    image = rospy.wait_for_message("/camera/color/image_raw", Image)
    image = ros_numpy.numpify(image)
    depth = ros_numpy.numpify(data)
    result = segmentation_model(image)

    all_objects = []
    for index, cls in enumerate(result[0].boxes.cls):
        class_index = int(cls.cpu().numpy())
        name = result[0].names[class_index]
        mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)

        # Resize mask to depth resolution
        resized_mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
        obj = depth[resized_mask == 1] # depth[mask == 1]

        obj = obj[~np.isnan(obj)]
        avg_distance = np.mean(obj) if len(obj) else np.inf
        avg_distance = avg_distance/1000

        # Find centroid in pixel coordinates
        # Keep in mind that "Operating Range (Min-Max): .6m-6m" from https://www.intel.com/content/www/us/en/products/sku/205847/intel-realsense-depth-camera-d455/specifications.html
        ys, xs = np.where(resized_mask == 1)
        if len(xs) > 0 and len(ys) > 0:
            cx = np.mean(xs)
            cy = np.mean(ys)
            # Back-project to camera coordinates
            X = (cx - cx_cam) * avg_distance / fx
            Y = (cy - cy_cam) * avg_distance / fy
        else:
            X, Y = None, None

        all_objects.append(f"{name}: X={X:.2f}m, Y={Y:.2f}m, distance={avg_distance:.2f}m")

    classes_pub.publish(String(data=str(all_objects)))

    if det_image_pub.get_num_connections():
        det_result = detection_model(image)
        det_annotated = det_result[0].plot(show=False)
        det_image_pub.publish(ros_numpy.msgify(Image, det_annotated, encoding="rgb8"))

    if seg_image_pub.get_num_connections():
        seg_result = segmentation_model(image)
        seg_annotated = seg_result[0].plot(show=False)
        seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="rgb8"))


rospy.Subscriber("/camera/depth/image_rect_raw", Image, callback)

while True:
    rospy.spin()