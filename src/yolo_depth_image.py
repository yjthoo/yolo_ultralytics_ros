#! /usr/bin/env python

import time
import numpy as np
import ros_numpy
import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
import cv2
from ultralytics import YOLO

class YoloPoseEstimator:
    def __init__(self):
        rospy.init_node("ultralytics_depth")

        # YOLO model loading
        time.sleep(1)
        self.detection_model = YOLO("yolo11m.pt")
        self.segmentation_model = YOLO("yolo11m-seg.pt")

        # ROS Publishers
        self.classes_pub = rospy.Publisher("/ultralytics/detection/distance", String, queue_size=5)
        self.seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=5)

        # Camera intrinsics (initialized as None until received)
        self.fx = None
        self.fy = None
        self.cx_cam = None
        self.cy_cam = None

        # Subscribe to topics
        rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.camera_info_callback)
        rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)

        rospy.loginfo("YoloPoseEstimator node initialized")

    def camera_info_callback(self, msg):
        """Retrieves camera intrinsics dynamically from /camera/depth/camera_info."""
        self.fx = msg.K[0]  # Focal length x
        self.fy = msg.K[4]  # Focal length y
        self.cx_cam = msg.K[2]  # Principal point x
        self.cy_cam = msg.K[5]  # Principal point y
        # rospy.loginfo(f"Camera intrinsics received: fx={self.fx}, fy={self.fy}, cx={self.cx_cam}, cy={self.cy_cam}")

    def depth_callback(self, depth_msg):
        """Processes depth and RGB images to estimate object distance and position."""
        if self.fx is None or self.fy is None:
            rospy.logwarn("Camera intrinsics not received yet, skipping frame...")
            return  # Skip processing until intrinsics are available

        try:
            # Retrieve corresponding RGB frame
            image_msg = rospy.wait_for_message("/camera/color/image_raw", Image, timeout=1.0)
            image = ros_numpy.numpify(image_msg)
            depth = ros_numpy.numpify(depth_msg).astype(np.float32)

            # Convert depth to meters (RealSense usually outputs in mm)
            depth /= 1000.0

            result = self.segmentation_model(image)

            all_objects = []
            for index, cls in enumerate(result[0].boxes.cls):
                class_index = int(cls.cpu().numpy())
                name = result[0].names[class_index]
                mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)

                # Resize mask to match depth image resolution
                resized_mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
                obj = depth[resized_mask == 1]
                obj = obj[~np.isnan(obj)]

                avg_distance = np.mean(obj) if len(obj) else np.inf  # Compute average depth

                # Compute centroid
                # Keep in mind that "Operating Range (Min-Max): .6m-6m" from https://www.intel.com/content/www/us/en/products/sku/205847/intel-realsense-depth-camera-d455/specifications.html
                ys, xs = np.where(resized_mask == 1)
                if len(xs) > 0 and len(ys) > 0 and not np.isinf(avg_distance):
                    cx = np.mean(xs)
                    cy = np.mean(ys)

                    # Convert to camera frame (X, Y, Z)
                    X = (cx - self.cx_cam) * avg_distance / self.fx
                    Y = (cy - self.cy_cam) * avg_distance / self.fy

                    # invert the Y coordinate axis
                    Y = -Y

                    all_objects.append(f"{name}: X={X:.2f}m, Y={Y:.2f}m, distance={avg_distance:.2f}m")
            
            self.classes_pub.publish(String(data=str(all_objects)))

            # Publish YOLO-annotated images with segmentation mask
            if self.seg_image_pub.get_num_connections():
                seg_result = self.segmentation_model(image)
                seg_annotated = seg_result[0].plot(show=False)
                self.seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="rgb8"))

        except rospy.ROSException as e:
            rospy.logwarn(f"Timeout waiting for RGB image: {e}")

if __name__ == "__main__":
    try:
        node = YoloPoseEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
