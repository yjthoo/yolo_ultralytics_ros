#! /usr/bin/env python

import time
import numpy as np
import ros_numpy
import rospy
from sensor_msgs.msg import Image, CameraInfo
from std_msgs.msg import String
import cv2
from ultralytics import YOLO
import json

class YoloPoseEstimator:
    def __init__(self):
        rospy.init_node("ultralytics_depth")

        # YOLO model loading
        time.sleep(1)
        self.detection_model = YOLO("yolo11m.pt")
        self.segmentation_model = YOLO("yolo11m-seg.pt")
        self.MIN_CONFIDENCE = 0.5

        # ROS Publishers
        self.classes_pub = rospy.Publisher("/ultralytics/detection/distance", String, queue_size=5)
        self.seg_image_pub = rospy.Publisher("/ultralytics/segmentation/image", Image, queue_size=5)

        # Store latest detections
        self.latest_detections = None

        # Camera intrinsics (initialized as None until received)
        self.fx = None
        self.fy = None
        self.cx_cam = None
        self.cy_cam = None

        # Subscribe to topics
        rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.camera_info_callback)
        rospy.Subscriber("/camera/depth/image_rect_raw", Image, self.depth_callback)

        # Add timer to publish detections with given rate
        rospy.Timer(rospy.Duration(1.0), self.publish_detections)

        rospy.loginfo("YoloPoseEstimator node initialized")

    def camera_info_callback(self, msg):
        """Retrieves camera intrinsics dynamically from /camera/depth/camera_info."""
        self.fx = msg.K[0]  # Focal length x
        self.fy = msg.K[4]  # Focal length y
        self.cx_cam = msg.K[2]  # Principal point x
        self.cy_cam = msg.K[5]  # Principal point y

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

            detected_objects = []
            for index, cls in enumerate(result[0].boxes.cls):

                confidence = float(result[0].boxes.conf[index].cpu().numpy())
                if confidence < self.MIN_CONFIDENCE:
                    continue  # skip low-confidence detections
                
                # class index, name, and bounding box 
                class_index = int(cls.cpu().numpy())
                name = result[0].names[class_index]
                mask = result[0].masks.data.cpu().numpy()[index, :, :].astype(int)

                # Bounding box size (in pixels)
                x_min, y_min, x_max, y_max = result[0].boxes.xyxy[index].cpu().numpy()
                box_width_px  = x_max - x_min
                box_height_px = y_max - y_min

                # Resize mask to match depth image resolution
                resized_mask = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST)
                obj = depth[resized_mask == 1]
                obj = obj[~np.isnan(obj)]

                avg_distance = np.mean(obj) if len(obj) else np.inf  # Compute average depth

                # Compute centroid
                # Keep in mind that "Operating Range (Min-Max): .6m-6m" from https://www.intel.com/content/www/us/en/products/sku/205847/intel-realsense-depth-camera-d455/specifications.html
                ys, xs = np.where(resized_mask == 1)
                if len(xs) > 0 and len(ys) > 0 and not np.isinf(avg_distance):
                    cx_mask = np.mean(xs)
                    cy_mask = np.mean(ys)
                    X_mask = (cx_mask - self.cx_cam) * avg_distance / self.fx
                    Y_mask = -(cy_mask - self.cy_cam) * avg_distance / self.fy  # Inverted for Unity convention

                    # Bounding box center
                    cx_bbox = (x_min + x_max) / 2
                    cy_bbox = (y_min + y_max) / 2
                    X_bbox = (cx_bbox - self.cx_cam) * avg_distance / self.fx
                    Y_bbox = -(cy_bbox - self.cy_cam) * avg_distance / self.fy

                    # Estimate real-world size (in meters)
                    real_width  = float(round(box_width_px  * avg_distance / self.fx, 3))
                    real_height = float(round(box_height_px * avg_distance / self.fy, 3))

                    detected_objects.append({
                        "name": name,
                        "X_mask": float(round(X_mask, 3)),
                        "Y_mask": float(round(Y_mask, 3)),
                        "X_bbox": float(round(X_bbox, 3)),
                        "Y_bbox": float(round(Y_bbox, 3)),
                        "Z": float(round(avg_distance, 3)),
                        "width": real_width,
                        "height": real_height,
                        "confidence": round(confidence, 3)
                    })
            
            # Store latest detections for timer-based publishing
            self.latest_detections = detected_objects

            # Publish YOLO-annotated images with segmentation mask
            if self.seg_image_pub.get_num_connections():
                seg_result = self.segmentation_model(image)
                seg_annotated = seg_result[0].plot(show=False)
                self.seg_image_pub.publish(ros_numpy.msgify(Image, seg_annotated, encoding="rgb8"))

        except rospy.ROSException as e:
            rospy.logwarn(f"Timeout waiting for RGB image: {e}")

    def publish_detections(self, event):
        """Publish latest detections once per second."""

        if self.latest_detections:
            msg_json = json.dumps({"objects": self.latest_detections})
            #rospy.loginfo(f"Publishing detections: {msg_json}")
            self.classes_pub.publish(String(data=msg_json))


if __name__ == "__main__":
    try:
        node = YoloPoseEstimator()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
