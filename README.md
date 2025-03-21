# YOLO ultralytics ROS

This ROS package, tested for ROS Noetic on Ubuntu 20.04.6 LTS, uses the Intel Realsense D455 camera to detect and estimate the pose of objects/obstacles in the environment following the [Ultralytics YOLO with ROS documentation](https://docs.ultralytics.com/guides/ros-quickstart/#setting-up-ultralytics-yolo-with-ros). 

It uses the [realsense-ros](https://github.com/IntelRealSense/realsense-ros/tree/ros1-legacy) package which will be need to installed in your catkin workspace. To launch the camera from ROS and gain access to the required data, you will need to run the following command:

```
roslaunch realsense2_camera rs_camera.launch filters:=pointcloud align_depth:=true
```

You can then run the following command, in another terminal, and observe the results of the image segmentation via the opened Rviz window:

```
roslaunch yolo_ultralytics_ros yolo_depth_image.launch 
```

which publishes the estimated distance of detected objects from the camera based on its depth map. To observe these estimations, you can run the following command from another terminal:

```
rostopic echo /ultralytics/detection/distance 
```