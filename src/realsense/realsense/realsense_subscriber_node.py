# import rclpy
# from rclpy.node import Node

# class RealsenseSubscriber(Node):
#     def __init__(self):
#         super().__init__('realsense_subscriber_node')
#         self.get_logger().info('RealSense Subscriber Node Started')

# def main(args=None):
#     rclpy.init(args=args)
#     node = RealsenseSubscriber()
#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         pass
#     finally:
#         node.destroy_node()
#         rclpy.shutdown()

# if __name__ == '__main__':
#     main()

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# import cv2
# from cv_bridge import CvBridge
# import time

# class RealsenseSubscriber(Node):
#     def __init__(self):
#         super().__init__('realsense_subscriber_node')
#         self.get_logger().info('RealSense Subscriber Node Started')
        
#         self.color_subscription = self.create_subscription(
#             Image, 
#             '/camera/realsense_node/color/image_raw', 
#             self.color_callback, 
#             10)
        
#         self.bridge = CvBridge()
#         self.prev_time = time.time()  # Store the initial time
#         self.fps = 0

#     def color_callback(self, msg):
#         # Convert ROS Image message to OpenCV format
#         cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

#         # Calculate FPS
#         curr_time = time.time()
#         self.fps = 1.0 / (curr_time - self.prev_time)
#         self.prev_time = curr_time

#         # Overlay FPS on the frame
#         cv2.putText(cv_image, f"FPS: {self.fps:.2f}", (10, 30), 
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Display the frame
#         cv2.imshow("RealSense Color Stream", cv_image)
        
#         # Use waitKey with a small delay to allow OpenCV to process events
#         key = cv2.waitKey(1)
#         if key == 27:  # Press 'ESC' to exit
#             rclpy.shutdown()

# def main(args=None):
#     rclpy.init(args=args)
#     realsense_subscriber = RealsenseSubscriber()
#     rclpy.spin(realsense_subscriber)
#     realsense_subscriber.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# import cv2
# from cv_bridge import CvBridge
# import time
# import torch
# from ultralytics import YOLO
# from yolov8_msgs.msg import Yolov8Inference, InferenceResult

# class RealsenseSubscriber(Node):
#     def __init__(self):
#         super().__init__('realsense_subscriber_node')
#         self.get_logger().info('RealSense Subscriber Node Started')

#         # Subscription to RealSense color stream
#         self.color_subscription = self.create_subscription(
#             Image,
#             '/camera/realsense_node/color/image_raw',
#             self.color_callback,
#             10
#         )

#         # Publishers
#         self.yolov8_pub = self.create_publisher(Yolov8Inference, "/Yolov8_Inference", 1)
#         self.img_pub = self.create_publisher(Image, "/inference_result", 1)

#         # YOLOv8 model initialization
#         self.model = YOLO('/home/james/realsense_ws/src/realsense/model/solar_panel.pt')

#         self.bridge = CvBridge()
#         self.prev_time = time.time()
#         self.fps = 0

#     def color_callback(self, msg):
#         # Convert ROS Image message to OpenCV format
#         cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

#         # Calculate FPS
#         curr_time = time.time()
#         self.fps = 1.0 / (curr_time - self.prev_time)
#         self.prev_time = curr_time

#         # Run YOLOv8 inference
#         results = self.model(cv_image)[0]  # YOLOv8 inference on the frame

#         # Process detections
#         for box in results.boxes.data.tolist():
#             x1, y1, x2, y2, conf, cls = box  # Bounding box coordinates, confidence, and class ID
#             label = f"{self.model.names[int(cls)]}: {conf:.2f}"
#             cv2.rectangle(cv_image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
#             cv2.putText(cv_image, label, (int(x1), int(y1) - 10),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

#         # Overlay FPS on the frame
#         cv2.putText(cv_image, f"FPS: {self.fps:.2f}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#         # Publish inference result as an image
#         inference_image_msg = self.bridge.cv2_to_imgmsg(cv_image, "bgr8")
#         self.img_pub.publish(inference_image_msg)

#         # Publish inference data
#         inference_msg = Yolov8Inference()
#         inference_msg.header.stamp = self.get_clock().now().to_msg()
#         inference_msg = Yolov8Inference()
#         inference_msg.header.stamp = self.get_clock().now().to_msg()

#         inference_msg.yolov8_inference = []  # Ensure it's a list

#         for box in results.boxes.data.tolist():
#             x1, y1, x2, y2, conf, cls = box
#             detection = InferenceResult()
#             detection.class_name = results.names[int(box.cls[0])]  # Set the detected class name
#             detection.left = int(box.xyxy[0][0])  # Use `left` instead of `x1`
#             detection.top = int(box.xyxy[0][1])   # Use `top` instead of `y1`
#             detection.right = int(box.xyxy[0][2]) # Use `right` instead of `x2`
#             detection.bottom = int(box.xyxy[0][3]) # Use `bottom` instead of `y2`
#             detection.confidence = float(conf)
#             detection.class_id = int(cls)

#             inference_msg.yolov8_inference.append(detection)

#         self.yolov8_pub.publish(inference_msg)

#         # Display the frame with YOLOv8 detections
#         cv2.imshow("RealSense YOLOv8 Inference", cv_image)
#         key = cv2.waitKey(1)
#         if key == 27:  # Press 'ESC' to exit
#             rclpy.shutdown()

# def main(args=None):
#     rclpy.init(args=args)
#     realsense_subscriber = RealsenseSubscriber()
#     rclpy.spin(realsense_subscriber)
#     realsense_subscriber.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from std_msgs.msg import Header
# from geometry_msgs.msg import Twist, Vector3
# from yolov8_msgs.msg import Yolov8Inference, InferenceResult  # Custom message types
# import cv2
# from cv_bridge import CvBridge
# import torch
# from ultralytics import YOLO

# class RealSenseSubscriber(Node):
#     def __init__(self):
#         super().__init__('realsense_subscriber_node')

#         # Initialize OpenCV bridge
#         self.bridge = CvBridge()

#         # Load YOLOv8 model
#         self.model = YOLO('/home/james/realsense_ws/src/realsense/model/solar_panel.pt')  # YOLOv8 Segmentation Model

#         # ROS2 Subscribers
#         self.color_sub = self.create_subscription(
#             Image, 
#             '/camera/realsense_node/color/image_raw', 
#             self.color_callback, 
#             10)

#         # ROS2 Publishers
#         self.inference_pub = self.create_publisher(Yolov8Inference, '/yolov8/detections', 10)
#         self.img_pub = self.create_publisher(Image, "/yolov8/annotated_image", 10)
#         self.cmd_vel_publisher = self.create_publisher(Twist, "/cmd_vel", 10)

#         # Movement control variables
#         self.counter = 0
#         self.counter_limit = 10
#         self.distance_front = False

#     def color_callback(self, msg):
#         """Processes incoming image, runs YOLOv8 segmentation, and publishes results."""
#         try:
#             # Convert ROS2 Image to OpenCV format
#             img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

#             # Run YOLOv8 inference
#             results = self.model(img)

#             # Frame dimensions
#             frame_width = img.shape[1]
#             frame_height = img.shape[0]
#             center_region_width = 40  # Central region width
#             center_region_height = 60  # Central region height

#             # Initialize inference message
#             inference_msg = Yolov8Inference()
#             inference_msg.header.stamp = self.get_clock().now().to_msg()
#             inference_msg.header.frame_id = "inference"

#             for r in results:
#                 masks = r.masks  # Segmentation masks
#                 confidences = r.boxes.conf

#                 if masks is not None:
#                     for mask, confidence in zip(masks.data, confidences):

#                         if confidence < 0.5:  # 🚀 Filter out detections below 0.5 confidence
#                             continue  

#                         # Compute centroid of mask
#                         mask_np = mask.cpu().numpy().astype('uint8')
#                         M = cv2.moments(mask_np)
#                         if M['m00'] != 0:
#                             center_x = int(M['m10'] / M['m00'])
#                             center_y = int(M['m01'] / M['m00'])

#                             # Movement control logic
#                             linear_vec = Vector3()
#                             angular_vec = Vector3()

#                             # Define central region
#                             central_x_min = (frame_width - center_region_width) // 2
#                             central_x_max = (frame_width + center_region_width) // 2
#                             central_y_min = (frame_height - center_region_height) // 2
#                             central_y_max = (frame_height + center_region_height) // 2

#                             if center_x < central_x_min:
#                                 linear_vec.y = 0.2  # Move left
#                             elif center_x > central_x_max:
#                                 linear_vec.y = -0.2  # Move right

#                             if center_y < central_y_min:
#                                 linear_vec.z = 0.2  # Move up
#                             elif center_y > central_y_max:
#                                 linear_vec.z = -0.2  # Move down

#                             if (central_x_min <= center_x <= central_x_max and 
#                                 central_y_min <= center_y <= central_y_max):
#                                 self.counter += 1
#                                 if self.counter >= self.counter_limit:
#                                     self.distance_front = True
#                                     self.counter = 0

#                             # Publish movement command
#                             twist = Twist(linear=linear_vec, angular=angular_vec)
#                             self.cmd_vel_publisher.publish(twist)

#             # Annotate frame with detection results
#             annotated_frame = results[0].plot()
#             img_msg = self.bridge.cv2_to_imgmsg(annotated_frame, encoding="bgr8")

#             # Publish annotated image and inference results
#             self.img_pub.publish(img_msg)
#             self.inference_pub.publish(inference_msg)

#         except Exception as e:
#             self.get_logger().error(f"Error processing image: {str(e)}")

# def main(args=None):
#     """ROS2 node entry point."""
#     rclpy.init(args=args)
#     realsense_subscriber = RealSenseSubscriber()
#     rclpy.spin(realsense_subscriber)
#     realsense_subscriber.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

# import rclpy
# from rclpy.node import Node
# from sensor_msgs.msg import Image
# from geometry_msgs.msg import Twist, Vector3
# from yolov8_msgs.msg import Yolov8Inference, InferenceResult
# from std_msgs.msg import Bool
# import cv2
# from cv_bridge import CvBridge
# from ultralytics import YOLO

# class RealSenseSubscriber(Node):
#     def __init__(self):
#         super().__init__('realsense_subscriber_node')

#         # Initialize OpenCV bridge
#         self.bridge = CvBridge()

#         # Load YOLOv8 model
#         self.model = YOLO('/home/james/realsense_ws/src/realsense/model/solar_panel.pt')

#         # ROS2 Subscribers
#         self.color_sub = self.create_subscription(
#             Image, 
#             '/camera/realsense_node/color/image_raw', 
#             self.color_callback, 
#             10)

#         # ROS2 Publishers
#         self.inference_pub = self.create_publisher(Yolov8Inference, '/yolov8/detections', 10)
#         self.img_pub = self.create_publisher(Image, "/yolov8/annotated_image", 10)
#         self.cmd_vel_publisher = self.create_publisher(Twist, "/offboard_velocity_cmd", 10)
#         self.arm_publisher = self.create_publisher(Bool, "/arm_message", 10)

#         # Movement control variables
#         self.counter = 0
#         self.counter_limit = 10
#         self.distance_front = False

#         # Automatically arm the drone when the node starts
#         self.arm_drone()

#     def arm_drone(self):
#         arm_msg = Bool()
#         arm_msg.data = True
#         self.arm_publisher.publish(arm_msg)
#         self.get_logger().info("Drone armed.")

#     # def color_callback(self, msg):
#     #     try:
#     #         img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

#     #         results = self.model(img)

#     #         frame_width = img.shape[1]
#     #         frame_height = img.shape[0]
#     #         center_region_width = 300  # 300 px width threshold
#     #         center_region_height = 100  # 100 px height threshold

#     #         inference_msg = Yolov8Inference()
#     #         inference_msg.header.stamp = self.get_clock().now().to_msg()
#     #         inference_msg.header.frame_id = "inference"

#     #         # Define center bounding box region
#     #         central_x_min = (frame_width - center_region_width) // 2
#     #         central_x_max = (frame_width + center_region_width) // 2
#     #         central_y_min = (frame_height - center_region_height) // 2
#     #         central_y_max = (frame_height + center_region_height) // 2

#     #         # Draw center bounding box (red rectangle)
#     #         cv2.rectangle(img, (central_x_min, central_y_min), (central_x_max, central_y_max), (0, 0, 255), 2)

#     #         # Variables to track global bounding box
#     #         x_min, y_min, x_max, y_max = float("inf"), float("inf"), float("-inf"), float("-inf")
#     #         object_detected = False  # Track if any object is detected

#     #         for r in results:
#     #             masks = r.masks
#     #             confidences = r.boxes.conf
#     #             boxes = r.boxes.xyxy  # Bounding boxes (xyxy format)

#     #             if masks is not None and boxes is not None:
#     #                 for confidence, box in zip(confidences, boxes):
#     #                     if confidence < 0.5:
#     #                         continue

#     #                     object_detected = True  # At least one object is detected

#     #                     # Extract bounding box coordinates
#     #                     x1, y1, x2, y2 = map(int, box.tolist())

#     #                     # Expand global bounding box to include this box
#     #                     x_min = min(x_min, x1)
#     #                     y_min = min(y_min, y1)
#     #                     x_max = max(x_max, x2)
#     #                     y_max = max(y_max, y2)

#     #         if object_detected:
#     #             # Compute the fused center point
#     #             bbox_center_x = (x_min + x_max) // 2
#     #             bbox_center_y = (y_min + y_max) // 2

#     #             print(f"Fused Bounding Box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
#     #             print(f"Fused Center Point: ({bbox_center_x}, {bbox_center_y})")

#     #             # Draw the fused bounding box (blue)
#     #             cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (255, 0, 0), 2)

#     #             # Draw the fused center point (red circle)
#     #             cv2.circle(img, (bbox_center_x, bbox_center_y), 5, (0, 0, 255), -1)

#     #             # Movement logic based on the fused center
#     #             linear_vec = Vector3()
#     #             angular_vec = Vector3()

#     #             if bbox_center_x < central_x_min:
#     #                 linear_vec.y = 1.0  # Move left
#     #             elif bbox_center_x > central_x_max:
#     #                 linear_vec.y = -1.0  # Move right

#     #             if bbox_center_y < central_y_min:
#     #                 linear_vec.z = 1.0  # Move up
#     #             elif bbox_center_y > central_y_max:
#     #                 linear_vec.z = -1.0  # Move down

#     #             twist = Twist(linear=linear_vec, angular=angular_vec)
#     #             self.cmd_vel_publisher.publish(twist)

#     #         # Image publishing
#     #         annotated_frame = results[0].plot()

#     #         # Overlay bounding box and center marker on the final annotated image
#     #         final_image = cv2.addWeighted(annotated_frame, 0.8, img, 0.2, 0)

#     #         img_msg = self.bridge.cv2_to_imgmsg(final_image, encoding="bgr8")
#     #         self.img_pub.publish(img_msg)
#     #         self.inference_pub.publish(inference_msg)

#     #     except Exception as e:
#     #         self.get_logger().error(f"Error processing image: {str(e)}")

#     def color_callback(self, msg):
#         try:
#             img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

#             results = self.model(img)

#             frame_width = img.shape[1]
#             frame_height = img.shape[0]
#             center_region_width = 300  # 300 px width threshold
#             center_region_height = 100  # 100 px height threshold

#             inference_msg = Yolov8Inference()
#             inference_msg.header.stamp = self.get_clock().now().to_msg()
#             inference_msg.header.frame_id = "inference"

#             # Define center bounding box region
#             central_x_min = (frame_width - center_region_width) // 2
#             central_x_max = (frame_width + center_region_width) // 2
#             central_y_min = (frame_height - center_region_height) // 2
#             central_y_max = (frame_height + center_region_height) // 2

#             # Draw center bounding box (red rectangle)
#             cv2.rectangle(img, (central_x_min, central_y_min), (central_x_max, central_y_max), (0, 0, 255), 2)

#             # Variables to track the lowest-right object
#             selected_bbox = None
#             max_position_sum = float("-inf")

#             for r in results:
#                 masks = r.masks
#                 confidences = r.boxes.conf
#                 boxes = r.boxes.xyxy  # Bounding boxes (xyxy format)

#                 if masks is not None and boxes is not None:
#                     for confidence, box in zip(confidences, boxes):
#                         if confidence < 0.5:
#                             continue

#                         # Extract bounding box coordinates
#                         x1, y1, x2, y2 = map(int, box.tolist())

#                         # Calculate "right-bottom priority" score
#                         position_sum = x2 + y2  # Prioritize the object with the highest x_max + y_max

#                         # Update the lowest-right object selection
#                         if position_sum > max_position_sum:
#                             max_position_sum = position_sum
#                             selected_bbox = (x1, y1, x2, y2)

#             if selected_bbox:
#                 x_min, y_min, x_max, y_max = selected_bbox

#                 # Compute the center point of the selected object
#                 bbox_center_x = (x_min + x_max) // 2
#                 bbox_center_y = (y_min + y_max) // 2

#                 print(f"Selected Bounding Box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
#                 print(f"Selected Center Point: ({bbox_center_x}, {bbox_center_y})")

#                 # Draw the selected bounding box (green)
#                 cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

#                 # Draw the center point (red circle)
#                 cv2.circle(img, (bbox_center_x, bbox_center_y), 5, (0, 0, 255), -1)

#                 # Movement logic based on the selected object's center
#                 linear_vec = Vector3()
#                 angular_vec = Vector3()

#                 if bbox_center_x < central_x_min:
#                     linear_vec.y = 1.0  # Move left
#                 elif bbox_center_x > central_x_max:
#                     linear_vec.y = -1.0  # Move right

#                 if bbox_center_y < central_y_min:
#                     linear_vec.z = 1.0  # Move up
#                 elif bbox_center_y > central_y_max:
#                     linear_vec.z = -1.0  # Move down

#                 twist = Twist(linear=linear_vec, angular=angular_vec)
#                 self.cmd_vel_publisher.publish(twist)

#             # Image publishing
#             annotated_frame = results[0].plot()

#             # Overlay bounding box and center marker on the final annotated image
#             final_image = cv2.addWeighted(annotated_frame, 0.8, img, 0.2, 0)

#             img_msg = self.bridge.cv2_to_imgmsg(final_image, encoding="bgr8")
#             self.img_pub.publish(img_msg)
#             self.inference_pub.publish(inference_msg)

#         except Exception as e:
#             self.get_logger().error(f"Error processing image: {str(e)}")

# def main(args=None):
#     rclpy.init(args=args)
#     realsense_subscriber = RealSenseSubscriber()
#     rclpy.spin(realsense_subscriber)
#     realsense_subscriber.destroy_node()
#     rclpy.shutdown()

# if __name__ == '__main__':
#     main()

# pip3 install openpyxl
import rclpy
from rclpy.node import Node
from px4_msgs.msg import VehicleCommand, OffboardControlMode, TrajectorySetpoint, VehicleOdometry, VehicleStatus
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import time
import math
import numpy as np
import logging
from cv_bridge import CvBridge
# logging.getLogger('ultralytics').setLevel(logging.ERROR)
from geometry_msgs.msg import Point
from std_msgs.msg import Float64, Float32MultiArray
import matplotlib.pyplot as plt
import open3d as o3d 
import sensor_msgs_py.point_cloud2 as pc2
from sensor_msgs.msg import Image, PointCloud2
from yolov8_msgs.msg import Yolov8Inference
import cv2
from ultralytics import YOLO
import os
from datetime import datetime
import pandas as pd

class ZDCalNode(Node):
    def __init__(self):
        super().__init__('realsense_subscriber_node')

        # Initialize OpenCV bridge
        self.bridge = CvBridge()

        # Load YOLOv8 model
        self.model = YOLO('/home/james/realsense_ws/src/realsense/model/solar_panel.pt')

        # Define QoS profile for PX4 compatibility
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Publishers
        self.inference_pub = self.create_publisher(Yolov8Inference, '/yolov8/detections', 10)
        self.img_pub = self.create_publisher(Image, "/yolov8/annotated_image", 10)

        self.vehicle_command_publisher = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.offboard_control_mode_publisher = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        
        self.drop_point_pub = self.create_publisher(Point, 'drop_point', 10)
        self.inclination_pub = self.create_publisher(Float64, 'solar_panel_inclination' , 10)

        # ROS2 Subscribers
        self.color_sub = self.create_subscription(
            Image, 
            '/iris/camera/image_raw', #'/camera/realsense_node/color/image_raw', 
            self.color_callback, 
            10)
    
        self.depth_sub = self.create_subscription(
            PointCloud2,                              
            '/iris/camera/points', 
            self.depth_callback,       
            10)
        
        self.create_subscription(
            VehicleOdometry,
            '/fmu/out/vehicle_odometry',
            self.odometry_callback,
            qos_profile
        )

        self.create_subscription(
            VehicleStatus,
            '/fmu/out/vehicle_status_v1', # '/fmu/out/vehicle_status',
            self.vehicle_status_callback,
            qos_profile
        )
        
        self.retake_mask = None
        self.frame_width = None
        self.frame_height = None
        self.mask_np = np.zeros((480, 640), dtype=np.uint8)

        # State variables
        self.current_position = [0.0, 0.0, 0.0]  # Current position in NED
        self.angular_velocity = [0.0, 0.0, 0.0]
        self.current_mode = None  # Current flight mode
        self.armed = False  # Armed state
        self.offboard_mode = "VELOCITY"
        self.state = "ARMING"  # State machine state
        self.takeoff_altitude = -8.0  # Takeoff altitude in NED (8 meters up)
        
        # Timing variables
        self.state_change_time = self.get_clock().now()
        self.current_yaw = 0.0  # Instantaneous target yaw
        self.running = True

        # Movement control variables
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0
        self.maintain_z = False  # Flag to indicate whether to maintain z position
        self.moving_in_z = False  # Flag to indicate explicit z movement
        self.target_acquired = False
        self.last_detection_time = None  # Track when an object was last detected
        self.initial_yaw = None  # Will store the yaw at takeoff completion

        # Timer to control the state machine
        self.timer = self.create_timer(0.05, self.timer_callback)  # 20Hz
        self.publish_offboard_timer = self.create_timer(0.01, self.publish_offboard_control_mode)  # 100Hz
        self.last_movement_time = None

        # Teraranger LIDAR Subscription
        # Create subscriber
        self.create_subscription(
            Float32MultiArray,
            '/teraranger_evo/distances',
            self.teraranger_callback,
            10
        )
        
        # Define sensor positions [front, right, back, left]
        self.directions = ['front', 'right', 'back', 'left']
        self.processed_distances = [float('inf'), float('inf'), float('inf'), float('inf')]
        
        # Max distance when sensor returns inf (in meters)
        self.lidar_max_distance = 60.0

        # Obstacle avoidance parameters
        self.safe_distance = 5.0  # Distance threshold in meters to start slowing down
        self.min_distance = 2.0   # Minimum distance for complete stop
        
        # PID controller parameters for velocity control
        self.kp = 0.3  # Proportional gain
        self.ki = 0.05 # Integral gain
        self.kd = 0.1  # Derivative gain
        
        # PID controller state variables for each direction
        self.prev_errors = {'front': 0.0, 'right': 0.0, 'back': 0.0, 'left': 0.0}
        self.integral_terms = {'front': 0.0, 'right': 0.0, 'back': 0.0, 'left': 0.0}
        self.last_time = {'front': None, 'right': None, 'back': None, 'left': None}
        
        # Data logging variables
        self.current_velocity = [0.0, 0.0, 0.0]  # [vx, vy, vz]
        self.data_log = []
        self.log_timer = self.create_timer(0.1, self.log_data_callback)  # 10Hz data logging
        
        # Create directory for data logs if it doesn't exist
        self.log_dir = os.path.expanduser('~/drone_data_logs')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        
        # Create a timestamp for the log file
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_file = os.path.join(self.log_dir, f'drone_data_{timestamp}.xlsx')
    
    def log_data_callback(self):
        """Collect data at regular intervals"""
        current_time = time.time()
        
        # Create data row
        data_row = {
            'timestamp': current_time,
            'state': self.state,
            'position_x': self.current_position[0],
            'position_y': self.current_position[1],
            'position_z': self.current_position[2],
            'velocity_x': self.current_velocity[0],
            'velocity_y': self.current_velocity[1],
            'velocity_z': self.current_velocity[2],
            'yaw': self.current_yaw,
            'distance_front': self.processed_distances[0] if len(self.processed_distances) > 0 else float('inf'),
            'distance_right': self.processed_distances[1] if len(self.processed_distances) > 1 else float('inf'),
            'distance_back': self.processed_distances[2] if len(self.processed_distances) > 2 else float('inf'),
            'distance_left': self.processed_distances[3] if len(self.processed_distances) > 3 else float('inf'),
        }
        
        # Append to the data log
        self.data_log.append(data_row)

        # Periodically print the size of the data log
        if len(self.data_log) % 10 == 0:  # Every 10 entries
            self.get_logger().info(f"Data log size: {len(self.data_log)} entries")

    def save_data_to_excel(self):
        """Save collected data to Excel file"""
        self.get_logger().info(f"Attempting to save {len(self.data_log)} data entries")
    
        """Save collected data to Excel file"""
        if len(self.data_log) == 0:
            self.get_logger().warn("No data to save")
            return
            
        try:
            # Convert to DataFrame
            df = pd.DataFrame(self.data_log)
            
            # Add human-readable timestamps
            start_time = df['timestamp'].iloc[0]
            df['elapsed_time'] = df['timestamp'] - start_time
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
            
            # Convert radians to degrees for better readability
            df['yaw_degrees'] = df['yaw'].apply(lambda x: math.degrees(x))
            
            # Reorder columns for better readability
            columns_order = [
                'datetime', 'elapsed_time', 'state',
                'position_x', 'position_y', 'position_z',
                'velocity_x', 'velocity_y', 'velocity_z',
                'yaw', 'yaw_degrees',
                'distance_front', 'distance_right', 'distance_back', 'distance_left'
            ]
            df = df[columns_order]
            
            # Save to Excel
            df.to_excel(self.log_file, index=False, engine='openpyxl')
            self.get_logger().info(f"Data saved to {self.log_file}")
        except Exception as e:
            self.get_logger().error(f"Error saving data to Excel: {e}")

    def teraranger_callback(self, msg):
            self.processed_distances = []
            # Process each distance value            
            for i, distance in enumerate(msg.data):
                direction = self.directions[i]
                
                # Handle special cases
                if distance == float('-inf'):
                    # Object too close - set to 0
                    self.processed_distances.append(0.0)
                    self.get_logger().debug(f"{direction}: Below minimum range (-inf), setting to 0.0m")
                elif distance == float('inf'):
                    # Object too far - set to max distance
                    self.processed_distances.append(self.lidar_max_distance)
                    self.get_logger().debug(f"{direction}: Above maximum range (inf), setting to {self.lidar_max_distance}m")
                elif distance != distance:  # Check for NaN
                    # Invalid reading - set to 0
                    self.processed_distances.append(0.0)
                    self.get_logger().debug(f"{direction}: Invalid reading (nan), setting to 0.0m")
                else:
                    # Normal reading
                    self.processed_distances.append(distance)
                    self.get_logger().debug(f"{direction}: Valid reading: {distance:.3f}m")
            
            # Log the processed distances
            self.get_logger().info(
                f"Processed distances - Front: {self.processed_distances[0]:.2f}m, "
                f"Right: {self.processed_distances[1]:.2f}m, "
                f"Back: {self.processed_distances[2]:.2f}m, "
                f"Left: {self.processed_distances[3]:.2f}m"
            )
    
    def get_distance_in_direction(self, direction):
        """Get the distance from the LIDAR in a specific direction."""
        direction_index = {
            "forward": 0,  # front
            "right": 1,    # right
            "backward": 2, # back
            "left": 3      # left
        }
        
        idx = direction_index.get(direction, 0)
        return self.processed_distances[idx] if len(self.processed_distances) > idx else self.lidar_max_distance
    
    def calculate_velocity_with_obstacle_avoidance(self, direction, desired_speed=1.0):
        """
        Calculate velocity components with obstacle avoidance based on LIDAR readings.
        Returns velocity scaled by PID controller for collision avoidance.
        """
        # Get raw velocity components based on direction
        vx, vy = self.calculate_velocity(direction, speed=desired_speed)
        
        # Get the distance in the current movement direction
        distance = self.get_distance_in_direction(direction)
        
        # Calculate velocity scale factor using PID controller
        scale_factor = self.calculate_velocity_scale(direction, distance)
        
        # Apply scale factor to velocity components
        adjusted_vx = vx * scale_factor
        adjusted_vy = vy * scale_factor
        
        self.get_logger().info(
            f"Direction: {direction}, Distance: {distance:.2f}m, "
            f"Scale: {scale_factor:.2f}, "
            f"Velocity adjusted from ({vx:.2f}, {vy:.2f}) to ({adjusted_vx:.2f}, {adjusted_vy:.2f})"
        )
        
        return adjusted_vx, adjusted_vy

    def calculate_velocity_scale(self, direction, distance):
        """
        Calculate velocity scale factor using PID controller based on distance to obstacle.
        Returns a value between 0.0 (stop) and 1.0 (full speed).
        """
        current_time = time.time()
        
        # If distance is above safe_distance, move at full speed
        if distance >= self.safe_distance:
            # Reset PID terms when we're in safe zone
            self.prev_errors[direction] = 0.0
            self.integral_terms[direction] = 0.0
            self.last_time[direction] = current_time
            return 1.0
            
        # If distance is below min_distance, stop
        if distance <= self.min_distance:
            return 0.0
            
        # Calculate error: how far we are from safe_distance
        # As we get closer to min_distance, error grows
        error = (distance - self.min_distance) / (self.safe_distance - self.min_distance)
        
        # Initialize time if first run
        if self.last_time[direction] is None:
            self.last_time[direction] = current_time
            self.prev_errors[direction] = error
            return error  # Return proportional term only on first run
            
        # Calculate time delta
        dt = current_time - self.last_time[direction]
        if dt <= 0:
            dt = 0.05  # Use default if time hasn't advanced
            
        # Calculate PID terms
        # Proportional term
        p_term = error
        
        # Integral term with anti-windup
        self.integral_terms[direction] += error * dt
        self.integral_terms[direction] = max(0.0, min(1.0, self.integral_terms[direction]))  # Clamp
        i_term = self.ki * self.integral_terms[direction]
        
        # Derivative term
        d_term = self.kd * (error - self.prev_errors[direction]) / dt if dt > 0 else 0.0
        
        # Calculate PID output
        pid_output = self.kp * p_term + i_term + d_term
        
        # Clamp output between 0 and 1
        velocity_scale = max(0.0, min(1.0, pid_output))
        
        # Store values for next iteration
        self.prev_errors[direction] = error
        self.last_time[direction] = current_time
        
        return velocity_scale
        
    def color_callback(self, msg):
        try:
            img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            results = self.model(img)

            if not self.frame_height and not self.frame_width:
                self.frame_width = img.shape[1]
                self.frame_height = img.shape[0]
                frame_width = img.shape[1]
                frame_height = img.shape[0]
            else:
                frame_width = img.shape[1]
                frame_height = img.shape[0]
                
            center_region_width = 300  # 300 px width threshold
            center_region_height = 100  # 100 px height threshold

            # Adjust vertical position by moving the box down
            vertical_offset = 150  # Number of pixels to move down

            # Calculate total frame area
            total_frame_area = frame_width * frame_height

            inference_msg = Yolov8Inference()
            inference_msg.header.stamp = self.get_clock().now().to_msg()
            inference_msg.header.frame_id = "inference"

            # Define center bounding box region
            central_x_min = (frame_width - center_region_width) // 2
            central_x_max = (frame_width + center_region_width) // 2
            central_y_min = (frame_height - center_region_height) // 2 + vertical_offset
            central_y_max = (frame_height + center_region_height) // 2 + vertical_offset

            # Draw center bounding box (red rectangle)
            cv2.rectangle(img, (central_x_min, central_y_min), (central_x_max, central_y_max), (0, 0, 255), 2)

            # Variables to track the lowest-right object
            selected_mask = None
            selected_bbox = None
            max_position_sum = float("-inf")
            selected_bbox_area = 0

            for r in results:
                masks = getattr(r, 'masks', None)
                confidences = r.boxes.conf
                boxes = r.boxes.xyxy  # Bounding boxes (xyxy format)
                # if masks is not None:
                #     for mask in masks.data:
                #         # Debug print/visualize
                #         print("\n[DEBUG] Mask detected!")
                #         print(f"Raw mask shape: {mask.shape}")
                        
                #         # Convert and check
                #         mask_np = mask.cpu().numpy().squeeze()
                #         print(f"NumPy shape: {mask_np.shape}")
                        
                #         # Visualize (optional)
                #         cv2.imshow("YOLOv8 Mask", (mask_np * 255).astype(np.uint8))
                #         cv2.waitKey(1)
                            
                if masks is not None and boxes is not None:
                    for mask, confidence, box in zip(masks.data, confidences, boxes):
                        if confidence < 0.7:
                            continue
                        
                        x1, y1, x2, y2 = map(int, box.tolist())
                        bbox_area = (x2 - x1) * (y2 - y1)
                        position_sum = x2 + y2

                        if position_sum > max_position_sum:
                            max_position_sum = position_sum
                            selected_bbox = (x1, y1, x2, y2)
                            selected_bbox_area = bbox_area
                            selected_mask = (mask > 0.5).cpu().numpy().astype('uint8')
                            if self.retake_mask:
                                self.mask_np = selected_mask
                else:
                    print("No valid detections in this frame")
            
            if selected_bbox and selected_mask is not None and selected_mask.any() and self.state == "ALLIGN":
                x_min, y_min, x_max, y_max = selected_bbox
                self.target_acquired = True
                self.last_detection_time = self.get_clock().now()

                # Compute the center point of the selected object
                bbox_center_x = (x_min + x_max) // 2
                bbox_center_y = (y_min + y_max) // 2

                self.mask_np = selected_mask

                # Calculate area ratio
                bbox_area_ratio = (selected_bbox_area / total_frame_area) * 100

                self.get_logger().debug(f"Bounding Box Area Ratio: {bbox_area_ratio:.2f}%")
                self.get_logger().debug(f"Selected Bounding Box: ({x_min}, {y_min}) to ({x_max}, {y_max})")
                self.get_logger().debug(f"Selected Center Point: ({bbox_center_x}, {bbox_center_y})")

                # Draw the selected bounding box (green)
                cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

                # Draw the center point (red circle)
                cv2.circle(img, (bbox_center_x, bbox_center_y), 5, (0, 0, 255), -1)

                # Reset velocities
                self.vx = 0.0
                self.vy = 0.0
                self.vz = 0.0
                h_vx = 0.0
                h_vy = 0.0
                b_vx = 0.0
                b_vy = 0.0

                # Track time of stationary state
                current_time = self.get_clock().now()
                if (abs(self.vx) < 0.01 and abs(self.vy) < 0.01 and abs(self.vz) < 0.01):
                    if not hasattr(self, '_stationary_start_time'):
                        self._stationary_start_time = current_time
                    
                    # Check if stationary for more than 3 seconds
                    if (current_time - self._stationary_start_time).nanoseconds / 1e9 > 6.0:
                        self.get_logger().info("Drone stationary for 3+ seconds. Transitioning to LANDING.")
                        self.change_state("HOVER")
                        return
                else:
                    # Reset stationary time if movement occurs
                    if hasattr(self, '_stationary_start_time'):
                        delattr(self, '_stationary_start_time')

                # Proportional control for vertical movement
                vertical_error = (bbox_center_y - ((central_y_min + central_y_max) // 2))
                vertical_p_gain = 0.005  # Adjust this for smoother vertical movement
                self.vz = vertical_error * vertical_p_gain

                # Clip vertical velocity
                self.vz = max(-1.0, min(1.0, self.vz))

                # Horizontal movement with proportional control and velocity calculation
                if bbox_center_x < central_x_min:
                    # Move left with proportional speed
                    horizontal_error = central_x_min - bbox_center_x
                    horizontal_p_gain = 0.005  # Adjust for smoother horizontal movement
                    speed = min(1.0, horizontal_error * horizontal_p_gain)
                    h_vx, h_vy = self.calculate_velocity_with_obstacle_avoidance("left", speed=speed)
                    # self.vx, self.vy = vx, vy
                elif bbox_center_x > central_x_max:
                    # Move right with proportional speed
                    horizontal_error = bbox_center_x - central_x_max
                    horizontal_p_gain = 0.005  # Adjust for smoother horizontal movement
                    speed = min(1.0, horizontal_error * horizontal_p_gain)
                    h_vx, h_vy = self.calculate_velocity_with_obstacle_avoidance("right", speed=speed)
                    # self.vx, self.vy = vx, vy

                # Check if area is too large - move backward
                if bbox_area_ratio > 80:
                    b_vx, b_vy = self.calculate_velocity_with_obstacle_avoidance("backward", speed=0.5)
                    # self.vx, self.vy = vx, vy
                
                # if (h_vx and h_vy and b_vx and b_vy) != 0:
                # Calculate resultant velocity using vector addition
                self.vx = h_vx + b_vx
                self.vy = h_vy + b_vy

                # Optional: Normalize velocity if total speed exceeds 1.0
                total_speed = np.sqrt(self.vx**2 + self.vy**2)
                if total_speed > 1.0:
                    self.vx /= total_speed
                    self.vy /= total_speed
                
                self.publish_trajectory_setpoint(vx=self.vx, vy=self.vy, vz=self.vz)
            
            # Image publishing code
            annotated_frame = results[0].plot()
            final_image = cv2.addWeighted(annotated_frame, 0.8, img, 0.2, 0)

            img_msg = self.bridge.cv2_to_imgmsg(final_image, encoding="bgr8")
            self.img_pub.publish(img_msg)
            self.inference_pub.publish(inference_msg)

        except Exception as e:
            self.get_logger().error(f"Error processing image: {str(e)}")

    def calculate_drop_point_position(self, cx, cy, cz, drop_X, drop_Y, drop_Z):
        """
        Calculate the drop point position in odometry coordinates 
        based on the camera's 3D point cloud measurements.
        
        Args:
        cx (float): Current x position in odometry frame
        cy (float): Current y position in odometry frame
        cz (float): Current z position in odometry frame
        drop_X (float): X coordinate of drop point in camera frame
        drop_Y (float): Y coordinate of drop point in camera frame
        drop_Z (float): Z coordinate of drop point in camera frame
        
        Returns:
        tuple: Final x, y, z coordinates in odometry frame
        """
        # Yaw angle for rotation
        yaw = self.current_yaw  # Current yaw angle in radians

        # Rotation matrix for yaw (rotation around z-axis)
        rotation_matrix = np.array([
            [math.cos(yaw), -math.sin(yaw), 0],
            [math.sin(yaw), math.cos(yaw), 0],
            [0, 0, 1]
        ])

        # Camera to odometry transformation
        # This assumes the camera is rigidly mounted to the drone
        # You may need to adjust these based on your specific drone setup
        camera_offset = np.array([0, 0, 0])  # Camera offset from drone center
        
        # Transform drop point from camera frame to drone's local frame
        camera_frame_point = np.array([drop_X, drop_Y, drop_Z])
        
        # Rotate the point based on drone's current yaw
        rotated_point = rotation_matrix @ camera_frame_point
        
        # Calculate final position by adding current position and rotated point
        final_x = cx + rotated_point[0]
        final_y = cy + rotated_point[1]
        final_z = cz + rotated_point[2]
        
        # Calculate distance from camera
        distance = np.linalg.norm(camera_frame_point)
        
        # Log the movement details
        self.get_logger().info(
            f"Drop Point Calculation: "
            f"Initial Position ({cx:.2f}, {cy:.2f}, {cz:.2f}) | "
            f"Yaw: {math.degrees(yaw):.1f}° | "
            f"Camera Frame Point: ({drop_X:.2f}, {drop_Y:.2f}, {drop_Z:.2f}) | "
            f"Distance from Camera: {distance:.2f}m | "
            f"Final Position ({final_x:.2f}, {final_y:.2f}, {final_z:.2f})"
        )
        
        return final_x, final_y, final_z, distance

    def publish_drop_point(self, x, y, z):
        """
        Publish drop point coordinates in odometry frame
        
        Args:
        x (float): X coordinate in odometry frame
        y (float): Y coordinate in odometry frame
        z (float): Z coordinate in odometry frame
        """
        # Create Point message
        point_msg = Point()
        point_msg.x = x
        point_msg.y = y
        point_msg.z = z
        
        # Publish the point
        self.drop_point_pub.publish(point_msg)

    def depth_callback(self, msg):
            depth_points = list(pc2.read_points(msg, field_names=("x", "y", "z"), skip_nans=True))
            
            if self.state == "HOVER" and self.target_acquired:
                depth_data = np.array([point[2] for point in depth_points])
                image_width = self.frame_width
                image_height = self.frame_height

                # Reshape depth data into an image format
                depth_image = depth_data.reshape((image_height, image_width))

                # Normalize the depth image for visualization
                depth_image_normalized = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))

                # Resize mask to match depth image dimensions
                mask_resized = cv2.resize(self.mask_np, (image_width, image_height))

                # Apply the mask - Keep only solar panel depth values
                fused_depth_image = np.where(mask_resized > 0, depth_image, np.nan)

                # Calculate white pixel area for fused depth image
                white_area_fused = np.sum(~np.isnan(fused_depth_image))
                print(f"White pixel area in fused depth image: {white_area_fused}")

                # Normalize masked depth image for visualization
                min_fused = np.nanmin(fused_depth_image)
                max_fused = np.nanmax(fused_depth_image)
                fused_depth_image_normalized = (fused_depth_image - min_fused) / (max_fused - min_fused)

                # Calculate white pixel area for normalized fused depth image
                white_area_normalized = np.sum(~np.isnan(fused_depth_image_normalized))
                print(f"White pixel area in normalized fused depth image: {white_area_normalized}")

                # Camera intrinsic parameters
                fx = 554.2547
                fy = 554.2547  
                cx = 320.5  
                cy = 240.5  

                # Extract 3D coordinates from the fused depth data
                points_3d = []
                pixel_coords = []
                for v in range(image_height):
                    for u in range(image_width):
                        if mask_resized[v, u] > 0:  # Only use masked pixels
                            Z = fused_depth_image[v, u]  # Depth value
                            if np.isnan(Z):  # Skip NaN values
                                continue
                            X = (u - cx) * Z / fx
                            Y = (v - cy) * Z / fy
                            points_3d.append([X, Y, Z])
                            pixel_coords.append((u, v))  # Store corresponding image coordinates

                points_3d = np.array(points_3d)
                
                if points_3d.shape[0] > 3:  # Ensure enough points for plane fitting
                    # Convert to Open3D point cloud
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(points_3d)

                    # Fit a plane using RANSAC
                    plane_model, inliers = pcd.segment_plane(distance_threshold=0.01,
                                                            ransac_n=3,
                                                            num_iterations=1000)
                    a, b, c, d = plane_model  # Plane equation: ax + by + cz + d = 0

                    # Compute the tilt angle (angle between plane normal and Z-axis)
                    tilt_angle_rad = np.arccos(abs(c) / np.sqrt(a**2 + b**2 + c**2))

                    if tilt_angle_rad > (math.pi/4):
                        tilt_angle_rad = (math.pi/2) - tilt_angle_rad
                        
                    tilt_angle_deg = np.degrees(tilt_angle_rad)

                    angle_msg = Float64(data=float(tilt_angle_rad))

                    # Identify the bottom-right corner
                    bottom_right_idx = np.argmax(points_3d[:, 0] + points_3d[:, 2])  # Rightmost + lowest point
                    X_br, Y_br, Z_br = points_3d[bottom_right_idx]

                    # Compute the dropping point in 3D (move 8cm up & left)
                    drop_X = X_br - 0.08  # Move 8cm left
                    drop_Y = Y_br + 0.08  # Move 8cm up
                    drop_Z = Z_br  # Same depth

                    # Calculate drop point position in odometry frame
                    # if hasattr(self, 'current_x') and hasattr(self, 'current_y') and hasattr(self, 'current_z'):
                    drone_pos = self.current_position
                    current_x, current_y, current_z = drone_pos

                    drop_odometry_x, drop_odometry_y, drop_odometry_z, drop_distance = self.calculate_drop_point_position(
                        current_x, current_y, current_z,
                        drop_X, drop_Y, drop_Z
                    )
                    
                    diff_x = drop_odometry_x - current_x
                    diff_y = drop_odometry_y - current_y
                    diff_z = drop_odometry_z - current_z

                    if abs(diff_x)>3 and abs(diff_y)>3:
                        self.retake_mask = True
                        return
                    else:
                        self.retake_mask = False

                    # Print all information
                    self.get_logger().info("\n--- Deployment Position Calculation ---")
                    self.get_logger().info(f"Current Drone Position (NED): X={current_x:.3f}m, Y={current_y:.3f}m, Z={current_z:.3f}m")
                    self.get_logger().info(f"Deployment Position (NED):    X={drop_odometry_x:.3f}m, Y={drop_odometry_y:.3f}m, Z={drop_odometry_z:.3f}m")
                    self.get_logger().info(f"Difference (Deployment - Current): ΔX={diff_x:.3f}m, ΔY={diff_y:.3f}m, ΔZ={diff_z:.3f}m")
                    self.get_logger().info("------------------------------------")

                    # Prompt for user confirmation
                    # user_confirmation = input("Do you want to proceed with publishing? (y/n/r): ").strip().lower()

                    # if user_confirmation == 'y':
                    # Proceed with publishing
                                        # Publish inclination angle
                    self.inclination_pub.publish(angle_msg)
                    self.publish_drop_point(drop_odometry_x, drop_odometry_y, drop_odometry_z)

                    # Project to 2D using intrinsic matrix
                    drop_u = int((drop_X * fx / drop_Z) + cx)
                    drop_v = int((drop_Y * fy / drop_Z) + cy)

                    # Plot full depth intensity map
                    plt.figure(figsize=(10, 6))
                    plt.imshow(depth_image_normalized, cmap='jet', interpolation='nearest')
                    plt.colorbar(label='Depth Intensity')
                    plt.scatter(drop_u, drop_v, color='red', s=100, label='Dropping Point')
                    plt.text(10, 30, f"Slant Angle: {tilt_angle_deg:.2f}°", fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.5))
                    plt.legend()
                    plt.title("Full Depth Intensity Map with Dropping Point")
                    plt.show()

                    # Plot fused depth intensity map (solar panel only)
                    plt.figure(figsize=(10, 6))
                    panel_only = np.copy(fused_depth_image_normalized)
                    panel_only[np.isnan(panel_only)] = 0  # Set background to 0 for better visibility

                    plt.imshow(panel_only, cmap='gray', interpolation='nearest')
                    plt.colorbar(label='Depth Intensity (Solar Panel Only)')
                    plt.scatter(drop_u, drop_v, color='red', s=100, label='Dropping Point')
                    plt.text(10, 30, f"Slant Angle: {tilt_angle_deg:.2f}°", fontsize=12, color='white', bbox=dict(facecolor='black', alpha=0.5))
                    plt.legend()
                    plt.title("Isolated Solar Panel Depth Map with Dropping Point")
                    plt.show()
                    
                    self.save_data_to_excel()
                    self.change_state("RETURN_TO_LAUNCH")

                    # elif user_confirmation == 'n':
                    #     print("Process cancelled by user.")
                    #     self.change_state("RETURN_TO_LAUNCH")
                    # elif user_confirmation == 'r':
                    #     self.retake_mask = True
                    # else:
                    #     print("Invalid input by user.")
                    #     self.change_state("RETURN_TO_LAUNCH")

    def calculate_velocity(self, direction, speed=1.0):
        """
        Calculate velocity components based on initial yaw after takeoff.
        
        Args:
            direction (str): Movement direction ('forward', 'backward', 'right', 'left')
            speed (float): Speed multiplier between 0 and 1.0
        
        Returns:
            tuple: Scaled vx and vy velocities
        """
        # Ensure speed is between 0 and 1.0
        speed = max(0.0, min(1.0, speed))
        
        # Use initial_yaw if set, otherwise use current_yaw (shouldn't happen)
        yaw = self.current_yaw #self.initial_yaw if hasattr(self, 'initial_yaw') and self.initial_yaw is not None else self.current_yaw
        
        # Base direction vectors (at 0° yaw)
        base_vectors = {
            "forward": (1, 0),
            "backward": (-1, 0),
            "right": (0, 1),
            "left": (0, -1)
        }
        
        # Get the base vector
        base_vx, base_vy = base_vectors.get(direction, (0, 0))
        
        # Apply rotation
        vx = base_vx * math.cos(yaw) - base_vy * math.sin(yaw)
        vy = base_vx * math.sin(yaw) + base_vy * math.cos(yaw)
        
        # Find max absolute value to scale
        max_val = max(abs(vx), abs(vy))
        
        # Scale to [-1.0, 1.0] range if needed
        if max_val > 1.0:
            vx /= max_val
            vy /= max_val
        
        # Apply speed multiplier
        vx *= speed
        vy *= speed
        
        self.get_logger().info(
            f"Moving {direction} relative to initial yaw: {(yaw):.1f} rad, {math.degrees(yaw):.1f}° | "
            f"Speed: {speed:.2f} | Velocities - X: {vx:.2f}, Y: {vy:.2f}"
        )
        
        return vx, vy

    def odometry_callback(self, msg):
        """Callback to update the current position."""
        self.current_position = [
            msg.position[0],  # X in NED
            msg.position[1],  # Y in NED
            msg.position[2],  # Z in NED
        ]
        self.angular_velocity = [
            msg.angular_velocity[0],  # Roll angular velocity
            msg.angular_velocity[1],  # Pitch angular velocity
            msg.angular_velocity[2],  # Yaw angular velocity
        ]
    
        # Extract quaternion and convert to yaw
        q = [msg.q[0], msg.q[1], msg.q[2], msg.q[3]]
        roll, pitch, yaw = self.quaternion_to_euler(q)
        self.current_yaw = yaw

    def quaternion_to_euler(self, q):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        # Roll (x-axis rotation)
        sinr_cosp = 2 * (q[0] * q[1] + q[2] * q[3])
        cosr_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (q[0] * q[2] - q[3] * q[1])
        if abs(sinp) >= 1:
            pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
        cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def vehicle_status_callback(self, msg):
        """Callback to update the current mode."""
        self.current_mode = msg.nav_state
        if msg.arming_state == 2:
            self.armed = True
        else:
            self.armed = False

    def publish_offboard_control_mode(self):
        """Publish OffboardControlMode message."""
        try:
            if self.offboard_mode == "VELOCITY":
                offboard_msg = OffboardControlMode()
                offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                offboard_msg.position = False
                offboard_msg.velocity = True  # Enable velocity control
                offboard_msg.acceleration = False
                offboard_msg.attitude = False
                offboard_msg.body_rate = False
                self.offboard_control_mode_publisher.publish(offboard_msg)
            elif self.offboard_mode == "POSITION":
                offboard_msg = OffboardControlMode()
                offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                offboard_msg.position = True
                offboard_msg.velocity = False  # Enable velocity control
                offboard_msg.acceleration = False
                offboard_msg.attitude = False
                offboard_msg.body_rate = False
                self.offboard_control_mode_publisher.publish(offboard_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing offboard control mode: {str(e)}")

    def publish_trajectory_setpoint(self, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
        """Publish a trajectory setpoint in velocity mode."""
        try:
            # Store current velocity for logging
            self.current_velocity = [vx, vy, vz]
            if self.offboard_mode == "VELOCITY":
                trajectory_msg = TrajectorySetpoint()
                trajectory_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                trajectory_msg.velocity[0] = vx
                trajectory_msg.velocity[1] = vy
                trajectory_msg.velocity[2] = vz
                trajectory_msg.position[0] = float('nan')
                trajectory_msg.position[1] = float('nan')
                trajectory_msg.position[2] = float('nan')
                trajectory_msg.acceleration[0] = float('nan')
                trajectory_msg.acceleration[1] = float('nan')
                trajectory_msg.acceleration[2] = float('nan')
                trajectory_msg.yaw = float('nan')
                self.trajectory_setpoint_publisher.publish(trajectory_msg)
            elif self.offboard_mode == "POSITION":
                trajectory_msg = TrajectorySetpoint()
                trajectory_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
                trajectory_msg.velocity[0] = float('nan')
                trajectory_msg.velocity[1] = float('nan')
                trajectory_msg.velocity[2] = float('nan')
                trajectory_msg.position[0] = vx
                trajectory_msg.position[1] = vy
                trajectory_msg.position[2] = vz
                trajectory_msg.acceleration[0] = float('nan')
                trajectory_msg.acceleration[1] = float('nan')
                trajectory_msg.acceleration[2] = float('nan')
                trajectory_msg.yaw = float('nan')
                self.trajectory_setpoint_publisher.publish(trajectory_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing trajectory setpoint: {str(e)}")

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0, param3=0.0):
        """Publish a VehicleCommand."""
        try:
            msg = VehicleCommand()
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            msg.param1 = param1
            msg.param2 = param2
            msg.param3 = param3
            msg.command = command
            msg.target_system = 1
            msg.target_component = 1
            msg.source_system = 1
            msg.source_component = 1
            msg.from_external = True
            self.vehicle_command_publisher.publish(msg)
            self.get_logger().info(f"Published VehicleCommand: command={command}, param1={param1}, param2={param2}, param3={param3}")
        except Exception as e:
            self.get_logger().error(f"Error publishing vehicle command: {str(e)}")

    def arm_drone(self):
        """Command the drone to arm."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

    def publish_takeoff(self):
        """Send takeoff command to PX4."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 2.0)  # original take off altitude
        self.get_logger().info(f"Sending takeoff command.")

    def time_since_state_change(self):
        """Get time since last state change in seconds."""
        return (self.get_clock().now() - self.state_change_time).nanoseconds / 1e9

    def change_state(self, new_state):
        """Change state with proper logging and timing."""
        self.get_logger().info(f"State transition: {self.state} -> {new_state}")
        self.state = new_state
        self.state_change_time = self.get_clock().now()

    def timer_callback(self):
        """Main loop that implements the state machine."""
        try:
            if self.state == "ARMING":
                if not self.armed:
                    if self.current_mode != 4:
                        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 3.0)
                    self.arm_drone()
                    self.publish_takeoff()
                else:
                    if self.current_mode != 4:
                        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 3.0)
                    self.get_logger().info("Drone armed and taking off")
                    
                    if self.time_since_state_change() >= 10.0:  # Wait for 15 seconds before switching to TAKEOFF
                        self.change_state("TAKEOFF")

            elif self.state == "TAKEOFF":
                if self.time_since_state_change() >= 2.0:  # Wait for 10 seconds before switching to ALLIGN
                    self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
                    # self.offboard_mode = "POSITION"
                    # self.publish_trajectory_setpoint(vx=-0.485, vy=1.620, vz=(-7.554+5))
                    self.change_state("ALLIGN")
                    self.target_acquired = False
                    self.last_detection_time = None
                    # self.initial_yaw = True  # Will store the yaw at takeoff completion
                
            elif self.state == "ALLIGN":
                # If no object detected, move upwards continuously
                if not self.target_acquired or (self.last_detection_time and 
                (self.get_clock().now() - self.last_detection_time).nanoseconds / 1e9 > 1.0):
                    # No target or haven't seen one recently (more than 1 second)
                    self.vz = -0.5  # Move upwards
                    self.vx = 0.0
                    self.vy = 0.0
                    self.publish_trajectory_setpoint(vx=self.vx, vy=self.vy, vz=self.vz)
                    self.get_logger().debug("Moving upward to search for object")
                    self.last_movement_time = self.get_clock().now()  # Update last movement time
                
                # If any velocity is non-zero, we're making adjustments
                if abs(self.vx) > 0.01 or abs(self.vy) > 0.01 or abs(self.vz) > 0.01:
                    self.last_movement_time = self.get_clock().now()  # Update last movement time
                
                # Check if we've been stable (no movement) for 5 seconds
                # if self.target_acquired and ((self.get_clock().now() - self.last_movement_time).nanoseconds / 1e9 >= 2.0):
                #     self.get_logger().info("No adjustments for 2 seconds, transitioning to HOVER")
                #     self.change_state("HOVER")
                
                # Check if we've been in ALLIGN state for more than 20 seconds without finding an object
                if self.time_since_state_change() >= 30.0 and not self.target_acquired:
                    self.get_logger().info("No object detected for 20 seconds, transitioning to LANDING")
                    self.change_state("LANDING")

            elif self.state == "HOVER":
                self.publish_trajectory_setpoint(vx=0.0, vy=0.0, vz=0.0)  # Stay in place
                if self.time_since_state_change() >= 15.0:  # Hover for 5 seconds
                    self.change_state("LANDING")

            elif self.state == "LANDING":
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 6.0)
                self.get_logger().info("Landing...")
                if self.time_since_state_change() >= 5.0:  # Wait for 5 seconds before declaring landed
                    self.change_state("LANDED")
            
            elif self.state == "RETURN_TO_LAUNCH":
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 5.0)
                self.get_logger().info("Returning...")

                # # Add periodic saving during hover
                # if self.time_since_state_change() >= 2.5:  # Save halfway through hover
                #     self.get_logger().info("Saving data during hover")
                #     self.save_data_to_excel()

                if self.time_since_state_change() >= 5.0:  # Wait for 5 seconds before declaring landed
                    self.change_state("LANDED")

            elif self.state == "LANDED":
                self.get_logger().info("Drone has landed successfully.")
                # Do nothing else, we're done

        except Exception as e:
            self.get_logger().error(f"Error in timer callback: {str(e)}")
            # Safety fallback - try to land if something goes wrong
            if self.state != "LANDING" and self.state != "LANDED":
                self.get_logger().error("Error detected, attempting emergency landing")
                self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 6.0)
                self.change_state("LANDING")

def main(args=None):
    rclpy.init(args=args)
    node = ZDCalNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard Interrupt detected. Shutting down...")
    except Exception as e:
        node.get_logger().error(f"Unexpected error: {str(e)}")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()

# import rclpy
# from rclpy.node import Node
# from px4_msgs.msg import VehicleCommand, OffboardControlMode, TrajectorySetpoint, VehicleOdometry, VehicleStatus
# from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
# import time
# import math
# import numpy as np

# from sensor_msgs.msg import Image
# from yolov8_msgs.msg import Yolov8Inference
# import cv2
# from cv_bridge import CvBridge
# from ultralytics import YOLO

# class ZDCalNode(Node):
#     def __init__(self):
#         super().__init__('realsense_subscriber_node')

#         # Initialize OpenCV bridge
#         self.bridge = CvBridge()

#         # Load YOLOv8 model
#         self.model = YOLO('/home/james/realsense_ws/src/realsense/model/solar_panel.pt')

#         # Define QoS profile for PX4 compatibility
#         qos_profile = QoSProfile(
#             reliability=QoSReliabilityPolicy.BEST_EFFORT,
#             history=QoSHistoryPolicy.KEEP_LAST,
#             depth=1
#         )

#         # Publishers
#         self.inference_pub = self.create_publisher(Yolov8Inference, '/yolov8/detections', 10)
#         self.img_pub = self.create_publisher(Image, "/yolov8/annotated_image", 10)

#         self.vehicle_command_publisher = self.create_publisher(VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
#         self.offboard_control_mode_publisher = self.create_publisher(OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
#         self.trajectory_setpoint_publisher = self.create_publisher(TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        
#         # ROS2 Subscribers
#         self.color_sub = self.create_subscription(
#             Image, 
#             '/iris/camera/image_raw', # '/camera/realsense_node/color/image_raw', 
#             self.color_callback, 
#             10)
        
#         self.create_subscription(
#             VehicleOdometry,
#             '/fmu/out/vehicle_odometry',
#             self.odometry_callback,
#             qos_profile
#         )

#         self.create_subscription(
#             VehicleStatus,
#             '/fmu/out/vehicle_status_v1', #'/fmu/out/vehicle_status',
#             self.vehicle_status_callback,
#             qos_profile
#         )

#         # State variables
#         self.current_position = [0.0, 0.0, 0.0]  # Current position in NED
#         self.angular_velocity = [0.0, 0.0, 0.0]
#         self.current_mode = None  # Current flight mode
#         self.armed = False  # Armed state
#         self.state = "ARMING"  # State machine state
#         self.takeoff_altitude = -8.0  # Takeoff altitude in NED (8 meters up)
        
#         # Timing variables
#         self.state_change_time = self.get_clock().now()
#         self.yaw_angle = 0  # Instantaneous target yaw
#         self.running = True

#         # Movement control variables
#         self.vx = 0.0
#         self.vy = 0.0
#         self.vz = 0.0
#         self.target_acquired = False
#         self.last_detection_time = None  # Track when an object was last detected
#         self.initial_yaw = None  # Will store the yaw at takeoff completion

#         # P controller variables for X, Y and Z
#         self.x_kp = 2.0  # Proportional gain for x controller
#         self.y_kp = 2.0  # Proportional gain for y controller
#         self.z_kp = 2.0  # Proportional gain for z controller
#         self.max_velocity = 1.0  # Maximum velocity magnitude
#         self.initial_x_position = None  # To store the initial x position for maintenance
#         self.initial_y_position = None  # To store the initial y position for maintenance
#         self.initial_z_position = None  # To store the initial z position for maintenance
#         self.maintain_x = False  # Flag to indicate whether to maintain x position
#         self.maintain_y = False  # Flag to indicate whether to maintain y position
#         self.maintain_z = False  # Flag to indicate whether to maintain z position
#         self.moving_in_x = False  # Flag to indicate explicit x movement
#         self.moving_in_y = False  # Flag to indicate explicit y movement
#         self.moving_in_z = False  # Flag to indicate explicit z movement

#         # Timer to control the state machine
#         self.last_velocity_print_time = self.get_clock().now()
#         self.timer = self.create_timer(0.05, self.timer_callback)  # 20Hz
#         self.publish_offboard_timer = self.create_timer(0.01, self.publish_offboard_control_mode)  # 100Hz
#         self.last_movement_time = None
    
#     def color_callback(self, msg):
#         try:
#             # Convert ROS image message to OpenCV format
#             img = self.bridge.imgmsg_to_cv2(msg, "bgr8")

#             # Run YOLOv8 inference
#             results = self.model(img)

#             frame_width = img.shape[1]
#             frame_height = img.shape[0]
            
#             # Define center region thresholds
#             center_region_width = 300  # 300 px width threshold
#             center_region_height = 100  # 100 px height threshold

#             inference_msg = Yolov8Inference()
#             inference_msg.header.stamp = self.get_clock().now().to_msg()
#             inference_msg.header.frame_id = "inference"

#             # Define center bounding box region
#             central_x_min = (frame_width - center_region_width) // 2
#             central_x_max = (frame_width + center_region_width) // 2
#             central_y_min = (frame_height - center_region_height) // 2
#             central_y_max = (frame_height + center_region_height) // 2

#             # Draw center bounding box (red rectangle)
#             cv2.rectangle(img, (central_x_min, central_y_min), (central_x_max, central_y_max), (0, 0, 255), 2)

#             # Variables to track the detected object
#             selected_bbox = None
#             max_confidence = 0.0

#             for r in results:
#                 boxes = r.boxes.xyxy  # Bounding boxes (xyxy format)
#                 confidences = r.boxes.conf

#                 if boxes is not None:
#                     for confidence, box in zip(confidences, boxes):
#                         if confidence < 0.5:  # Confidence threshold
#                             continue

#                         # Track the highest confidence detection
#                         if confidence > max_confidence:
#                             max_confidence = confidence
#                             selected_bbox = box.tolist()

#             if selected_bbox and self.state == "TRACKING":
#                 x_min, y_min, x_max, y_max = map(int, selected_bbox)
#                 self.target_acquired = True
#                 self.last_detection_time = self.get_clock().now()

#                 # Compute the center point of the selected object
#                 bbox_center_x = (x_min + x_max) // 2
#                 bbox_center_y = (y_min + y_max) // 2
                
#                 # Calculate the bbox size to determine distance
#                 bbox_width = x_max - x_min
#                 bbox_height = y_max - y_min
#                 bbox_area = bbox_width * bbox_height
#                 frame_area = frame_width * frame_height
#                 area_ratio = bbox_area / frame_area

#                 # Draw the selected bounding box (green)
#                 cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
#                 # Draw the center point (red circle)
#                 cv2.circle(img, (bbox_center_x, bbox_center_y), 5, (0, 0, 255), -1)

#                 # Calculate errors
#                 frame_center_x = frame_width // 2
#                 frame_center_y = frame_height // 2
#                 y_error = bbox_center_x - frame_center_x  # Horizontal error
#                 z_error = bbox_center_y - frame_center_y  # Vertical error

#                 # Calculate movement commands based on errors
#                 if abs(y_error) > 50:  # Only move if error is significant
#                     # Determine direction based on error sign
#                     direction = "right" if y_error > 0 else "left"
#                     # Calculate velocity using yaw-based method from first code
#                     vx, vy = self.calculate_velocity(direction, speed=0.5)
#                     self.vy = vy  # Store for publishing
#                 else:
#                     self.vy = 0.0

#                 # Z-axis movement (altitude)
#                 if abs(z_error) > 50:
#                     self.vz = -0.3 if z_error > 0 else 0.3  # Move up/down based on error
#                 else:
#                     self.vz = 0.0

#                 # X-axis movement (distance)
#                 target_area_ratio = 0.3  # Target size of object in frame
#                 if area_ratio > target_area_ratio * 1.2:  # Too close
#                     vx, _ = self.calculate_velocity("backward", speed=0.3)
#                     self.vx = vx
#                 elif area_ratio < target_area_ratio * 0.8:  # Too far
#                     vx, _ = self.calculate_velocity("forward", speed=0.3)
#                     self.vx = vx
#                 else:
#                     self.vx = 0.0

#                 # Publish the movement command
#                 self.publish_trajectory_setpoint(vx=self.vx, vy=self.vy, vz=self.vz)

#             # Publish annotated image
#             annotated_frame = results[0].plot()
#             final_image = cv2.addWeighted(annotated_frame, 0.8, img, 0.2, 0)
#             img_msg = self.bridge.cv2_to_imgmsg(final_image, encoding="bgr8")
#             self.img_pub.publish(img_msg)
#             self.inference_pub.publish(inference_msg)

#         except Exception as e:
#             self.get_logger().error(f"Error processing image: {str(e)}")

#     def calculate_velocity(self, direction, speed=1.0):
#         """
#         Calculate velocity components based on initial yaw after takeoff.
#         """
#         # Use initial_yaw if set, otherwise use current_yaw (shouldn't happen)
#         yaw = self.initial_yaw if self.initial_yaw is not None else self.current_yaw

#         # direction = direction.lower()
        
#         # Base direction vectors (at 0° yaw)
#         base_vectors = {
#             "forward": (1, 0),
#             "backward": (-1, 0),
#             "right": (0, 1),
#             "left": (0, -1)
#         }
        
#         # Get the base vector
#         base_vx, base_vy = base_vectors.get(direction, (0, 0))
        
#         # Apply rotation
#         vx = base_vx * math.cos(yaw) - base_vy * math.sin(yaw)
#         vy = base_vx * math.sin(yaw) + base_vy * math.cos(yaw)
        
#         # Round to avoid floating-point precision issues (optional)
#         # vx = round(vx, 6)
#         # vy = round(vy, 6)
            
#         self.get_logger().info(
#             f"Moving {direction} relative to initial yaw: {(yaw):.1f} rad, {math.degrees(yaw):.1f}° | "
#             f"Velocities - X: {vx:.2f}, Y: {vy:.2f}"
#         )
            
#         return vx, vy

#     def odometry_callback(self, msg):
#         """Callback to update the current position."""
#         self.current_position = [
#             msg.position[0],  # X in NED
#             msg.position[1],  # Y in NED
#             msg.position[2],  # Z in NED
#         ]
#         self.angular_velocity = [
#             msg.angular_velocity[0],  # Roll angular velocity
#             msg.angular_velocity[1],  # Pitch angular velocity
#             msg.angular_velocity[2],  # Yaw angular velocity
#         ]
        
#         # Extract quaternion and convert to yaw
#         q = [msg.q[0], msg.q[1], msg.q[2], msg.q[3]]
#         roll, pitch, yaw = self.quaternion_to_euler(q)
#         self.current_yaw = yaw

#         # Only set initial_yaw once after takeoff if it hasn't been set
#         if self.state == "ALLIGN" and self.initial_yaw is None:
#             self.initial_yaw = yaw
#             self.get_logger().info(f"Initial yaw set to: {math.degrees(self.initial_yaw):.1f}°")

#     def quaternion_to_euler(self, q):
#         """Convert quaternion to Euler angles (roll, pitch, yaw)"""
#         # Roll (x-axis rotation)
#         sinr_cosp = 2 * (q[0] * q[1] + q[2] * q[3])
#         cosr_cosp = 1 - 2 * (q[1] * q[1] + q[2] * q[2])
#         roll = np.arctan2(sinr_cosp, cosr_cosp)
 
#         # Pitch (y-axis rotation)
#         sinp = 2 * (q[0] * q[2] - q[3] * q[1])
#         if abs(sinp) >= 1:
#             pitch = np.copysign(np.pi / 2, sinp)  # Use 90 degrees if out of range
#         else:
#             pitch = np.arcsin(sinp)
 
#         # Yaw (z-axis rotation)
#         siny_cosp = 2 * (q[0] * q[3] + q[1] * q[2])
#         cosy_cosp = 1 - 2 * (q[2] * q[2] + q[3] * q[3])
#         yaw = np.arctan2(siny_cosp, cosy_cosp)
 
#         return roll, pitch, yaw

#     def vehicle_status_callback(self, msg):
#         """Callback to update the current mode."""
#         self.current_mode = msg.nav_state
#         if msg.arming_state == 2:
#             self.armed = True
#         else:
#             self.armed = False

#     def publish_offboard_control_mode(self):
#         """Publish OffboardControlMode message."""
#         try:
#             offboard_msg = OffboardControlMode()
#             offboard_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
#             offboard_msg.position = False
#             offboard_msg.velocity = True  # Enable velocity control
#             offboard_msg.acceleration = False
#             offboard_msg.attitude = False
#             offboard_msg.body_rate = False
#             self.offboard_control_mode_publisher.publish(offboard_msg)
#         except Exception as e:
#             self.get_logger().error(f"Error publishing offboard control mode: {str(e)}")

#     def publish_trajectory_setpoint(self, vx=0.0, vy=0.0, vz=0.0, yaw_rate=0.0):
#         """Publish a trajectory setpoint in velocity mode with position maintenance."""
#         try:
#             # Initialize calculated velocities with the requested values
#             calculated_vx = vx
#             calculated_vy = vy
#             calculated_vz = vz
            
#             # Apply position maintenance for X if needed
#             if self.maintain_x and not self.moving_in_x and self.initial_x_position is not None:
#                 # Calculate error (desired - current)
#                 x_error = self.initial_x_position - self.current_position[0]
#                 # Apply P controller
#                 calculated_vx = self.x_kp * x_error
#                 # Limit the velocity to reasonable values
#                 calculated_vx = max(min(calculated_vx, self.max_velocity), -self.max_velocity)
#                 self.get_logger().debug(f"X maintenance: error={x_error:.2f}, vx={calculated_vx:.2f}")
            
#             # Apply position maintenance for Y if needed
#             if self.maintain_y and not self.moving_in_y and self.initial_y_position is not None:
#                 # Calculate error (desired - current)
#                 y_error = self.initial_y_position - self.current_position[1]
#                 # Apply P controller
#                 calculated_vy = self.y_kp * y_error
#                 # Limit the velocity to reasonable values
#                 calculated_vy = max(min(calculated_vy, self.max_velocity), -self.max_velocity)
#                 self.get_logger().debug(f"Y maintenance: error={y_error:.2f}, vy={calculated_vy:.2f}")
            
#             # Apply position maintenance for Z if needed
#             if self.maintain_z and not self.moving_in_z and self.initial_z_position is not None:
#                 # Calculate error (desired - current)
#                 z_error = self.initial_z_position - self.current_position[2]
#                 # Apply P controller
#                 calculated_vz = self.z_kp * z_error
#                 # Limit the velocity to reasonable values
#                 calculated_vz = max(min(calculated_vz, self.max_velocity), -self.max_velocity)
#                 self.get_logger().debug(f"Z maintenance: error={z_error:.2f}, vz={calculated_vz:.2f}")

#             self.publish_offboard_control_mode()
            
#             # Prepare trajectory message
#             trajectory_msg = TrajectorySetpoint()
#             trajectory_msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
#             trajectory_msg.velocity[0] = calculated_vx  # Use calculated value for X
#             trajectory_msg.velocity[1] = calculated_vy  # Use calculated value for Y
#             trajectory_msg.velocity[2] = calculated_vz  # Use calculated value for Z
#             trajectory_msg.position[0] = float('nan')
#             trajectory_msg.position[1] = float('nan')
#             trajectory_msg.position[2] = float('nan')
#             trajectory_msg.acceleration[0] = float('nan')
#             trajectory_msg.acceleration[1] = float('nan')
#             trajectory_msg.acceleration[2] = float('nan')
#             trajectory_msg.yaw = float('nan')
#             self.trajectory_setpoint_publisher.publish(trajectory_msg)
#         except Exception as e:
#             self.get_logger().error(f"Error publishing trajectory setpoint: {str(e)}")

#     def publish_vehicle_command(self, command, param1=0.0, param2=0.0, param3=0.0):
#         """Publish a VehicleCommand."""
#         try:
#             msg = VehicleCommand()
#             msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
#             msg.param1 = param1
#             msg.param2 = param2
#             msg.param3 = param3
#             msg.command = command
#             msg.target_system = 1
#             msg.target_component = 1
#             msg.source_system = 1
#             msg.source_component = 1
#             msg.from_external = True
#             self.vehicle_command_publisher.publish(msg)
#             self.get_logger().info(f"Published VehicleCommand: command={command}, param1={param1}, param2={param2}, param3={param3}")
#         except Exception as e:
#             self.get_logger().error(f"Error publishing vehicle command: {str(e)}")

#     def arm_drone(self):
#         """Command the drone to arm."""
#         self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, 1.0)

#     def publish_takeoff(self):
#         """Send takeoff command to PX4."""
#         self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 2.0)  # original take off altitude
#         self.get_logger().info(f"Sending takeoff command.")

#     def time_since_state_change(self):
#         """Get time since last state change in seconds."""
#         return (self.get_clock().now() - self.state_change_time).nanoseconds / 1e9

#     def change_state(self, new_state):
#         """Change state with proper logging and timing."""
#         self.get_logger().info(f"State transition: {self.state} -> {new_state}")
#         self.state = new_state
#         self.state_change_time = self.get_clock().now()
        
#         # Reset position maintenance variables on state change
#         if new_state == "ALLIGN":
#             self.initial_x_position = None  # Reset X position as well
#             self.initial_y_position = None
#             self.initial_z_position = None
#             self.maintain_x = False  # Reset X maintenance flag
#             self.maintain_y = False
#             self.maintain_z = False
        
#         # Start position maintenance in hover state
#         if new_state == "HOVER":
#             self.initial_x_position = self.current_position[0]  # Store X position
#             self.initial_y_position = self.current_position[1]
#             self.initial_z_position = self.current_position[2]
#             self.maintain_x = True  # Enable X maintenance
#             self.maintain_y = True
#             self.maintain_z = True
#             self.moving_in_x = False  # Reset movement flags
#             self.moving_in_y = False
#             self.moving_in_z = False

#     def print_velocity_info(self):
#         """Print detailed velocity information for debugging."""
#         if self.state == "ALLIGN":
#             # Format the velocity information
#             x_info = f"X: Moving: {'Yes' if self.moving_in_x else 'No'}, " \
#                     f"Maintaining: {'Yes' if self.maintain_x else 'No'}, " \
#                     f"Velocity: {self.vx:.2f}, " \
#                     f"Error: {(self.initial_x_position - self.current_position[0]):.2f} m" if self.initial_x_position is not None else "N/A"
            
#             y_info = f"Y: Moving: {'Yes' if self.moving_in_y else 'No'}, " \
#                     f"Maintaining: {'Yes' if self.maintain_y else 'No'}, " \
#                     f"Velocity: {self.vy:.2f}, " \
#                     f"Error: {(self.initial_y_position - self.current_position[1]):.2f} m" if self.initial_y_position is not None else "N/A"
            
#             z_info = f"Z: Moving: {'Yes' if self.moving_in_z else 'No'}, " \
#                     f"Maintaining: {'Yes' if self.maintain_z else 'No'}, " \
#                     f"Velocity: {self.vz:.2f}, " \
#                     f"Error: {(self.initial_z_position - self.current_position[2]):.2f} m" if self.initial_z_position is not None else "N/A"
            
#             self.get_logger().info("----- VELOCITY INFO -----")
#             self.get_logger().info(x_info)
#             self.get_logger().info(y_info)
#             self.get_logger().info(z_info)
#             self.get_logger().info("------------------------")

#     def timer_callback(self):
#         """Main loop that implements the state machine."""
#         try:
#             if self.state == "ARMING":
#                 if not self.armed:
#                     if self.current_mode != 4:
#                         self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 3.0)
#                     self.arm_drone()
#                     self.publish_takeoff()
#                 else:
#                     if self.current_mode != 4:
#                         self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 3.0)
#                     self.get_logger().info("Drone armed and taking off")
                    
#                     if self.time_since_state_change() >= 15.0:  # Wait for 15 seconds before switching to TAKEOFF
#                         self.change_state("TAKEOFF")

#             elif self.state == "TAKEOFF":
#                 if self.time_since_state_change() >= 10.0:  # Wait for 10 seconds before switching to ALLIGN
#                     self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 6.0)
#                     self.change_state("ALLIGN")
#                     self.target_acquired = False
#                     self.last_detection_time = None
                
#             elif self.state == "ALLIGN":
#                 # Print velocity information every second during alignment
#                 current_time = self.get_clock().now()
#                 if not hasattr(self, 'last_velocity_print_time') or \
#                 (current_time - self.last_velocity_print_time).nanoseconds / 1e9 >= 1.0:
#                     self.print_velocity_info()
#                     self.last_velocity_print_time = current_time

#                 # If no object detected, move upwards continuously
#                 if not self.target_acquired or (self.last_detection_time and 
#                 (self.get_clock().now() - self.last_detection_time).nanoseconds / 1e9 > 1.0):
#                     # No target or haven't seen one recently (more than 1 second)
#                     self.moving_in_z = True
#                     self.maintain_y = True
#                     self.vz = -0.5  # Move upwards
#                     self.vx = 0.0
                    
#                     # If we have an initial Y position, maintain it
#                     if self.initial_y_position is None:
#                         self.initial_y_position = self.current_position[1]
                    
#                     self.publish_trajectory_setpoint(vx=self.vx, vy=0.0, vz=self.vz)
#                     self.get_logger().debug("Moving upward to search for object")
#                     self.last_movement_time = self.get_clock().now()  # Update last movement time
                
#                 # If any velocity is non-zero, we're making adjustments
#                 if abs(self.vx) > 0.01 or abs(self.vy) > 0.01 or abs(self.vz) > 0.01:
#                     self.last_movement_time = self.get_clock().now()  # Update last movement time
                
#                 # Check if we've been stable (no movement) for 5 seconds
#                 if self.target_acquired and ((self.get_clock().now() - self.last_movement_time).nanoseconds / 1e9 >= 5.0):
#                     self.get_logger().info("No adjustments for 5 seconds, transitioning to HOVER")
#                     self.change_state("HOVER")
                
#                 # Check if we've been in ALLIGN state for more than 20 seconds without finding an object
#                 if self.time_since_state_change() >= 20.0 and not self.target_acquired:
#                     self.get_logger().info("No object detected for 20 seconds, transitioning to LANDING")
#                     self.change_state("LANDING")
            
#             elif self.state == "MOVING_FORWARD":
#                 # Store positions if not already stored
#                 if self.initial_y_position is None:
#                     self.initial_y_position = self.current_position[1]
#                 if self.initial_z_position is None:
#                     self.initial_z_position = self.current_position[2]
                
#                 # Set movement flags
#                 self.moving_in_y = False
#                 self.moving_in_z = False
#                 self.maintain_y = True
#                 self.maintain_z = True
                
#                 self.publish_trajectory_setpoint(vx=1.0, vy=0.0, vz=0.0)  # Move forward
#                 if self.time_since_state_change() >= 10.0:
#                     self.publish_trajectory_setpoint(vx=0.0, vy=0.0, vz=0.0)  # Stop moving
#                     self.change_state("MOVING_LEFT")

#             elif self.state == "MOVING_LEFT":
#                 # Set movement flags
#                 self.moving_in_y = True
#                 self.moving_in_z = False
#                 self.maintain_y = False
#                 self.maintain_z = True
                
#                 self.publish_trajectory_setpoint(vx=0.0, vy=1.0, vz=0.0)  # Move left
#                 if self.time_since_state_change() >= 10.0:
#                     self.publish_trajectory_setpoint(vx=0.0, vy=0.0, vz=0.0)  # Stop moving
#                     self.change_state("MOVING_RIGHT")

#             elif self.state == "MOVING_RIGHT":
#                 # Set movement flags
#                 self.moving_in_y = True
#                 self.moving_in_z = False
#                 self.maintain_y = False
#                 self.maintain_z = True
                
#                 self.publish_trajectory_setpoint(vx=0.0, vy=-1.0, vz=0.0)  # Move right
#                 if self.time_since_state_change() >= 10.0:
#                     self.publish_trajectory_setpoint(vx=0.0, vy=0.0, vz=0.0)  # Stop moving
#                     self.change_state("MOVING_BACKWARD")

#             elif self.state == "MOVING_BACKWARD":
#                 # Set movement flags
#                 self.moving_in_y = False
#                 self.moving_in_z = False
#                 self.maintain_y = True
#                 self.maintain_z = True
                
#                 self.publish_trajectory_setpoint(vx=-1.0, vy=0.0, vz=0.0)  # Move backward
#                 if self.time_since_state_change() >= 10.0:
#                     self.publish_trajectory_setpoint(vx=0.0, vy=0.0, vz=0.0)  # Stop moving
#                     self.change_state("HOVER")

#             elif self.state == "HOVER":
#                 # In hover, maintain both Y and Z positions
#                 self.moving_in_y = False
#                 self.moving_in_z = False
#                 self.maintain_y = True
#                 self.maintain_z = True
                
#                 self.publish_trajectory_setpoint(vx=0.0, vy=0.0, vz=0.0)  # Stay in place
#                 if self.time_since_state_change() >= 5.0:  # Hover for 5 seconds
#                     self.change_state("LANDING")

#             elif self.state == "LANDING":
#                 # During landing, don't maintain positions
#                 self.maintain_y = False
#                 self.maintain_z = False
                
#                 self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 6.0)
#                 self.get_logger().info("Landing...")
#                 if self.time_since_state_change() >= 5.0:  # Wait for 5 seconds before declaring landed
#                     self.change_state("LANDED")

#             elif self.state == "LANDED":
#                 self.get_logger().info("Drone has landed successfully.")
#                 # Do nothing else, we're done

#         except Exception as e:
#             self.get_logger().error(f"Error in timer callback: {str(e)}")
#             # Safety fallback - try to land if something goes wrong
#             if self.state != "LANDING" and self.state != "LANDED":
#                 self.get_logger().error("Error detected, attempting emergency landing")
#                 self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_DO_SET_MODE, 1.0, 4.0, 6.0)
#                 self.change_state("LANDING")

# def main(args=None):
#     rclpy.init(args=args)
#     node = ZDCalNode()

#     try:
#         rclpy.spin(node)
#     except KeyboardInterrupt:
#         node.get_logger().info("Keyboard Interrupt detected. Shutting down...")
#     except Exception as e:
#         node.get_logger().error(f"Unexpected error: {str(e)}")
#     finally:
#         node.destroy_node()
#         if rclpy.ok():
#             rclpy.shutdown()

# if __name__ == '__main__':
#     main()