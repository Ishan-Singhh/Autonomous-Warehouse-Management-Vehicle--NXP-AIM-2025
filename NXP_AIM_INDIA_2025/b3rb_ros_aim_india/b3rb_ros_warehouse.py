# Copyright 2025 NXP

# Copyright 2016 Open Source Robotics Foundation, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import rclpy
from rclpy.node import Node
from rclpy.timer import Timer
from rclpy.action import ActionClient
from rclpy.parameter import Parameter

import math
import time
import numpy as np
import cv2
from typing import Optional, Tuple
import asyncio
import threading

from sensor_msgs.msg import Joy
from sensor_msgs.msg import LaserScan
from sensor_msgs.msg import CompressedImage

from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import PoseWithCovarianceStamped

from nav_msgs.msg import OccupancyGrid
from nav2_msgs.msg import BehaviorTreeLog
from nav2_msgs.action import NavigateToPose
from action_msgs.msg import GoalStatus

from synapse_msgs.msg import Status
from synapse_msgs.msg import WarehouseShelf

from scipy.ndimage import label, center_of_mass
from scipy.spatial.distance import euclidean
from sklearn.decomposition import PCA

import tkinter as tk
from tkinter import ttk

QOS_PROFILE_DEFAULT = 10
SERVER_WAIT_TIMEOUT_SEC = 5.0

PROGRESS_TABLE_GUI = True


class WindowProgressTable:
	def __init__(self, root, shelf_count):
		self.root = root
		self.root.title("Shelf Objects & QR Link")
		self.root.attributes("-topmost", True)

		self.row_count = 2
		self.col_count = shelf_count

		self.boxes = []
		for row in range(self.row_count):
			row_boxes = []
			for col in range(self.col_count):
				box = tk.Text(root, width=10, height=3, wrap=tk.WORD, borderwidth=1,
					      relief="solid", font=("Helvetica", 14))
				box.insert(tk.END, "NULL")
				box.grid(row=row, column=col, padx=3, pady=3, sticky="nsew")
				row_boxes.append(box)
			self.boxes.append(row_boxes)

		# Make the grid layout responsive.
		for row in range(self.row_count):
			self.root.grid_rowconfigure(row, weight=1)
		for col in range(self.col_count):
			self.root.grid_columnconfigure(col, weight=1)

	def change_box_color(self, row, col, color):
		self.boxes[row][col].config(bg=color)

	def change_box_text(self, row, col, text):
		self.boxes[row][col].delete(1.0, tk.END)
		self.boxes[row][col].insert(tk.END, text)

box_app = None
def run_gui(shelf_count):
	global box_app
	root = tk.Tk()
	box_app = WindowProgressTable(root, shelf_count)
	root.mainloop()


class WarehouseExplore(Node):
	""" Initializes warehouse explorer node with the required publishers and subscriptions.

		Returns:
			None
	"""
	def __init__(self):
		super().__init__('warehouse_explore')

		self.action_client = ActionClient(
			self,
			NavigateToPose,
			'/navigate_to_pose')

		# Maps qr string to best (object_names, object_counts)
		self.best_shelf_results = {}

		# Store last published message (for debugging / GUI)
		self.last_published_shelf = None

		self.current_gui_col = 0


		self.subscription_pose = self.create_subscription(
			PoseWithCovarianceStamped,
			'/pose',
			self.pose_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_global_map = self.create_subscription(
			OccupancyGrid,
			'/global_costmap/costmap',
			self.global_map_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_simple_map = self.create_subscription(
			OccupancyGrid,
			'/map',
			self.simple_map_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_status = self.create_subscription(
			Status,
			'/cerebri/out/status',
			self.cerebri_status_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_behavior = self.create_subscription(
			BehaviorTreeLog,
			'/behavior_tree_log',
			self.behavior_tree_log_callback,
			QOS_PROFILE_DEFAULT)

		self.subscription_shelf_objects = self.create_subscription(
			WarehouseShelf,
			'/shelf_objects',
			self.shelf_objects_callback,
			QOS_PROFILE_DEFAULT)

		# Subscription for camera images.
		self.subscription_camera = self.create_subscription(
			CompressedImage,
			'/camera/image_raw/compressed',
			self.camera_image_callback,
			QOS_PROFILE_DEFAULT)

		self.publisher_joy = self.create_publisher(
			Joy,
			'/cerebri/in/joy',
			QOS_PROFILE_DEFAULT)

		# Publisher for output image (for debug purposes).
		self.publisher_qr_decode = self.create_publisher(
			CompressedImage,
			"/debug_images/qr_code",
			QOS_PROFILE_DEFAULT)

		self.publisher_shelf_data = self.create_publisher(
			WarehouseShelf,
			"/shelf_data",
			QOS_PROFILE_DEFAULT)

		self.declare_parameter('shelf_count', 1)
		self.declare_parameter('initial_angle', 0.0)

		self.shelf_count = \
			self.get_parameter('shelf_count').get_parameter_value().integer_value
		self.initial_angle = \
			self.get_parameter('initial_angle').get_parameter_value().double_value
		

		# --- Robot State ---
		self.armed = False
		self.logger = self.get_logger()

		# --- Robot Pose ---
		self.pose_curr = PoseWithCovarianceStamped()
		self.buggy_pose_x = 0.0
		self.buggy_pose_y = 0.0
		self.buggy_center = (0.0, 0.0)
		self.world_center = (0.0, 0.0)

		# --- Map Data ---
		self.simple_map_curr = None
		self.global_map_curr = None

		# --- Goal Management ---
		self.xy_goal_tolerance = 0.5
		self.goal_completed = True  # No goal is currently in-progress.
		self.goal_handle_curr = None
		self.cancelling_goal = False
		self.recovery_threshold = 10

		# --- Goal Creation ---
		self._frame_id = "map"

		# --- Exploration Parameters ---
		self.max_step_dist_world_meters = 7.0   #7
		self.min_step_dist_world_meters = 4.0	#4
		self.full_map_explored_count = 0
		self.coverage_percent = 0.0  # Percentage of map explored
		# --- QR Code Data ---
		self.qr_code_str = "Empty"
		if PROGRESS_TABLE_GUI:
			self.table_row_count = 0
			self.table_col_count = 0
		self.is_exploring = True
		# --- Shelf Data ---
		self.shelf_objects_curr = WarehouseShelf()
		# --- Shelf Locations ---
		self.shelf_locations = []
		self.visited_shelves = set()  # Track visited shelf locations
		self.current_shelf_index = 0  # Index of current shelf being processed
		self.shelf_visit_state = "MOVING_TO_SHELF"  # States: MOVING_TO_SHELF, SCANNING_OBJECTS, SCANNING_QR, COMPLETED
		self.current_shelf = None  # Current shelf being processed
		self.shelf_poses_queue = []  # Queue of poses to visit for current shelf
		self.next_heuristic_angle = self.initial_angle  # Next shelf heuristic angle
		self.shelf_sequence_complete = False
		self.next_pose_timer = None  # Timer to delay next pose
		self.prev_shelf = None  # Previous shelf processed

	def _delayed_process_next_pose(self):
		if self.next_pose_timer is not None:
			self.next_pose_timer.cancel()
			self.next_pose_timer = None

		self.logger.info("Delay over. Processing next shelf pose.")
		self.process_shelf_pose_queue()

	def pose_callback(self, message):
		"""Callback function to handle pose updates.

		Args:
			message: ROS2 message containing the current pose of the rover.

		Returns:
			None
		"""
		self.pose_curr = message
		self.buggy_pose_x = message.pose.pose.position.x
		self.buggy_pose_y = message.pose.pose.position.y
		self.buggy_center = (self.buggy_pose_x, self.buggy_pose_y)

	def simple_map_callback(self, message):
		"""Callback function to handle simple map updates.

		Args:
			message: ROS2 message containing the simple map data.

		Returns:
			None
		"""
		self.simple_map_curr = message
		map_info = self.simple_map_curr.info
		self.world_center = self.get_world_coord_from_map_coord(
			map_info.width / 2, map_info.height / 2, map_info
		)

	def global_map_callback(self, message):
		self.global_map_curr = message
		map_data = np.array(self.global_map_curr.data)
		known_cells = np.count_nonzero(map_data != -1)
		total_cells = map_data.size
		self.coverage_percent = (known_cells / total_cells) * 100

		# Log once every 10%
		# if int(coverage_percent) % 10 == 0:
		# self.get_logger().info(f"Exploration coverage: {self.coverage_percent:.8f}%")

		if not self.is_exploring:
			return
		if not self.goal_completed:
			return
		


		if self.full_map_explored_count >= 8 or self.coverage_percent >= 99.96:
			self.get_logger().info("Exploration limit reached; starting shelf detection and navigation.")
			self.next_pose_timer = self.create_timer(1.0, self._delayed_process_next_pose)

			self.detect_shelves(self.simple_map_curr)
			self.is_exploring = False
			self.move_to_shelf()
			return
		height, width = self.global_map_curr.info.height, self.global_map_curr.info.width
		map_array = np.array(self.global_map_curr.data).reshape((height, width))

		frontier_goals = self.get_frontiers_for_space_exploration(map_array)

		if frontier_goals:
			x_goal, y_goal = frontier_goals[0]
			goal = self.create_goal_from_world_coord(x_goal, y_goal)
			self.send_goal_from_world_pose(goal)
			self.get_logger().info(f"Sent exploration goal to: ({x_goal:.2f}, {y_goal:.2f})")
			return
		else:
			# No frontiers found → expand search range slightly
			self.max_step_dist_world_meters += 2.0
			self.min_step_dist_world_meters = max(0.25, self.min_step_dist_world_meters - 1.0)

		self.full_map_explored_count += 1
		self.get_logger().info(f"No frontier found; exploration count = {self.full_map_explored_count}")


	def get_frontiers_for_space_exploration(self, map_array):
		# from scipy.ndimage import label, center_of_mass

		# Step 1: Detect candidate frontier pixels
		frontier_mask = np.zeros_like(map_array, dtype=bool)
		for y in range(1, map_array.shape[0] - 1):
			for x in range(1, map_array.shape[1] - 1):
				if map_array[y, x] == -1:  # Unknown
					neighbors = [map_array[y+dy, x+dx] for dx, dy in [(-1,0),(1,0),(0,-1),(0,1)]]
					if any(n == 0 for n in neighbors):  # Adjacent to free
						frontier_mask[y, x] = True

		# Step 2: Cluster frontiers
		labeled, num_features = label(frontier_mask)

		best_score = -float('inf')
		best_frontier_world = None

		for i in range(1, num_features + 1):
			cluster_mask = (labeled == i)
			size = np.sum(cluster_mask)

			if size < 4:
				continue  # Skip tiny frontiers

			y_com, x_com = center_of_mass(cluster_mask)
			x_com = int(x_com)
			y_com = int(y_com)

			map_info = self.global_map_curr.info
			x_world, y_world = self.get_world_coord_from_map_coord(x_com, y_com, map_info)

			distance = euclidean(self.buggy_center, (x_world, y_world))

			if distance > self.max_step_dist_world_meters or distance < self.min_step_dist_world_meters:
				continue  # Skip too close or too far

			score = 1.0 * size - 1.5 * distance  # Tunable weights

			if score > best_score:
				best_score = score
				best_frontier_world = (x_world, y_world)

		if best_frontier_world:
			return [best_frontier_world]
		else:
			return []

	def detect_shelves(self, message):
		"""Detects shelves in the given map array using PCA and shape filtering.

		Args:
			message: nav_msgs/OccupancyGrid message.

		Returns:
			None. Updates self.shelf_locations with shelf coordinates and orientation.
		"""
		import numpy as np
		import cv2
		from sklearn.decomposition import PCA

		self.shelf_locations.clear()
		SHELF_WIDTH_M = 1.35
		SHELF_DEPTH_M = 0.5
		MERGE_DISTANCE_M = 0.5  # meters
		MIN_CLUSTER_SIZE = 90

		# --- Load Map ---
		map_array = np.array(message.data).reshape(
			(message.info.height, message.info.width))
		resolution = message.info.resolution
		origin_x = message.info.origin.position.x
		origin_y = message.info.origin.position.y
		height, width = map_array.shape

		# --- Preprocessing ---
		binary_map = np.uint8(map_array == 100)  # obstacles

		# --- Connected Components ---
		num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_map, connectivity=8)
		clusters = []
		for i in range(1, num_labels):  # Skip background
			if stats[i, cv2.CC_STAT_AREA] >= MIN_CLUSTER_SIZE:
				mask = (labels == i)
				ys, xs = np.where(mask)
				coords = np.column_stack((xs, ys))  # pixel coordinates
				clusters.append(coords)

		# --- Merge Close Clusters ---
		def merge_close_clusters(clusters, threshold_m):
			merged = []
			used = set()
			for i, ci in enumerate(clusters):
				if i in used:
					continue
				current = ci
				for j in range(i + 1, len(clusters)):
					if j in used:
						continue
					dist = np.linalg.norm(np.mean(current, axis=0) - np.mean(clusters[j], axis=0))
					if dist * resolution < threshold_m:
						current = np.vstack((current, clusters[j]))
						used.add(j)
				merged.append(current)
				used.add(i)
			return merged

		clusters = merge_close_clusters(clusters, MERGE_DISTANCE_M)

		# --- PCA and Shape Filtering ---
		for cluster in clusters:
			if cluster.shape[0] < MIN_CLUSTER_SIZE:
				continue

			cluster_m = cluster * resolution
			pca = PCA(n_components=2)
			pca.fit(cluster_m)

			center = np.mean(cluster_m, axis=0)
			angle = np.arctan2(pca.components_[0][1], pca.components_[0][0])

			# Project onto PCA axes
			proj = (cluster_m - center) @ pca.components_.T
			lengths = np.max(proj, axis=0) - np.min(proj, axis=0)
			w, h = sorted(lengths)

			if (
				abs(w - SHELF_DEPTH_M) < 0.3 and abs(h - SHELF_WIDTH_M) < 0.3
			) or (
				abs(h - SHELF_DEPTH_M) < 0.3 and abs(w - SHELF_WIDTH_M) < 0.3
			):
				world_x = center[0] + origin_x
				world_y = center[1] + origin_y
				self.shelf_locations.append({
					"x": world_x,
					"y": world_y,
					"angle": angle
				})

		# --- Logging ---
		self.get_logger().info(f"\nDetected {len(self.shelf_locations)} shelves:")
		for i, shelf in enumerate(self.shelf_locations, 1):
			self.get_logger().info(f"Shelf {i}: (x={shelf['x']:.3f}, y={shelf['y']:.3f}), θ={shelf['angle']:.3f} rads")

	
	def compute_shelf_view_poses(self, shelf, dist_obj=1.0, dist_qr=1.0):
		"""
		Compute two poses for a shelf:
		- One for viewing objects (perpendicular to shelf orientation)
		- One for viewing QR (parallel to shelf orientation)

		Args:
			shelf: Dictionary with keys "x", "y", "angle"
			dist_obj: Distance from shelf to view objects
			dist_qr: Distance from shelf to view QR

		Returns:
			Tuple:
				(object_pose1, object_pose2, qr_pose1, qr_pose2)
			Each is a PoseStamped message.
		"""
		from geometry_msgs.msg import PoseStamped

		sx, sy, theta = shelf["x"], shelf["y"], shelf["angle"]

		# --- QR Poses (Parallel) ---
		qr_pose1_x = sx + dist_qr * math.cos(theta)
		qr_pose1_y = sy + dist_qr * math.sin(theta)
		qr_pose2_x = sx - dist_qr * math.cos(theta)
		qr_pose2_y = sy - dist_qr * math.sin(theta)

		qr_pose1 = self.create_goal_from_world_coord(qr_pose1_x, qr_pose1_y, yaw=theta+ math.pi)
		qr_pose2 = self.create_goal_from_world_coord(qr_pose2_x, qr_pose2_y, yaw=theta )

		# --- Object Poses (Perpendicular) ---
		obj_angle1 = theta + math.pi / 2
		obj_angle2 = theta - math.pi / 2

		obj_pose1_x = sx + dist_obj * math.cos(obj_angle1)
		obj_pose1_y = sy + dist_obj * math.sin(obj_angle1)
		obj_pose2_x = sx + dist_obj * math.cos(obj_angle2)
		obj_pose2_y = sy + dist_obj * math.sin(obj_angle2)

		obj_pose1 = self.create_goal_from_world_coord(obj_pose1_x, obj_pose1_y, yaw=obj_angle1+ math.pi)
		obj_pose2 = self.create_goal_from_world_coord(obj_pose2_x, obj_pose2_y, yaw=obj_angle2+ math.pi)

		return obj_pose1, obj_pose2, qr_pose1, qr_pose2
	
	def move_to_shelf(self):
		"""Moves the rover to the next shelf location using heuristic-based navigation."""
		if self.is_exploring:
			return  # Skip during exploration
		
		if self.shelf_sequence_complete:
			self.get_logger().info("All shelves have been visited!")
			return
		
		if not self.shelf_locations:
			self.get_logger().warn("No shelf locations available.")
			return

		# If we don't have a current shelf, find the next one
		if self.current_shelf is None:
			self.current_shelf = self.find_next_shelf_by_heuristic()
			if self.current_shelf is None:
				self.get_logger().warn("No more shelves to visit based on heuristic.")
				self.shelf_sequence_complete = True
				return
			
			# Generate poses for this shelf
			self.generate_shelf_visit_poses()
			self.get_logger().info(f"Starting to visit shelf at ({self.current_shelf['x']:.2f}, {self.current_shelf['y']:.2f})")
		
		# Process the pose queue for current shelf
		self.process_shelf_pose_queue()

	def identify_next_shelf(self, current_shelf, heuristic_angle, visited_set):
		"""
		Finds the next shelf whose vector from current shelf CoM matches the given heuristic angle.
		
		Args:
			current_shelf: dict with 'x', 'y', 'angle' of the current shelf
			heuristic_angle: float (0-360) angle from x-axis to next shelf
			visited_set: set of (x, y) tuples marking visited shelves
		
		Returns:
			next_shelf (dict) or None
		"""
		curr_x, curr_y = current_shelf["x"], current_shelf["y"]

		best_shelf = None
		min_angle_diff = float('inf')

		for shelf in self.shelf_locations:
			next_x, next_y = shelf["x"], shelf["y"]
			key = (round(next_x, 2), round(next_y, 2))
			if key in visited_set or key == (round(curr_x, 2), round(curr_y, 2)):
				continue

			# Angle from current shelf to this candidate shelf
			dx = next_x - curr_x
			dy = next_y - curr_y
			angle_rad = math.atan2(dy, dx)
			angle_deg = math.degrees(angle_rad) % 360

			# Angular difference
			diff = abs((angle_deg - heuristic_angle + 180) % 360 - 180)

			if diff < min_angle_diff:
				min_angle_diff = diff
				best_shelf = shelf

		# Define tolerance (±10°)
		if min_angle_diff <= 10.0:
			return best_shelf
		else:
			self.get_logger().warn(f"No shelf matched heuristic angle {heuristic_angle}° within tolerance.")
			return None

	def find_next_shelf_by_heuristic(self):
		"""Find the next shelf using the heuristic angle."""
		if self.current_shelf_index == 0:
			# First shelf: use robot's current position and initial_angle
			current_pos = {"x": 0, "y": 0}
			heuristic_angle = self.initial_angle
		else:
			# Subsequent shelves: use previous shelf position and angle from QR
			current_pos = self.prev_shelf
			heuristic_angle = self.next_heuristic_angle
		
		next_shelf = self.identify_next_shelf(current_pos, heuristic_angle, self.visited_shelves)
		
		if next_shelf:
			# Mark as visited
			shelf_key = (round(next_shelf["x"], 2), round(next_shelf["y"], 2))
			self.visited_shelves.add(shelf_key)
			self.current_shelf_index += 1
			self.get_logger().info(f"Selected shelf {self.current_shelf_index} using heuristic angle {heuristic_angle}°")
		
		return next_shelf
	
	def generate_shelf_visit_poses(self):
		"""Generate the sequence of poses to visit for the current shelf."""
		if not self.current_shelf:
			return
		curr_x, curr_y= self.buggy_pose_x, self.buggy_pose_y
		# Get poses for object detection and QR scanning
		obj_pose1, obj_pose2, qr_pose1, qr_pose2 = self.compute_shelf_view_poses(
			self.current_shelf, dist_obj=3.2, dist_qr=2.5
		)
		obj_pose11, obj_pose22, qr_pose11, qr_pose22 = self.compute_shelf_view_poses(
			self.current_shelf, dist_obj=1.8, dist_qr=1.5
		)
		d1=np.linalg.norm((np.array([obj_pose1.pose.position.x, obj_pose1.pose.position.y]) - np.array([curr_x, curr_y])))
		d2=np.linalg.norm((np.array([obj_pose2.pose.position.x, obj_pose2.pose.position.y]) - np.array([curr_x, curr_y])))
		
		
		if d1 < d2:
			self.shelf_poses_queue = [
				("SCANNING_OBJECTS", obj_pose1),
				("SCANNING_OBJECTS", obj_pose11),
				("SCANNING_OBJECTS", obj_pose1),
				("SCANNING_OBJECTS", obj_pose11),
				("SCANNING_QR", qr_pose1),
				("SCANNING_QR", qr_pose11),
				("SCANNING_QR", qr_pose1),
				("SCANNING_QR", qr_pose11),
				("SCANNING_QR", qr_pose1),
				("SCANNING_QR", qr_pose11)
			]
		else:
			self.shelf_poses_queue = [
				("SCANNING_OBJECTS", obj_pose2),
				("SCANNING_OBJECTS", obj_pose22),
				("SCANNING_OBJECTS", obj_pose2),
				("SCANNING_OBJECTS", obj_pose22),
				("SCANNING_QR", qr_pose1),
				("SCANNING_QR", qr_pose11),
				("SCANNING_QR", qr_pose1),
				("SCANNING_QR", qr_pose11),
				("SCANNING_QR", qr_pose1),
				("SCANNING_QR", qr_pose11)
			]
		
		self.shelf_visit_state = "MOVING_TO_SHELF"
	
	def process_shelf_pose_queue(self):
		"""Process the next pose in the shelf visit queue."""
		# self.get_logger().info(f"Processing shelf pose queue. Queue length: {len(self.shelf_poses_queue)}")
		# self.get_logger().info(f"Goal completed: {self.goal_completed}")
		
		if not self.shelf_poses_queue:
			# Finished with current shelf
			self.get_logger().info("Shelf pose queue empty, completing current shelf")
			self.complete_current_shelf()
			return
		
		# Only proceed if no goal is active
		if not self.goal_completed:
			# self.get_logger().warn("Cannot process next pose - goal not completed yet")
			return
		
		# Get next pose
		state, pose = self.shelf_poses_queue.pop(0)
		self.shelf_visit_state = state
		
		self.get_logger().info(f"Moving to {state} pose for shelf {self.current_shelf_index}")
		self.get_logger().info(f"Remaining poses in queue: {len(self.shelf_poses_queue)}")
		
		success = self.send_goal_from_world_pose(pose)
		
		if not success:
			self.get_logger().warn(f"Failed to send goal for {state}")
			# Try next pose
			self.process_shelf_pose_queue()


	def complete_current_shelf(self):
		"""Complete processing of the current shelf and prepare for next."""
		if self.current_shelf:
			self.get_logger().info(f"Completed shelf {self.current_shelf_index}")
			
			# # Check if we have a valid QR code with heuristic for next shelf
			# if self.qr_code_str and self.qr_code_str != "Empty":
			# 	try:
			# 		# Parse next heuristic angle from QR code
			# 		parts = self.qr_code_str.split('_')
			# 		if len(parts) >= 2:
			# 			self.next_heuristic_angle = float(parts[1])
			# 			self.get_logger().info(f"Next heuristic angle: {self.next_heuristic_angle}°")
			# 	except (ValueError, IndexError) as e:
			# 		self.get_logger().warn(f"Could not parse heuristic from QR: {e}")
			# 		self.next_heuristic_angle = 0.0  # Default fallback

			# No need to re-parse QR code here
			if self.qr_code_str and self.qr_code_str != "Empty":
				self.get_logger().info(f"Using heuristic angle: {self.next_heuristic_angle}° from QR: {self.qr_code_str}")
			else:
				self.get_logger().warn("No QR code found, using default angle 0.0")
				self.next_heuristic_angle = 0.0  # Default fallback
			
			
			# Reset for next shelf
			self.prev_shelf = self.current_shelf
			self.current_shelf = None
			self.shelf_poses_queue = []
			self.qr_code_str = "Empty"  # Reset QR for next shelf
			
			# Small delay before moving to next shelf
			self.create_timer(1.0, self.move_to_shelf_timer_callback)


	def move_to_shelf_timer_callback(self):
		"""Timer callback to move to next shelf after a delay."""
		self.move_to_shelf()


	def publish_debug_image(self, publisher, image):
		"""Publishes images for debugging purposes.

		Args:
			publisher: ROS2 publisher of the type sensor_msgs.msg.CompressedImage.
			image: Image given by an n-dimensional numpy array.

		Returns:
			None
		"""
		if image.size:
			message = CompressedImage()
			_, encoded_data = cv2.imencode('.jpg', image)
			message.format = "jpeg"
			message.data = encoded_data.tobytes()
			publisher.publish(message)

	def camera_image_callback(self, message):
		"""Callback function to handle incoming camera images."""
		if self.is_exploring:
			return  # Skip QR decoding during exploration
		
		# Only scan for QR codes when in QR scanning state
		if self.shelf_visit_state != "SCANNING_QR":
			return
		
		np_arr = np.frombuffer(message.data, np.uint8)
		image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
		qr_detector = cv2.QRCodeDetector()
		try:
			qr_data, _, _ = qr_detector.detectAndDecode(image)
		except Exception as e:
			print(f"QR decoding error: {e}")
			qr_data = "Empty"
		if qr_data:
			
			# Skip if we already have this QR code
			if qr_data == self.qr_code_str:
				return
			self.get_logger().info(f"QR Code Detected: {qr_data}")
			
			
			try:
				parts = qr_data.split('_')
				if len(parts) != 3:
					self.get_logger().warn("QR Code format is incorrect.")
					return
				
				shelf_id = int(parts[0])
				angle = float(parts[1])
				secret = parts[2]

				if len(secret) != 22:
					self.get_logger().warn("QR Code secret is not 22 characters long.")
					return
				
				# Update QR code
				self.qr_code_str = qr_data
				# CRITICAL FIX: Update the heuristic angle immediately here
				self.next_heuristic_angle = angle
				self.get_logger().info(f"Updated next heuristic angle to: {self.next_heuristic_angle}°")
				
				
				# Publish shelf data with QR code
				if self.last_published_shelf:
					shelf_data_message = WarehouseShelf()
					shelf_data_message.qr_decoded = qr_data
					shelf_data_message.object_name = self.last_published_shelf.object_name
					shelf_data_message.object_count = self.last_published_shelf.object_count
					self.publisher_shelf_data.publish(shelf_data_message)
					
					self.get_logger().info(f"Published shelf data with QR: Shelf ID={shelf_id}, Angle={angle}")
				
				# Update GUI
				if PROGRESS_TABLE_GUI:
					self.update_gui_qr(shelf_id, qr_data)
				
				# QR found, can move to next shelf sooner
				self.shelf_poses_queue = []  # Clear remaining poses
				self.last_published_shelf=None  # Reset last published shelf
				self.best_shelf_results = {}
				self.create_timer(1.0, self.move_to_shelf_timer_callback)
				

				
			except Exception as e:
				self.get_logger().error(f"Error parsing QR Code: {e}")

	def update_gui_qr(self, shelf_id, qr_data):
		"""Update GUI with QR code data."""
		try:
			col = shelf_id - 1 if shelf_id > 0 else 0
			# Row 1: QR code
			box_app.change_box_text(1, col, qr_data)
			box_app.change_box_color(1, col, "yellow")
		except Exception as e:
			self.get_logger().warn(f"GUI QR update failed: {e}")


	def cerebri_status_callback(self, message):
		"""Callback function to handle cerebri status updates.

		Args:
			message: ROS2 message containing cerebri status.

		Returns:
			None
		"""
		if message.mode == 3 and message.arming == 2:
			self.armed = True
		else:
			# Initialize and arm the CMD_VEL mode.
			msg = Joy()
			msg.buttons = [0, 1, 0, 0, 0, 0, 0, 1]
			msg.axes = [0.0, 0.0, 0.0, 0.0]
			self.publisher_joy.publish(msg)

	def behavior_tree_log_callback(self, message):
		"""Alternative method for checking goal status.

		Args:
			message: ROS2 message containing behavior tree log.

		Returns:
			None
		"""
		for event in message.event_log:
			if (event.node_name == "FollowPath" and
				event.previous_status == "SUCCESS" and
				event.current_status == "IDLE"):
				# self.goal_completed = True
				# self.goal_handle_curr = None
				pass

	
		"""
		* Example for sending WarehouseShelf messages for evaluation.
			shelf_data_message = WarehouseShelf()

			shelf_data_message.object_name = ["car", "clock"]
			shelf_data_message.object_count = [1, 2]
			shelf_data_message.qr_decoded = "test qr string"

			self.publisher_shelf_data.publish(shelf_data_message)

		* Alternatively, you may store the QR for current shelf as self.qr_code_str.
			Then, add it as self.shelf_objects_curr.qr_decoded = self.qr_code_str
			Then, publish as self.publisher_shelf_data.publish(self.shelf_objects_curr)
			This, will publish the current detected objects with the last QR decoded.
		"""

		# Optional code for populating TABLE GUI with detected objects and QR data.
		"""
		if PROGRESS_TABLE_GUI:
			shelf = self.shelf_objects_curr
			obj_str = ""
			for name, count in zip(shelf.object_name, shelf.object_count):
				obj_str += f"{name}: {count}\n"

			box_app.change_box_text(self.table_row_count, self.table_col_count, obj_str)
			box_app.change_box_color(self.table_row_count, self.table_col_count, "cyan")
			self.table_row_count += 1

			box_app.change_box_text(self.table_row_count, self.table_col_count, self.qr_code_str)
			box_app.change_box_color(self.table_row_count, self.table_col_count, "yellow")
			self.table_row_count = 0
			self.table_col_count += 1
		"""
	



	def rover_move_manual_mode(self, speed, turn):
		"""Operates the rover in manual mode by publishing on /cerebri/in/joy.

		Args:
			speed: The speed of the car in float. Range = [-1.0, +1.0];
				   Direction: forward for positive, reverse for negative.
			turn: Steer value of the car in float. Range = [-1.0, +1.0];
				  Direction: left turn for positive, right turn for negative.

		Returns:
			None
		"""
		msg = Joy()
		msg.buttons = [1, 0, 0, 0, 0, 0, 0, 1]
		msg.axes = [0.0, speed, 0.0, turn]
		self.publisher_joy.publish(msg)



	def shelf_objects_callback(self, message):
		"""Callback function to handle shelf objects updates."""
		if self.is_exploring:
			return  # Skip shelf updates during exploration
		
		self.shelf_objects_curr = message
		object_names = message.object_name
		object_count = message.object_count
		total_count = sum(object_count)
		
		# Only process if we're in object scanning state
		if self.shelf_visit_state != "SCANNING_OBJECTS":
			return
		# self.get_logger().info("Processing shelf objects callback")
		# Check if this is a good detection (total count should be reasonable)
		if total_count == 0 or total_count > 6:
			# self.get_logger().info(f"Ignoring detection with {total_count} objects")
			return
		
		# Use QR code from current shelf processing, default to "Empty" if not available
		qr_for_shelf = self.qr_code_str if self.qr_code_str else "Empty"
		
		# Check if this is better than previous results for this shelf
		current_best = self.best_shelf_results.get(qr_for_shelf, ([], []))
		best_prev_total = sum(current_best[1]) if current_best[1] else 0

		if total_count > best_prev_total:
			# Better result found → update memory and publish
			self.best_shelf_results[qr_for_shelf] = (object_names, object_count)

			shelf_data_message = WarehouseShelf()
			shelf_data_message.object_name = object_names
			shelf_data_message.object_count = object_count
			# shelf_data_message.qr_decoded = qr_for_shelf

			self.publisher_shelf_data.publish(shelf_data_message)
			self.last_published_shelf = shelf_data_message
			
			self.get_logger().info(f"Published improved shelf data: Total={total_count}")

			
			
			# Update GUI if enabled
			if PROGRESS_TABLE_GUI:
				self.update_gui_objects(object_names, object_count)

			if total_count == 6:
				self.get_logger().info("Detected 6 objects. Skipping remaining object poses and moving to QR scanning.")
				
				# Retain only QR scanning poses in the queue
				self.shelf_poses_queue = [
					(state, pose) for (state, pose) in self.shelf_poses_queue if state == "SCANNING_QR"
				]
				
				# Start timer to move to next pose (which will now be a QR pose)
				self.create_timer(1.0, self._delayed_process_next_pose)
				return

	def update_gui_objects(self, object_names, object_count):
		"""Update GUI with detected objects."""
		try:
			col = self.current_shelf_index - 1 if self.current_shelf_index > 0 else 0
			
			# Format object display
			obj_str = ""
			for name, count in zip(object_names, object_count):
				obj_str += f"{name}: {count}\n"

			# Row 0: Detected objects
			box_app.change_box_text(0, col, obj_str)
			box_app.change_box_color(0, col, "cyan")

		except Exception as e:
			self.get_logger().warn(f"GUI update failed: {e}")


	def cancel_goal_callback(self, future):
		"""
		Callback function executed after a cancellation request is processed.

		Args:
			future (rclpy.Future): The future is the result of the cancellation request.
		"""
		cancel_result = future.result()
		if cancel_result:
			self.logger.info("Goal cancellation successful.")
			self.cancelling_goal = False  # Mark cancellation as completed (success).
			return True
		else:
			self.logger.error("Goal cancellation failed.")
			self.cancelling_goal = False  # Mark cancellation as completed (failed).
			return False

	def cancel_current_goal(self):
		"""Requests cancellation of the currently active navigation goal."""
		if self.goal_handle_curr is not None and not self.cancelling_goal:
			self.cancelling_goal = True  # Mark cancellation in-progress.
			self.logger.info("Requesting cancellation of current goal...")
			cancel_future = self.action_client._cancel_goal_async(self.goal_handle_curr)
			cancel_future.add_done_callback(self.cancel_goal_callback)

	# def goal_result_callback(self, future):
	# 	"""
	# 	Callback function executed when the navigation goal reaches a final result.

	# 	Args:
	# 		future (rclpy.Future): The future that is result of the navigation action.
	# 	"""
	# 	status = future.result().status
	# 	# NOTE: Refer https://docs.ros2.org/foxy/api/action_msgs/msg/GoalStatus.html.

	# 	if status == GoalStatus.STATUS_SUCCEEDED:
	# 		self.logger.info("Goal completed successfully!")
	# 		self.detect_shelves(self.simple_map_curr)  # Detect shelves after goal completion.
	# 	else:
	# 		self.logger.warn(f"Goal failed with status: {status}")

	# 	self.goal_completed = True  # Mark goal as completed.
	# 	self.goal_handle_curr = None  # Clear goal handle.


	def goal_result_callback(self, future):
		"""Callback function executed when the navigation goal reaches a final result."""
		status = future.result().status
		
		if status == GoalStatus.STATUS_SUCCEEDED:
			self.logger.info("Goal completed successfully!")
			
			if self.is_exploring:
				# During exploration, detect shelves after each successful goal
				if self.full_map_explored_count >= 8 or self.coverage_percent >= 99.96:
					self.get_logger().info("Exploration complete, starting shelf visits")
					self.next_pose_timer = self.create_timer(1.0, self._delayed_process_next_pose)
					self.detect_shelves(self.simple_map_curr)
					self.is_exploring = False
					self.move_to_shelf()
			else:
				# During shelf visiting, immediately process next pose in queue
				self.get_logger().info(f"Completed {self.shelf_visit_state} pose, processing next pose")
				self.logger.info("Delaying before processing next pose...")
				self.next_pose_timer = self.create_timer(1.0, self._delayed_process_next_pose)

		else:
			self.logger.warn(f"Goal failed with status: {status}")
			
			if not self.is_exploring:
				# If shelf navigation fails, try next pose in queue
				self.get_logger().info("Goal failed, trying next pose in queue")
				self.logger.info("Delaying before processing next pose...")
				self.next_pose_timer = self.create_timer(1.0, self._delayed_process_next_pose)
				# self.process_shelf_pose_queue()

		self.goal_completed = True
		self.goal_handle_curr = None

	def goal_response_callback(self, future):
		"""
		Callback function executed after the goal is sent to the action server.

		Args:
			future (rclpy.Future): The future that is server's response to goal request.
		"""
		goal_handle = future.result()
		if not goal_handle.accepted:
			self.logger.warn('Goal rejected :(')
			self.goal_completed = True  # Mark goal as completed (rejected).
			self.goal_handle_curr = None  # Clear goal handle.
		else:
			self.logger.info('Goal accepted :)')
			self.goal_completed = False  # Mark goal as in progress.
			self.goal_handle_curr = goal_handle  # Store goal handle.

			get_result_future = goal_handle.get_result_async()
			get_result_future.add_done_callback(self.goal_result_callback)

	def goal_feedback_callback(self, msg):
		"""
		Callback function to receive feedback from the navigation action.

		Args:
			msg (nav2_msgs.action.NavigateToPose.Feedback): The feedback message.
		"""
		distance_remaining = msg.feedback.distance_remaining
		number_of_recoveries = msg.feedback.number_of_recoveries
		navigation_time = msg.feedback.navigation_time.sec
		estimated_time_remaining = msg.feedback.estimated_time_remaining.sec

		self.logger.debug(f"Recoveries: {number_of_recoveries}, "
				  f"Navigation time: {navigation_time}s, "
				  f"Distance remaining: {distance_remaining:.2f}, "
				  f"Estimated time remaining: {estimated_time_remaining}s")

		if number_of_recoveries > self.recovery_threshold and not self.cancelling_goal:
			self.logger.warn(f"Cancelling. Recoveries = {number_of_recoveries}.")
			self.cancel_current_goal()  # Unblock by discarding the current goal.

	def send_goal_from_world_pose(self, goal_pose):
		"""
		Sends a navigation goal to the Nav2 action server.

		Args:
			goal_pose (geometry_msgs.msg.PoseStamped): The goal pose in the world frame.

		Returns:
			bool: True if the goal was successfully sent, False otherwise.
		"""
		if not self.goal_completed or self.goal_handle_curr is not None:
			return False

		self.goal_completed = False  # Starting a new goal.

		goal = NavigateToPose.Goal()
		goal.pose = goal_pose

		if not self.action_client.wait_for_server(timeout_sec=SERVER_WAIT_TIMEOUT_SEC):
			self.logger.error('NavigateToPose action server not available!')
			return False

		# Send goal asynchronously (non-blocking).
		goal_future = self.action_client.send_goal_async(goal, self.goal_feedback_callback)
		goal_future.add_done_callback(self.goal_response_callback)

		return True



	def _get_map_conversion_info(self, map_info) -> Optional[Tuple[float, float]]:
		"""Helper function to get map origin and resolution."""
		if map_info:
			origin = map_info.origin
			resolution = map_info.resolution
			return resolution, origin.position.x, origin.position.y
		else:
			return None

	def get_world_coord_from_map_coord(self, map_x: int, map_y: int, map_info) \
					   -> Tuple[float, float]:
		"""Converts map coordinates to world coordinates."""
		if map_info:
			resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
			world_x = (map_x + 0.5) * resolution + origin_x
			world_y = (map_y + 0.5) * resolution + origin_y
			return (world_x, world_y)
		else:
			return (0.0, 0.0)

	def get_map_coord_from_world_coord(self, world_x: float, world_y: float, map_info) \
					   -> Tuple[int, int]:
		"""Converts world coordinates to map coordinates."""
		if map_info:
			resolution, origin_x, origin_y = self._get_map_conversion_info(map_info)
			map_x = int((world_x - origin_x) / resolution)
			map_y = int((world_y - origin_y) / resolution)
			return (map_x, map_y)
		else:
			return (0, 0)

	def _create_quaternion_from_yaw(self, yaw: float) -> Quaternion:
		"""Helper function to create a Quaternion from a yaw angle."""
		cy = math.cos(yaw * 0.5)
		sy = math.sin(yaw * 0.5)
		q = Quaternion()
		q.x = 0.0
		q.y = 0.0
		q.z = sy
		q.w = cy
		return q

	def create_yaw_from_vector(self, dest_x: float, dest_y: float,
				   source_x: float, source_y: float) -> float:
		"""Calculates the yaw angle from a source to a destination point.
			NOTE: This function is independent of the type of map used.

			Input: World coordinates for destination and source.
			Output: Angle (in radians) with respect to x-axis.
		"""
		delta_x = dest_x - source_x
		delta_y = dest_y - source_y
		yaw = math.atan2(delta_y, delta_x)

		return yaw

	def create_goal_from_world_coord(self, world_x: float, world_y: float,
					 yaw: Optional[float] = None) -> PoseStamped:
		"""Creates a goal PoseStamped from world coordinates.
			NOTE: This function is independent of the type of map used.
		"""
		goal_pose = PoseStamped()
		goal_pose.header.stamp = self.get_clock().now().to_msg()
		goal_pose.header.frame_id = self._frame_id

		goal_pose.pose.position.x = world_x
		goal_pose.pose.position.y = world_y

		if yaw is None and self.pose_curr is not None:
			# Calculate yaw from current position to goal position.
			source_x = self.pose_curr.pose.pose.position.x
			source_y = self.pose_curr.pose.pose.position.y
			yaw = self.create_yaw_from_vector(world_x, world_y, source_x, source_y)
		elif yaw is None:
			yaw = 0.0
		else:  # No processing needed; yaw is supplied by the user.
			pass

		goal_pose.pose.orientation = self._create_quaternion_from_yaw(yaw)

		pose = goal_pose.pose.position
		print(f"Goal created: ({pose.x:.2f}, {pose.y:.2f}, yaw={yaw:.2f})")
		return goal_pose

	def create_goal_from_map_coord(self, map_x: int, map_y: int, map_info,
				       yaw: Optional[float] = None) -> PoseStamped:
		"""Creates a goal PoseStamped from map coordinates."""
		world_x, world_y = self.get_world_coord_from_map_coord(map_x, map_y, map_info)

		return self.create_goal_from_world_coord(world_x, world_y, yaw)


def main(args=None):
	rclpy.init(args=args)

	warehouse_explore = WarehouseExplore()

	if PROGRESS_TABLE_GUI:
		gui_thread = threading.Thread(target=run_gui, args=(warehouse_explore.shelf_count,))
		gui_thread.start()

	rclpy.spin(warehouse_explore)

	# Destroy the node explicitly
	# (optional - otherwise it will be done automatically
	# when the garbage collector destroys the node object)
	warehouse_explore.destroy_node()
	rclpy.shutdown()


if __name__ == '__main__':
	main()
