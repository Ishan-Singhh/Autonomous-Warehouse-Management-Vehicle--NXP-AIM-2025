# Autonomous Warehouse Management Vehicle 🤖📦
**NXP AIM India 2025 Challenge**

This repository contains the ROS 2 workspace and source code for an autonomous warehouse management rover (B3RB). The system is designed to navigate a simulated warehouse environment, dynamically discover shelves, decode QR codes for heuristic routing, and perform real-time object recognition using a localized AI model.

## 🌟 Key Features

*   **Heuristic-Based Autonomous Navigation:** Replaces standard random frontier exploration with a directed search. The rover extracts angular heuristics from decoded QR codes to calculate the precise trajectory to the next hidden shelf.
*   **Intelligent Shelf Detection:** Utilizes DBSCAN clustering, Convex Hull, and Principal Component Analysis (PCA) on the SLAM occupancy grid to identify shelves matching exact physical dimensions.
*   **Computer Vision & Object Recognition:** Integrates a YOLOv5-int8 TFLite model to run lightweight, real-time object detection.
*   **Coordinate-Based Validation:** Implements a dynamic memory system using a 1-meter radius check to prevent the rover from getting stuck in loops or rescanning previously visited shelves.
*   **Automated State Machine:** Seamlessly handles complex multi-step navigation queues (e.g., `MOVING_TO_SHELF` → `SCANNING_OBJECTS` → `SCANNING_QR`).
*   **Live GUI Tracker:** Features a Tkinter-based responsive interface that runs concurrently to visualize shelf discoveries, scanned objects, and decoded QR strings in real-time.

---

## 🎥 Simulation Videos & Demos

Watch the B3RB rover in action! You can view full simulation recordings, including autonomous navigation, obstacle avoidance, and object/QR scanning here:

🔗 **[View Simulation Videos on Google Drive](https://drive.google.com/drive/folders/1DVH4kJL5a2vGV8RFFW7pIKMmXdg-TCHh?usp=sharing)**

---

## 🚀 Phase-Wise Development Journey

Our approach to solving the NXP AIM warehouse challenge evolved through several distinct phases of optimization and logic refinement:

### Phase 1: Core Navigation & Parameter Tuning
*   **Initial Setup:** Configured the ROS 2 Nav2 stack and established basic manual and autonomous movement capabilities.
*   **Costmap Optimization:** Tuned the `nav2.yaml` inflation radius to exactly 0.5m to match the buggy's inscribed footprint, resolving critical costmap collision errors that were paralyzing the planner.
*   **Exploration Baseline:** Implemented a standard frontier-based exploration to map unknown areas.

### Phase 2: Directed Heuristic Navigation
*   **Abandoning Blind Search:** Transitioned from random frontier exploration to a targeted heuristic approach. 
*   **QR Integration:** Programmed the system to start moving at an `initial_angle` and dynamically update its search vector based on the angle encoded in the decoded QR strings.

### Phase 3: Mathematical Shelf Detection
*   **Grid Analysis:** Instead of relying purely on vision to find shelves, we utilized the `/map` occupancy grid.
*   **PCA & Clustering:** Applied DBSCAN clustering to group obstacle cells, followed by Principal Component Analysis (PCA) and Convex Hull to filter out random obstacles and accurately identify structures matching the exact 1.35m x 0.5m shelf dimensions.

### Phase 4: Pose Queues & State Machine
*   **Structured Scanning:** Created a queue processing system to generate optimal viewing poses for the rover: perpendicular poses for YOLO object detection and parallel poses for QR scanning.
*   **State Transitions:** Overhauled the `goal_result_callback` to instantly transition between states (`SCANNING_OBJECTS` → `SCANNING_QR`) without relying on brittle timeouts, ensuring fast and continuous scanning.

### Phase 5: Coordinate-Based Memory (Current)
*   **Solving the Infinite Loop:** Replaced the initial heuristic-based shelf validation with a robust coordinate tracking system. 
*   **Radius Thresholding:** The buggy now stores the `(x, y)` coordinates of completed shelves. Before scanning a new detection, it performs a 1-meter radius check (`self.scanned_shelf_coords`) to guarantee it never wastes time rescanning a completed shelf, even if SLAM map coordinates shift slightly.

---

## 🛠️ System Architecture

### 1. `b3rb_ros_warehouse.py` (Core Controller)
The master node that acts as the brain of the rover. 
*   Acts as a `NavigateToPose` action client for the Nav2 stack.
*   Processes `/map` and `/global_costmap/costmap` data for spatial awareness.
*   Generates optimal viewing poses (parallel for QR codes, perpendicular for objects).
*   Handles recovery behaviors and goal cancellations if the rover encounters unmapped obstacles.

### 2. `b3rb_ros_object_recog.py` (Vision Node)
The dedicated perception node.
*   Subscribes to `/camera/image_raw/compressed`.
*   Uses `tflite_runtime` for accelerated on-device neural network inference.
*   Applies Non-Maximum Suppression (NMS) to filter redundant bounding boxes.
*   Publishes finalized object counts to the `/shelf_objects` topic to unlock subsequent warehouse zones.

### 3. `nav2.yaml` (Tuned Navigation Parameters)
*   **Planner:** Utilizes `SmacPlannerHybrid` with `REEDS_SHEPP` motion models for smooth, continuous curves suitable for Ackermann steering.

---

## 📦 Prerequisites and Dependencies

To run this package, ensure you have the following installed in your ROS 2 environment:

*   **ROS 2** (Foxy / Humble)
*   **Python 3.x**
*   **OpenCV** (`cv2`) for image processing and QR decoding.
*   **TensorFlow Lite Runtime** (`tflite_runtime`) for the YOLOv5 model.
*   **SciPy & Scikit-learn** for spatial clustering and PCA calculations.
*   **Tkinter** for the live progress GUI.

## ⚙️ Setup and Installation

1. Clone this repository into the `src` directory of your ROS 2 workspace:
   ```bash
   cd ~/ros2_ws/src
   git clone [https://github.com/Ishan-Singhh/Autonomous-Warehouse-Management-Vehicle--NXP-AIM-2025.git](https://github.com/Ishan-Singhh/Autonomous-Warehouse-Management-Vehicle--NXP-AIM-2025.git)

```

2. Build the workspace:
```bash
cd ~/ros2_ws
colcon build --symlink-install

```


3. Source the setup file:
```bash
source install/setup.bash

```



## 🏁 How to Run

1. Launch the NXP AIM 2025 simulation environment and the B3RB bringup nodes as per the challenge instructions.
2. Run the object recognition node:
```bash
ros2 run b3rb_ros_aim_india object_recognizer

```


3. In a separate terminal, launch the warehouse exploration controller:
```bash
ros2 run b3rb_ros_aim_india warehouse_explore --ros-args -p shelf_count:=<NUMBER_OF_SHELVES> -p initial_angle:=<STARTING_ANGLE>

```



*Note: The GUI will automatically launch in a separate thread once the exploration node starts.*

---
