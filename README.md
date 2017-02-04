# as_urobotiq

<div align="center">
  <img src="http://ancorasir.com/wp-content/uploads/2017/01/URobotiq-ur5a3f-DefaultScene.jpg"><br><br>
</div>

[as_urobotiq](https://github.com/ancorasir/as_urobotiq) is intended to be setup as a general purpose robot platform for autonomous pick-and-place with deep learning. The current robot setup includes a [UR5 arm from Universal Robot](https://www.universal-robots.com/products/ur5-robot/), an [Adaptive 3 Finger gripper from Robotiq](http://robotiq.com/products/industrial-robot-hand/), a [Kinect for X-box One camera from Microsoft](http://www.xbox.com/en-US/xbox-one/accessories/kinect), an [Xtion Pro camera from ASUS](https://www.asus.com/3D-Sensor/Xtion_PRO/), and an [FT300 sensor from Robotiq](http://robotiq.com/products/robotics-force-torque-sensor/), all controlled by a high performance desktop computer with [NVIDIA TITAN X Graphics Card](https://www.nvidia.com/en-us/geforce/products/10series/titan-x-pascal/) running [Tensorflow](https://www.tensorflow.org/) to autonomously learn object grasping using Deep Learning.

This is a project under-development at the Sustainable and Intelligent Robotics ([SIR](http://ancorasir.com)) Group , under the supervision of Dr. Chaoyang Song.

Song, C., 2017, as_urobotiq: A general purpose robot for autonomous pick-and-place with deep learning. Github.com/ancorasir/as_urobotiq, [![DOI](https://zenodo.org/badge/80539341.svg)](https://zenodo.org/badge/latestdoi/80539341)


# Installation

Start a terminal, run the following codes, then restart terminal and done.

```shell
mkdir ~/ancorasir_ws/src
catkin_init_workspace
cd ..
catkin_make

cd src
git clone -b indigo https://github.com/ros-industrial/universal_robot.git
git clone -b indigo-devel https://github.com/ros-industrial/robotiq.git
git clone -b ros-indigo https://github.com/ancorasir/as_urobotiq.git

rosdep install --from-paths ~/ancorasir_ws/src/ --ignore-src

cd ..
catkin_make
echo "source ~/ancorasir_ws/devel/setup.bash" >> ~/.bashrc
```

# Simulation

## Description pkg Test Launch

```shell
roslaunch urobotiq_description urobotiq_view.launch
```

## Gazebo pkg Test Launch

```shell
roslaunch urobotiq_gazebo urobotiq.launch
```

To test the gripper open simulation

```shell
rostopic pub --once left_hand/command robotiq_s_model_articulated_msgs/SModelRobotOutput {1,2,1,0,0,0,0,255,0,155,0,0,255,0,0,0,0,0}
```

To test the gripper close simulation

```shell
rostopic pub --once left_hand/command robotiq_s_model_articulated_msgs/SModelRobotOutput {1,1,1,0,0,0,255,255,0,155,0,0,255,0,0,0,0,0}
```

[Reference to Robotiq adaptive 3 finger gripper control command](http://support.robotiq.com/display/IMB/4.6+Control+logic+-+example).


## Moveit pkg Test Launch

### Just the robot arm
Test with just the robot arm, you can run the following code:

```shell
roslaunch urobotiq_ur5_moveit_config demo.launch
```

Or, you can do the following, which enables you to do path planning in moveit and execute results in gazebo.

```shell
roslaunch urobotiq_gazebo urobotiq_ur5.launch

roslaunch urobotiq_ur5_moveit_config urobotiq_ur5_moveit_planning_execution.launch sim:=true

roslaunch urobotiq_ur5_moveit_config moveit_rviz.launch config:=true
```
### Just the robot gripper (working)



### The whole robot system (working)
Test with just the whole robot system, you can run the following code:

```shell
roslaunch urobotiq_moveit_config demo.launch
```

# Hardware (working)

## Decomposed Integrated Task Flows (ITFs)

### ITF-RaMo: Rand => Move => Shot (working)
...

### ITF-SoMo: Shot 00 => Move 00 => Shot 01 (working)
...

### ITF-Serv: CEMs => Move x => Shot x (working)
...

### ITF-CEMs: CEMs = CEM1 => CEM2 => CEM 3 (working)
...

### ITF-Pick: Pick => Shot x1 => Move => Shot x2 => Drop => Shot x3 (working)
...

### ITF-Hand: Take => Move => Shot (working)
...
