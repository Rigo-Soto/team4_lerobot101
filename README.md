# Development Team
- Arturo Balboa
- Oscar de la Rosa
- Angel Hernandez
- Emiliano Niño
- Rigoberto Soto


# Building

pip install git+https://github.com/huggingface/lerobot@1396b9fab7aecddd10006c33c47a487ffdcb54b4


# Introduction: LEGO disassembler
Disarm a column of legos, store it in a designated area

Using the lerobot library and a SO101 pair of leader and follower, this repository contains the 
neccessary environment for the development and training of a Inverse Reinforcement Learning
model based on the _(insert final decision here)_ technique.

## Setup
The setup consists of two bases. The first base holds the red column in its initial position, where it can be detected and picked up by the robotic arm. The second base is the target area where the red column will be placed after the robot grabs and releases it.

The camera is positioned to keep both bases within its field of view, allowing the system to detect the object and execute the pick-and-place task.

![SO101 robot setup](readme_images/Setup1.jpeg)

# Problem formulation
The objective of this project is to enable the SO101 robot to autonomously disassemble a LEGO column by identifying, grasping, and relocating it to a designated storage area.

The robot receives visual information from a workspace camera together with its current state, including arm position and gripper status. Based on this information, it must decide the next movement required to complete the task.

A dataset of demonstrations is collected using the leader-follower setup. Each demonstration contains a sequence of observations and the corresponding actions performed by the operator while completing the LEGO disassembly task.

The goal is to learn a control policy capable of reproducing the demonstrated behavior and generalizing to different object positions and scene configurations. The learned policy should allow the robot to detect the LEGO column, approach it, grasp it, remove it from its initial position, transport it to the storage area, and release it inside the designated container.

A task execution is considered successful when the LEGO column is correctly removed from the assembly area and deposited inside the storage container without human intervention.

# Dataset


# Methodology
The proposed solution combines a perception module and a learning-based control policy.

A YOLO object detection model is used to identify and localize the target LEGO column within the workspace. The detected object information is combined with the robot state to generate the observations used by the control policy.

The policy is trained from demonstrations collected using the SO101 leader-follower setup. During data collection, an operator performs the complete task while observations and actions are recorded.

After training, the learned policy receives the current observation and predicts the corresponding robot action, allowing the system to autonomously approach, grasp, transport, and store the LEGO column.

# System Architecture


# Experiments


# Results


# Conclusion

