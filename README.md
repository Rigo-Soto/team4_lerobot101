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
model based on the ACT technique.

## Setup
The setup consists of two bases. The first base holds the red column in its initial position, where it can be detected and picked up by the robotic arm. The second base is the target area where the red column will be placed after the robot grabs and releases it.

The camera is positioned to keep both bases within its field of view, allowing the system to detect the object and execute the pick-and-place task.

![SO101 robot setup](readme_images/Setup1.jpeg)

# Problem formulation
The objective of this project is to enable the SO101 robot to autonomously disassemble a LEGO column by identifying, grasping, and relocating it to a designated storage area.

The robot receives visual information from a workspace camera together with its current state, including arm position and gripper status. Based on this information, it must determine the sequence of actions required to complete the task.

The challenge consists of learning a control policy capable of performing the complete manipulation sequence under different object positions and workspace configurations. The desired behavior includes detecting the LEGO column, approaching it, grasping it, removing it from its initial position, transporting it to the storage area, and releasing it inside the designated container.

A task execution is considered successful when the LEGO column is correctly removed from the assembly area and deposited inside the storage container without human intervention.

# Dataset
The dataset was collected using the SO101 leader-follower setup provided by the LeRobot framework. During data collection, an operator manually executed the complete LEGO disassembly task while the system recorded observations and robot actions.

A total of approximately 250 demonstrations were initially collected. After reviewing the recordings, the demonstrations with the highest execution quality and task consistency were selected. The final dataset consists of 50 successful demonstrations covering the complete manipulation sequence from object detection to final placement.

Each demonstration includes:

- RGB camera observations.
- Robot joint state information.
- Gripper state information.
- Action trajectories executed during the task.

The final dataset was uploaded to Hugging Face and is publicly available at:

https://huggingface.co/datasets/emiliano-ng/SO101

# Methodology
The proposed solution combines a perception module and a learning-based control policy.

A YOLO object detection model is used to identify and localize the target LEGO column within the workspace. The detected object information is combined with the robot state to generate the observations used by the policy.

The collected demonstrations are used to train a policy capable of reproducing the behavior observed in the dataset. Training was performed using the LeRobot framework for 10,000 training steps.

Throughout the training process, model performance and training metrics were monitored using Weights & Biases (WandB), allowing continuous tracking of loss values and training progress.

After training, the resulting model is capable of receiving the current observation and predicting the next robot action required to complete the task. The trained model was published on Hugging Face and is available at:

https://huggingface.co/emiliano-ng/SO101_Model

# System Architecture


# Experiments


# Results


# Conclusion

