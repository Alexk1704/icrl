# Intentional Continual Reinforcement Learning (ICRL)

This framework is written to facilitate research in the field of Continual Reinforcement Learning (CRL) based on a robotic simulation running with Gazebo and ROS2. The reinforcement learning policy is defined by a high-level package written in Python, supporting the use of common ML frameworks such as TensorFlow and PyTorch. We use a differential-drive robot ([Pololu 3pi robot](https://www.pololu.com/product/975)) to investigate the line following scenario, but it is easy to extend it to other simulation scenarios.

This library was used to generate the empirical data for two accepted conference articles:

1) ["Continual Reinforcement Learning Without Replay Buffers"](https://ieeexplore.ieee.org/abstract/document/10705256)
```
@inproceedings{krawczyk2024continual,
  title={Continual Reinforcement Learning Without Replay Buffers},
  author={Krawczyk, Alexander and Bagus, Benedikt and Denker, Yannick and Gepperth, Alexander},
  booktitle={2024 IEEE 12th International Conference on Intelligent Systems (IS)},
  pages={1--9},
  year={2024},
  organization={IEEE}
}
```
2) ["Informative Performance Measures for Continual Reinforcement Learning"](https://ieeexplore.ieee.org/abstract/document/10793039)
```
@inproceedings{denker2024informative,
  title={Informative Performance Measures for Continual Reinforcement Learning},
  author={Denker, Yannick and Bagus, Benedikt and Krawczyk, Alexander and Gepperth, Alexander},
  booktitle={2024 IEEE 20th International Conference on Intelligent Computer Communication and Processing (ICCP)},
  pages={1--6},
  year={2024},
  organization={IEEE}
}
```

## Main components description

### Package `"./workspace/base/*"` (Written and maintained by Benedikt Bagus)
This package allows communication between the Gazebo Simulator and ROS2. It contains all the necessary interfaces to allow data exchange between the two. It is first compiled and built to be deployed on the running system, whether it is a local node or distributed via the workload manager (Slurm/Singularity in our case).
1) `ros_gz_bridge` \
This component provides the network bridge that allows messages to be exchanged between ROS2 and Gazebo Transport.
1) `ros_gz_interfaces` \
Defines the message and service data structures to allow an interaction between ROS2 and Gazebo.
2) `ros_gz_sim` \
Contains various utilities for conveniently integrating ROS2 into Gazebo, such as preparing launch files and providing ROS-enabled executables.

### Package `"./eval/*` (Written and maintained by Benedikt Bagus, Alexander Krawczyk)
This package is currently in an incomplete state as we have started a major rework of the API. The basic principles behind these components are to allow generic evaluation for all kinds of environments and scenarios, regardless of the simulation or policy being used. Besides providing the obvious plot formats, such as scatter, bar, violing and pie plots, we also considered the possibility of visualizing the robot's trajectory, showing the distribution of certain states and actions as a heat map. This could speed up debugging without the need to observe the live simulation and facilitate numerical investigation of real-time odometry data.

### Package `"./workspace/devel/*"` (Written and maintained by Benedikt Bagus, Alexander Krawczyk)
The `gazebo_sim` package contains the "client-side" source code, i.e., starting the simulation, spawning and controlling the robotic agent, training the RL policy, evaluation, serialization of metrics, etc.
1) `/simulation/LineFollowing.py`: This defines and executes the **main control loop** for the specified "Line Following (LF)" scenario. It keeps the environment (a running gazebo simulation) and the operating agent (a 3Pi robot) synchronized by "stepping" through the environment. Each environment defines the track layout, sensory inputs, and possible actions. The goal for the robotic agent is to continually improve its policy by learning from rewards (Q-Learning). This loop allows the agent to learn an optimal policy for the line following track efficiently. Note, however, that this is easily adaptable to other types of RL policies, e.g. Policy Gradient (PG).
2) `/agent/LFAgent.py`: Represents the robotic agent that interacts with the simulation environment by selectively performing actions. It also acts as an interface to the RL policy, the machine learning algorithm running in the background, which is trained and evaluated.
3) `./workspace/devel/gazebo_sim/agent/LFLearner.py`: This defines the RL policy, i.e. Deep Q-Learning (DQN) or QGMM. 
4) `./workspace/devel/gazebo_sim/model/*.py`: This defines and builds a concrete ML model that represents the trainable part of an RL policy. Currently we support two types of models: A Deep Neural Network (DNN) for DQN, and a Gaussian Mixture Model (GMM) for QGMM. We use the [**sccl**](https://github.com/Alexk1704/scclv2) package to create and run a functional TensorFlow model.
5) `./workspace/devel/gazebo_sim/utils/*`: Provides some useful classes, such as buffer structures (for different types of experience replay), threading, an argument parser, data serialization, logging, and so on.

### Package `"./scripts/*` (Written and maintained by Benedikt Bagus)
`wrapper.bash`: This is the main invocation script for local users or to enable deployment via the Workload Manager.

### Run instructions
Please follow the instructions in `./setup.bash` to set up your machine accordingly.

***~ There is no free lunch (: ~***

### TODOs
* Robot tends to roll back slightly after braking with high acceleration, driving seems a bit "clumsy" due to high acceleration
* Traction issues? Currently, the wheels have a higher mass, but does this cause any other relevant side effects?
* Test new PER buffer
* Add the ability to adjust simulation speed via RTF -> may speed up code execution!
* How do we reduce epsilon delta on each task reset?
  * Back to initial value or linear reduction?
* Random spawn positions
  * Seems to be a bit more complicated and does not really comply with the concept of concept shifts induced by changing racetracks
* The choice of track order should show the incremental evolution of learning in terms of robot control.
  * Learn forward, learn to drive left turns, then learn to drive right turns