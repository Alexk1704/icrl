#!/bin/bash

#-------------------------------------------------------------------------------------------------- SYSTEM

# use X11 window system instead wayland, because of gazebo and nvidia issues
sudo sed -i 's/#WaylandEnable=false/WaylandEnable=false/g' /etc/gdm3/custom.conf

# update the ubuntu distro to get the latest versions
sudo apt-get update && sudo apt-get upgrade -y && sudo apt-get dist-upgrade -y && sudo apt-get autoremove -y && sudo apt-get autoclean
sudo reboot now

# install some essential packages if missing
sudo apt-get install -y openssh-server wakeonlan
sudo apt-get install -y nano curl git gnupg 
sudo apt-get install -y htop nv-top neofetch
sudo apt-get install -y xvfb mesa-utils lsb-release
sudo apt-get install -y antimicrox terminator

# update and activate firewall
sudo ufw allow ssh ; sudo ufw enable && sudo ufw reload

# adding some aliases
echo -e "\n# create some update aliases" >> ~/.bashrc
echo "alias update-firmware='sudo fwupdmgr refresh && sudo fwupdmgr update'" >> ~/.bashrc
echo "alias update-apt='sudo apt-get update && sudo apt-get upgrade -y && sudo apt-get dist-upgrade -y && sudo apt-get autoremove -y && sudo apt-get autoclean'" >> ~/.bashrc
echo "alias update-snap='sudo snap refresh'" >> ~/.bashrc
echo "alias update-pip='sudo pip3 install -U $(sudo pip3 list --outdated | awk '\\''NR >= 3 { print $1 }'\\'') ; pip3 install -U $(pip3 list --outdated | awk '\\''NR >= 3 { print $1 }'\\'')'" >> ~/.bashrc

source ~/.bashrc

#-------------------------------------------------------------------------------------------------- PYTHON

sudo apt-get install -y python-is-python3 libpython3-dev python3-pip

# prevent the pip building errors of sdist (vs wheel)
# python -m pip install -I instead python -m pip install -U
sudo apt-get install -y \
 libgirepository1.0-dev \
 libcairo2-dev \
 libcups2-dev \
 libsnappy-dev \
 librsync-dev \
 gettext

#-------------------------------------------------------------------------------------------------- TENSORFLOW

# add keys and repos
curl -sSL "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub" | gpg --dearmor | sudo tee /usr/share/keyrings/nvidia_cuda-archive-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia_cuda-archive-keyring.gpg] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" | sudo tee -a /etc/apt/sources.list.d/nvidia_cuda.list

sudo apt-get update

# install the nvidia driver
sudo apt-get install -y --no-install-recommends nvidia-driver-550
sudo reboot now

# install the cuda driver
sudo apt-get install -y --no-install-recommends cuda-11-8

# install nvidia cudnn
sudo apt-get install -y --no-install-recommends \
 libcudnn8=8.6.0.163-1+cuda11.8 \
 libcudnn8-dev=8.6.0.163-1+cuda11.8

# install nvidia tensorrt
sudo apt-get install -y --no-install-recommends \
 libnvinfer8=8.5.3-1+cuda11.8 \
 libnvinfer-plugin8=8.5.3-1+cuda11.8 \
 libnvinfer-dev=8.5.3-1+cuda11.8 \
 libnvinfer-plugin-dev=8.5.3-1+cuda11.8

# prevent autoupdate for some packages
sudo apt-mark hold libcudnn8 libcudnn8-dev libnvinfer8 libnvinfer-plugin8 libnvinfer-dev libnvinfer-plugin-dev
sudo reboot now

#-------------------------------------------------------------------------------------------------- PIP

# install essential packages
pip3 install -U scipy numpy matplotlib scikit-learn tensorflow torch
pip3 install -U numba opencv-python qpsolvers pyyaml pyparse
pip3 install -U black pylint pytype pytest pydbg pyqt6

#-------------------------------------------------------------------------------------------------- ROS

# check for UTF-8
locale

# optinal part
sudo apt-get update && sudo apt-get install -y locales
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8

# verify settings
locale

# check if universe repo is enabled
apt-cache policy | grep universe

# if not, install and enable
sudo apt-get install software-properties-common
sudo add-apt-repository universe

# add key and repo
sudo curl -sSL "https://raw.githubusercontent.com/ros/rosdistro/master/ros.key" -o /usr/share/keyrings/osrf_ros-archive-keyring.gpg
sudo echo "deb [signed-by=/usr/share/keyrings/osrf_ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu jammy main" >> /etc/apt/sources.list.d/osrf_ros.list

sudo apt-get update

# install ros2
sudo apt-get install -y ros-humble-desktop

# install rqt packages via the wrapper package
# ~n matches package name with regex
sudo apt-get install -y ~nros-humble-rqt*

# install dev-tools
sudo apt-get install -y ros-dev-tools

# install python extentions
sudo apt-get install -y \
 python3-colcon-common-extensions \
 python3-osrf-pycommon \
 python3-rosdep

# source the distro script (underlay)
echo -e "\n# source the ROS2 backend" >> ~/.bashrc
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# export domain id [0, 100]
echo -e "\n# export the ROS2 domain" >> ~/.bashrc
echo "export ROS_DOMAIN_ID=0" >> ~/.bashrc

# export localonly constraint (prevent network-wide communication)
echo -e "\n# export the ROS2 localhost" >> ~/.bashrc
echo "export ROS_LOCALHOST_ONLY=1" >> ~/.bashrc

# export a specific implementation of the middleware
echo -e "\n# export the ROS2 middleware" >> ~/.bashrc
echo "export RMW_IMPLEMENTATION=rmw_fastrtps_cpp" >> ~/.bashrc

source ~/.bashrc

# source the colcon_cd command
echo -e "\n# source the colcon_cd command" >> ~/.bashrc
echo "source /usr/share/colcon_cd/function/colcon_cd.sh" >> ~/.bashrc
echo "export _colcon_cd_root=/opt/ros/humble/" >> ~/.bashrc

# source argcomplete functionality
echo -e "\n# source the argcomplete functionality" >> ~/.bashrc
echo "source /usr/share/colcon_argcomplete/hook/colcon-argcomplete.bash" >> ~/.bashrc

source ~/.bashrc

# init and update the ros dependencies
sudo rosdep init && rosdep update

# create workspace directory
mkdir -p ~/ros2_ws/src/

# close the terminal and open a new one
# install dependencies
cd ~/ros2_ws/ && rosdep install -i --from-path src --rosdistro humble -y

# build and source the package (overlay)
cd ~/ros2_ws/ && colcon build --symlink-install
cd ~/ros2_ws/ && source ./install/local_setup.bash

# print ROS environment variables (exports)
printenv | grep -i ROS

#-------------------------------------------------------------------------------------------------- GAZEBO

# add key and repo
sudo curl -sSL "https://packages.osrfoundation.org/gazebo.gpg" -o /usr/share/keyrings/osrf_gazebo-archive-keyring.gpg
sudo echo "deb [signed-by=/usr/share/keyrings/osrf_gazebo-archive-keyring.gpg] http://packages.osrfoundation.org/gazebo/ubuntu-stable jammy main" >> /etc/apt/sources.list.d/osrf_gazebo.list

sudo apt-get update

# install ignition
sudo apt-get install -y ignition-fortress

# install ros ignition bridge
sudo apt-get install -y ros-humble-ros-ign

# export the ignition version info
echo -e "\n# export the ignition version" >> ~/.bashrc
echo "export IGN_VERSION=6" >> ~/.bashrc

# export the ignition distro info
echo -e "\n# export the ignition version" >> ~/.bashrc
echo "export IGN_DISTRO=fortress" >> ~/.bashrc

source ~/.bashrc

# check further environment variables
# echo $IGN_GAZEBO_RESOURCE_PATH
# echo $IGN_GAZEBO_SYSTEM_PLUGIN_PATH