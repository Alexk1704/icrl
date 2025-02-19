// Copyright 2022 Open Source Robotics Foundation, Inc.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.


#include <gtest/gtest.h>

#include <memory>
#include <thread>

#include "rclcpp/rclcpp.hpp"

#include "ros_gz_interfaces/srv/adjust_physics.hpp"
#include "ros_gz_interfaces/srv/control_world.hpp"
#include "ros_gz_interfaces/srv/delete_entity.hpp"
#include "ros_gz_interfaces/srv/set_entity_pose.hpp"
#include "ros_gz_interfaces/srv/spawn_entity.hpp"

using namespace std::chrono_literals;

/////////////////////////////////////////////////
TEST(ROSClientTest, Physics)
{
  rclcpp::init(0, NULL);
  auto node = std::make_shared<rclcpp::Node>("test_ros_client_to_gz_service");
  auto client = node->create_client<ros_gz_interfaces::srv::AdjustPhysics>(
    "/gz_ros/test/serviceclient/set_physics");
  std::this_thread::sleep_for(1s);
  ASSERT_TRUE(client->wait_for_service(5s));
  auto msg = std::make_shared<ros_gz_interfaces::srv::AdjustPhysics::Request>();
  auto future = client->async_send_request(msg);
  rclcpp::executors::SingleThreadedExecutor ex;
  ex.add_node(node);
  ex.spin_until_future_complete(future);
  auto res = future.get();
  ASSERT_TRUE(res->success);
}

TEST(ROSClientTest, WorldControl)
{
  rclcpp::init(0, NULL);
  auto node = std::make_shared<rclcpp::Node>("test_ros_client_to_gz_service");
  auto client = node->create_client<ros_gz_interfaces::srv::ControlWorld>(
    "/gz_ros/test/serviceclient/world_control");
  std::this_thread::sleep_for(1s);
  ASSERT_TRUE(client->wait_for_service(5s));
  auto msg = std::make_shared<ros_gz_interfaces::srv::ControlWorld::Request>();
  auto future = client->async_send_request(msg);
  rclcpp::executors::SingleThreadedExecutor ex;
  ex.add_node(node);
  ex.spin_until_future_complete(future);
  auto res = future.get();
  ASSERT_TRUE(res->success);
}

TEST(ROSClientTest, EntityFactory)
{
  rclcpp::init(0, NULL);
  auto node = std::make_shared<rclcpp::Node>("test_ros_client_to_gz_service");
  auto client = node->create_client<ros_gz_interfaces::srv::SpawnEntity>(
    "/gz_ros/test/serviceclient/create");
  std::this_thread::sleep_for(1s);
  ASSERT_TRUE(client->wait_for_service(5s));
  auto msg = std::make_shared<ros_gz_interfaces::srv::SpawnEntity::Request>();
  auto future = client->async_send_request(msg);
  rclcpp::executors::SingleThreadedExecutor ex;
  ex.add_node(node);
  ex.spin_until_future_complete(future);
  auto res = future.get();
  ASSERT_TRUE(res->success);
}

TEST(ROSClientTest, Entity)
{
  rclcpp::init(0, NULL);
  auto node = std::make_shared<rclcpp::Node>("test_ros_client_to_gz_service");
  auto client = node->create_client<ros_gz_interfaces::srv::DeleteEntity>(
    "/gz_ros/test/serviceclient/remove");
  std::this_thread::sleep_for(1s);
  ASSERT_TRUE(client->wait_for_service(5s));
  auto msg = std::make_shared<ros_gz_interfaces::srv::DeleteEntity::Request>();
  auto future = client->async_send_request(msg);
  rclcpp::executors::SingleThreadedExecutor ex;
  ex.add_node(node);
  ex.spin_until_future_complete(future);
  auto res = future.get();
  ASSERT_TRUE(res->success);
}

TEST(ROSClientTest, Pose)
{
  rclcpp::init(0, NULL);
  auto node = std::make_shared<rclcpp::Node>("test_ros_client_to_gz_service");
  auto client = node->create_client<ros_gz_interfaces::srv::SetEntityPose>(
    "/gz_ros/test/serviceclient/set_pose");
  std::this_thread::sleep_for(1s);
  ASSERT_TRUE(client->wait_for_service(5s));
  auto msg = std::make_shared<ros_gz_interfaces::srv::SetEntityPose::Request>();
  auto future = client->async_send_request(msg);
  rclcpp::executors::SingleThreadedExecutor ex;
  ex.add_node(node);
  ex.spin_until_future_complete(future);
  auto res = future.get();
  ASSERT_TRUE(res->success);
}

/////////////////////////////////////////////////
int main(int argc, char ** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
