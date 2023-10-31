cmake_minimum_required (VERSION 3.0.2)

# Set flags
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Message out the current compiler flags
message("CMAKE_CXX_FLAGS_DEBUG is ${CMAKE_CXX_FLAGS_DEBUG}")
message("CMAKE_CXX_FLAGS_RELEASE is ${CMAKE_CXX_FLAGS_RELEASE}")

project(mcts_network)

find_package(catkin REQUIRED COMPONENTS
             roscpp
             std_msgs
             cmake_modules
            )
set(CMAKE_PREFIX_PATH "/home/dyros-mk/catkin_ws/src/libtorch")
find_package(Torch REQUIRED)
find_package(Boost REQUIRED COMPONENTS)
find_package(Eigen REQUIRED)

catkin_package(
  CATKIN_DEPENDS 
    roscpp 
    std_msgs
    geometry_msgs
  DEPENDS system_lib
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${TORCH_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
)

set(SOURCE_FILES
  src/net.cpp
  src/replay_buffer.cpp
  src/train.cpp
  src/mcts.cpp
  src/utils.cpp
)

add_executable(mcts_network src/main.cpp ${SOURCE_FILES})
target_link_libraries(mcts_network
  ${catkin_LIBRARIES} 
  ${TORCH_LIBRARIES}
  ${Boost_LIBRARIES}
)

add_executable(test_tensor_basics test/test_tensor_basics.cpp ${SOURCE_FILES})
target_link_libraries(test_tensor_basics
  ${catkin_LIBRARIES} 
  ${TORCH_LIBRARIES}
  ${Boost_LIBRARIES}
)

add_executable(test_network_train test/test_network_train.cpp ${SOURCE_FILES})
target_link_libraries(test_network_train
  ${catkin_LIBRARIES} 
  ${TORCH_LIBRARIES}
  ${Boost_LIBRARIES}
)

add_executable(test_MNIST_train test/test_MNIST_train.cpp ${SOURCE_FILES})
target_link_libraries(test_MNIST_train
  ${catkin_LIBRARIES} 
  ${TORCH_LIBRARIES}
  ${Boost_LIBRARIES}
)

add_executable(test_mcts test/test_mcts.cpp ${SOURCE_FILES})
target_link_libraries(test_mcts
  ${catkin_LIBRARIES} 
  ${TORCH_LIBRARIES}
  ${Boost_LIBRARIES}
)

add_executable(test_replay_buffer test/test_replay_buffer.cpp ${SOURCE_FILES})
target_link_libraries(test_replay_buffer
  ${catkin_LIBRARIES} 
  ${TORCH_LIBRARIES}
  ${Boost_LIBRARIES}
)

add_executable(test_Phi test/test_Phi.cpp ${SOURCE_FILES})
target_link_libraries(test_Phi
  ${catkin_LIBRARIES} 
  ${TORCH_LIBRARIES}
  ${Boost_LIBRARIES}
)