cmake_minimum_required(VERSION 3.5)
project(camaratermica)

# Verificación de librería propietaria
if(NOT EXISTS "/usr/include/libirimager/IRImager.h")
  message(FATAL_ERROR "libirimager not found. Install binary driver before compiling these ROS nodes.")
endif()

# Compilación moderna
add_compile_options(-std=c++17)

# Dependencias
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(std_srvs REQUIRED)
find_package(image_transport REQUIRED)
find_package(camera_info_manager REQUIRED)
find_package(OpenCV REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread)

# Incluir cabeceras
include_directories(
  include
  /usr/include/libirimager/
)

# Nodo
add_executable(publicador_filtrado2_node src/publicador_filtrado2.cpp)

ament_target_dependencies(publicador_filtrado2_node
  rclcpp
  sensor_msgs
  std_msgs
  cv_bridge
  OpenCV
  Boost
)

target_link_libraries(publicador_filtrado2_node
  irdirectsdk
  udev
  pthread
  usb-1.0
  ${OpenCV_LIBRARIES}
  ${Boost_LIBRARIES}
)

# Instalación
install(TARGETS
  publicador_filtrado2_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_export_dependencies(
  rclcpp
  sensor_msgs
  std_msgs
  std_srvs
  image_transport
  camera_info_manager
  cv_bridge
  OpenCV
  Boost
)

ament_package()

