// === Nodo ROS2 en C++ que publica imágenes filtradas por temperatura ===

#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include "libirimager/direct_binding.h"

class PublicadorFiltrado : public rclcpp::Node {
public:
  PublicadorFiltrado(const std::string &config_file)
      : Node("publicador_filtrado") {
    image_pub_ = this->create_publisher<sensor_msgs::msg::Image>("/imagen_termica_filtrada", 10);

    if (::evo_irimager_usb_init(config_file.c_str(), 0, 0) != 0) {
      RCLCPP_ERROR(this->get_logger(), "Error al inicializar la cámara.");
      rclcpp::shutdown();
      return;
    }

    if (::evo_irimager_get_thermal_image_size(&t_w_, &t_h_) != 0) {
      RCLCPP_ERROR(this->get_logger(), "No se pudo obtener tamaño de imagen térmica.");
      rclcpp::shutdown();
      return;
    }

    thermal_data_.resize(t_w_ * t_h_);
    timer_ = this->create_wall_timer(
        std::chrono::milliseconds(8),  // ~128 Hz
        std::bind(&PublicadorFiltrado::publishImage, this));
  }

  ~PublicadorFiltrado() {
    ::evo_irimager_terminate();
  }

private:
  void publishImage() {
    int err = ::evo_irimager_get_thermal_image(&t_w_, &t_h_, thermal_data_.data());
    if (err != 0) {
      RCLCPP_WARN(this->get_logger(), "No se pudo capturar imagen térmica: %d", err);
      return;
    }

    // Crear imagen en escala de grises con mayor contraste: blanco a partir de 36 ºC
    cv::Mat gray_image(cv::Size(t_w_, t_h_), CV_8UC1);

    for (int y = 0; y < t_h_; y++) {
      for (int x = 0; x < t_w_; x++) {
        unsigned short temp_raw = thermal_data_[y * t_w_ + x];
        float temp_celsius = temp_raw / 10.0f - 100.0f;

        // Limitar el rango entre 29 y 36 ºC (Modo Verano)
        float clamped = std::min(std::max(temp_celsius, 29.0f), 36.0f);

        // Rango visual: 30 ºC = negro (0), 38 ºC = blanco (255)
        float scale;
        if (clamped <= 36.0f) {
          scale = (clamped - 29.0f) / (36.0f - 29.0f);  // escala de 0 a 1
        } else {
          scale = 1.0f;  // cualquier valor > 36 será blanco
        }

        uint8_t gray_value = static_cast<uint8_t>(255.0f * scale);
        gray_image.at<uint8_t>(y, x) = gray_value;
      }
    }

    // Convertir a 3 canales para publicar como 'bgr8'
    cv::Mat bgr_image;
    cv::cvtColor(gray_image, bgr_image, cv::COLOR_GRAY2BGR);

    // Publicar imagen
    auto msg = cv_bridge::CvImage(std_msgs::msg::Header(), "bgr8", bgr_image).toImageMsg();
    image_pub_->publish(*msg);
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::TimerBase::SharedPtr timer_;
  int t_w_, t_h_;
  std::vector<unsigned short> thermal_data_;
};

int main(int argc, char **argv) {
  rclcpp::init(argc, argv);

  if (argc != 2) {
    std::cerr << "Uso: ros2 run optris_drivers publicador_filtrado_node <config.xml>" << std::endl;
    return -1;
  }

  auto node = std::make_shared<PublicadorFiltrado>(argv[1]);
  rclcpp::spin(node);
  rclcpp::shutdown();
  return 0;
}
