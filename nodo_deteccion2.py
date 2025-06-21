#!/usr/bin/env python3

# Librerías necesarias para ROS2, OpenCV, MediaPipe, FFT, etc.
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import mediapipe as mp
import time
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks, savgol_filter
from scipy.fft import fft, fftfreq
import os
import csv

# Configuración de MediaPipe para malla facial
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

# Índices de landmarks usados para nariz y boca
LANDMARKS_NOSE = [60, 290]
LANDMARKS_MOUTH = [13]

bridge = CvBridge()

class FaceDetectorNode(Node):
    def __init__(self):
        super().__init__('nodo_deteccion')

        # Desplazamientos para ajustar el ROI respecto a los landmarks
        self.offset_nose_y = 0
        self.offset_mouth_y = 6
        self.offset_nose_x = 0
        self.offset_mouth_x = 0

        # Definición de publishers y subscribers
        self.sub = self.create_subscription(Image, '/imagen_termica_filtrada', self.image_callback, 10)
        self.pub = self.create_publisher(Float32, 'breathing/source2', 10)
        self.pub_rate = self.create_publisher(Float32, '/respiracion/frecuencia_30s', 10)


        # Configuración del detector facial MediaPipe
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5)

        # Variables de estado y almacenamiento de señales
        self.analysis_active = False
        self.frame_count = 0
        self.signal_nose = []
        self.signal_mouth = []
        self.signal_timestamps = []
        self.fixed_rects_nose = []
        self.fixed_rects_mouth = []
        self.temp_rects_nose = []
        self.temp_rects_mouth = []
        self.rect_size = 5
        self.analysis_ready = False
        self.start_time = None

        
    def image_callback(self, msg):
        current_time = time.time()

        # Limita la frecuencia de muestreo a 8 Hz
        if self.analysis_active:
            if not hasattr(self, 'last_sample_time'):
                self.last_sample_time = current_time
            if current_time - self.last_sample_time < 0.125:
                return
            self.last_sample_time = current_time
        
        # Conversión del mensaje de imagen ROS2 a formato OpenCV
        frame = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        if frame is None:
            self.get_logger().error("Imagen vacía")
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        # Detección de landmarks faciales y generación de ROIs temporales
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                self.temp_rects_nose = []
                self.temp_rects_mouth = []

                for idx in LANDMARKS_NOSE + LANDMARKS_MOUTH:
                    landmark = face_landmarks.landmark[idx]
                    x = int(landmark.x * frame.shape[1])
                    if idx in LANDMARKS_MOUTH:
                        x += self.offset_mouth_x
                    else:
                        x += self.offset_nose_x
                    if idx in LANDMARKS_MOUTH:
                        y = int(landmark.y * frame.shape[0]) + self.offset_mouth_y
                    else:
                        y = int(landmark.y * frame.shape[0]) + self.offset_nose_y
                    cv2.circle(frame, (x, y), 3, (0, 255, 255), -1)

                    x0 = max(0, x - self.rect_size)
                    x1 = min(frame.shape[1], x + self.rect_size)
                    y0 = max(0, y - self.rect_size)
                    y1 = min(frame.shape[0], y + self.rect_size)
                    cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 0), 1)

                    if idx in LANDMARKS_NOSE:
                        self.temp_rects_nose.append((x0, x1, y0, y1))
                    elif idx in LANDMARKS_MOUTH:
                        self.temp_rects_mouth.append((x0, x1, y0, y1))

                # Fijar los rectángulos al comienzo del análisis
                if self.analysis_active and not self.fixed_rects_nose and not self.fixed_rects_mouth:
                    self.fixed_rects_nose = list(self.temp_rects_nose)
                    self.fixed_rects_mouth = list(self.temp_rects_mouth)

        # Cálculo de temperatura media en nariz
        valor_nose = None
        if self.fixed_rects_nose:
            valores = []
            for x0, x1, y0, y1 in self.fixed_rects_nose:
                roi = gray[y0:y1, x0:x1]
                valores.append(np.mean(roi))
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 1)
            valor_nose = np.mean(valores)
            self.signal_nose.append(valor_nose)

        # Cálculo de temperatura media en boca
        valor_mouth = None
        if self.fixed_rects_mouth:
            for x0, x1, y0, y1 in self.fixed_rects_mouth:
                roi = gray[y0:y1, x0:x1]
                valor_mouth = np.mean(roi)
                self.signal_mouth.append(valor_mouth)
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 1)
        # Publicar señal combinada en tiempo real
        if valor_nose is not None and valor_mouth is not None and self.analysis_active:
            valor_comb = (valor_nose + valor_mouth) / 2
            self.pub.publish(Float32(data=float(valor_comb)))
            self.get_logger().info(f"[Tiempo real] Señal combinada: {valor_comb:.2f}")
            
            # Guardar marca de tiempo actual
            self.signal_timestamps.append(time.time())

            # Calcular frecuencia respiratoria cada 30 segundos
            if time.time() - self.last_rate_check >= 30:
                tiempo = np.array(self.signal_timestamps)
                señal = np.array(self.signal_nose) + np.array(self.signal_mouth)
                señal = señal / 2

                if len(señal) >= 15:
                    window_length = int(0.8 * 10)
                    if window_length % 2 == 0:
                        window_length += 1
                    if window_length >= len(señal):
                        window_length = len(señal) - 1
                        if window_length % 2 == 0:
                            window_length -= 1

                    señal_filtrada = savgol_filter(señal, window_length, 3)
                    valleys, _ = find_peaks(-señal_filtrada, distance=8, prominence=0.45)

                    duracion = tiempo[-1] - tiempo[0]
                    if duracion > 0:
                        rpm = len(valleys) * 60 / duracion
                        self.pub_rate.publish(Float32(data=float(rpm)))
                        self.get_logger().info(f"Frecuencia respiratoria estimada: {rpm:.2f} rpm")

                self.last_rate_check = time.time()
       
        # Mostrar imagen en tiempo real con los ROIs
        cv2.imshow("Deteccion Facial", frame)

        # Controles de teclado para ajustar manualmente los desplazamientos
        key = cv2.waitKey(1) & 0xFF
        if key == ord('w'): self.offset_nose_y -= 1
        if key == ord('s'): self.offset_nose_y += 1
        if key == ord('a'): self.offset_nose_x -= 1
        if key == ord('d'): self.offset_nose_x += 1
        if key == 82: self.offset_mouth_y -= 1
        if key == 84: self.offset_mouth_y += 1
        if key == 81: self.offset_mouth_x -= 1
        if key == 83: self.offset_mouth_x += 1

        self.get_logger().info(f"Offset nariz: ({self.offset_nose_x}, {self.offset_nose_y}) px | Offset boca: ({self.offset_mouth_x}, {self.offset_mouth_y}) px")

        # Activar o detener el análisis térmico con tecla 't'
        if key == ord('t'):
            if not self.analysis_active:
                self.get_logger().info("Análisis iniciado")
                self.analysis_active = True
                self.signal_nose = []
                self.signal_mouth = []
                self.signal_timestamps = []
                self.fixed_rects_nose = []
                self.fixed_rects_mouth = []
                self.start_time = time.time()
                self.last_rate_check = self.start_time
            else:
                self.end_time = time.time()
                duracion = self.end_time - self.start_time
                self.get_logger().info(f"Análisis detenido. Duración: {duracion:.2f} segundos. Procesando señales...")
                self.analysis_ready = True
                

    def process_signals(self):
        # Verificar que haya suficientes muestras para aplicar el filtro
        if len(self.signal_nose) < 15 or len(self.signal_mouth) < 15:
            self.get_logger().error("No hay suficientes datos para procesar señales (mínimo 15 muestras).")
            return

        # Procesa las señales guardadas y representa gráficamente los resultados
        raw_nose = np.array(self.signal_nose)
        raw_mouth = np.array(self.signal_mouth)
        raw_comb = (raw_nose + raw_mouth) / 2

        # Generar vector de tiempo según la duración real
        num_muestras = len(raw_comb)
        tiempo_total = self.end_time - self.start_time if self.end_time and self.start_time else 1
        tiempo = np.linspace(0, tiempo_total, num_muestras)

        # Aplicar filtro de Savitzky-Golay
        window_length = int(0.8 * 8) 
        if window_length % 2 == 0:
            window_length += 1
        if window_length >= len(raw_comb):
            window_length = len(raw_comb) - 1
            if window_length % 2 == 0:
                window_length -= 1

        señal_nose = savgol_filter(raw_nose, window_length, 3)
        señal_mouth = savgol_filter(raw_mouth, window_length, 3)
        señal_comb = savgol_filter(raw_comb, window_length, 3)

        # Representar las 3 gráficas en la misma ventana
        plt.figure(figsize=(10, 8))

        # Señal nariz
        plt.subplot(3, 1, 1)
        plt.plot(tiempo, señal_nose, label='Nariz', color='orange')
        plt.title("Señal térmica - Nariz")
        plt.ylabel("Intensidad")
        plt.grid(True)

        # Señal boca
        plt.subplot(3, 1, 2)
        plt.plot(tiempo, señal_mouth, label='Boca', color='blue')
        plt.title("Señal térmica - Boca")
        plt.ylabel("Intensidad")
        plt.grid(True)

        # Señal combinada + valles
        plt.subplot(3, 1, 3)
        plt.plot(tiempo, señal_comb, label='Combinada', color='green')

        # Detección de valles (mínimos locales) con prominencia mínima
        valleys, _ = find_peaks(-señal_comb, distance=8, prominence=0.45)
        #plt.plot(tiempo[valleys], señal_comb[valleys], 'ro', label='Valles (respiraciones)')

        plt.title("Señal térmica - Combinada (Nariz + Boca)")
        plt.xlabel("Tiempo (s)")
        plt.ylabel("Intensidad")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()
        
                # Mostrar análisis FFT de nariz, boca y señal combinada
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        self.show_fft(señal_mouth, "Boca", tiempo_total, axs[0])
        self.show_fft(señal_nose, "Nariz", tiempo_total, axs[1])
        self.show_fft(señal_comb, "Combinada", tiempo_total, axs[2])
        plt.tight_layout()
        plt.show()


    def show_fft(self, signal, label, tiempo_total, ax):
        # Calcular frecuencia de muestreo
        fs = len(signal) / tiempo_total
        N = len(signal)

        # Aplicar ventana de Hamming
        from scipy.signal import windows
        window = windows.hamming(N)
        signal_windowed = signal * window

        # Calcular FFT y eje de frecuencias
        yf = fft(signal_windowed)
        xf = fftfreq(N, 1 / fs)[:N // 2]
        magnitudes = 2.0 / N * np.abs(yf[:N // 2])

        # Filtrar solo frecuencias positivas útiles (por encima de 0.125 Hz)
        valid_indices = np.where(xf > 0.125)[0]
        if valid_indices.size > 0:
            freq_max = xf[valid_indices[np.argmax(magnitudes[valid_indices])]]
            rpm_fft = freq_max * 60  # convertir a respiraciones por minuto
        else:
            freq_max = 0
            rpm_fft = 0

        # Dibujar espectro
        ax.plot(xf, magnitudes)
        ax.axvline(x=freq_max, color='red', linestyle='--', label=f'Dominante: {freq_max:.2f} Hz')
        ax.set_xlim(0, 2)
        ax.set_ylim(0, 50)
        ax.set_title(f"FFT {label} - {rpm_fft:.1f} rpm")
        ax.set_xlabel("Frecuencia (Hz)")
        ax.set_ylabel("Magnitud")
        ax.grid()
        ax.legend()

def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectorNode()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)
            if node.analysis_ready:
                break
        if node.analysis_ready:
            node.process_signals()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
