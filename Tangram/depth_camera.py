import pyrealsense2 as rs
import numpy as np
import cv2

# Configuração do pipeline
pipeline = rs.pipeline()
config = rs.config()

# Ativar streams de cor (RGB) e profundidade
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Iniciar o pipeline
pipeline.start(config)

# Alinhar os frames de profundidade com os frames RGB
align_to = rs.stream.color
align = rs.align(align_to)

try:
    while True:
        # Capturar os frames e alinhar
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            continue

        # Converter os frames para numpy arrays
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())

        # Coordenada do ponto desejado (no centro)
        x, y = 320, 240

        # Obter a distância (em metros)

        distance = depth_frame.get_distance(x, y)
        print(f"Distância no ponto ({x}, {y}): {distance:.3f} m")

        # Desenhar o ponto no frame RGB
        cv2.circle(color_image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(color_image, f"{distance:.2f}m", (x + 10, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        # Exibir a imagem RGB com o ponto marcado
        cv2.imshow('Depth Image', depth_colormap)
        cv2.imshow('RGB Image', color_image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
