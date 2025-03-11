import pyrealsense2 as rs
import numpy as np
import cv2

# Configuração do pipeline da RealSense
pipe = rs.pipeline()
cfg = rs.config()

cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# Inicia a captura
pipe.start(cfg)

num = 0  # Contador para salvar imagens

while True:
    frames = pipe.wait_for_frames()
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()

    if not depth_frame or not color_frame:
        continue

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    # Aplica um mapa de cores na imagem de profundidade
    depth_cm = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.5), cv2.COLORMAP_JET)

    # Exibe as imagens
    cv2.imshow('RGB Image', color_image)
    cv2.imshow('Depth Image', depth_cm)

    # Captura teclas pressionadas
    key = cv2.waitKey(1)

    if key == ord('q'):  # Pressione 'q' para sair
        break
    elif key == ord('s'):  # Pressione 's' para salvar imagem RGB
        cv2.imwrite(f'images/img{num}.png', color_image)
        print(f"Image img{num}.png saved!")
        num += 1

# Encerra o pipeline e fecha janelas
pipe.stop()
cv2.destroyAllWindows()
