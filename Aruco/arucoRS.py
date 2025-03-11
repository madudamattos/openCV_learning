import cv2
import numpy as np
import cv2.aruco as aruco
import pyrealsense2 as rs
import socket
import json
import time
import os
from collections import deque

# Configuração do socket UDP
UDP_IP = "127.0.0.1"
UDP_PORT = 9050
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Configuração da câmera RealSense
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)

# Dicionário de marcadores
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
arucoDetector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

# Matriz intrínseca e distorção
intrinsic_camera = np.array([[641.208, 0, 633.497], [0, 641.208, 367.593], [0, 0, 1]])
distortion = np.array([-0.43948, 0.18514, 0, 0, 0])

# Margem de erro aceitável para posição e rotação
tolerance_tvec = 0.025  #  25mm
tolerance_rvec = 0.50  # ~15 graus

# Buffer de leituras (últimas 5 leituras por marcador)
marker_readings = {}

# Carregar posições salvas
def load_saved_positions():
    save_path = "Aruco/transformData.json"
    if os.path.exists(save_path):
        with open(save_path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
        return data.get("markers", [])
    return []

saved_markers = load_saved_positions()

while True:
    frames = pipe.wait_for_frames()
    color_frame = frames.get_color_frame()
    
    if not color_frame:
        continue
    
    color_image = np.asanyarray(color_frame.get_data())

    # Detecção de marcadores
    markerCorners, markerIds, _ = arucoDetector.detectMarkers(color_image)

    if markerIds is not None:
        detected_ids = set(markerIds.flatten())  # Conjunto com os IDs detectados

        # Verifica se os dois marcadores necessários estão presentes
        required_ids = {1, 4, 2}  # IDs que você quer garantir que estejam no frame
        if required_ids.issubset(detected_ids):  # Só continua se ambos foram detectados
            aruco.drawDetectedMarkers(color_image, markerCorners, markerIds)

            all_markers_data = []
            reference_rvec = None
            reference_tvec = None
            
            # Buscar o ArUco de ID 4 como referência
            for i in range(len(markerIds)):
                if markerIds[i][0] == 4:
                    reference_rvec, reference_tvec, _ = aruco.estimatePoseSingleMarkers(markerCorners[i], 0.015, intrinsic_camera, distortion)
                    reference_rvec = reference_rvec.flatten()
                    reference_tvec = reference_tvec.flatten()
                    reference_rvec[0] = 0  # Zerar a rotação em x
                    break

            if reference_rvec is not None and reference_tvec is not None:
                for i in range(len(markerIds)):
                    marker_id = int(markerIds[i][0])
                    rvec, tvec, _ = aruco.estimatePoseSingleMarkers(markerCorners[i], 0.015, intrinsic_camera, distortion)
                    rvec = rvec.flatten()
                    tvec = tvec.flatten()
                    
                    # Ignorar a rotação no eixo x (zerar a componente x do rvec)
                    rvec[0] = 0  # Zerar a rotação em x
                    
                    # Transformação relativa ao ID 4
                    relative_tvec = tvec - reference_tvec
                    relative_rvec = rvec - reference_rvec

                    # Adicionar leituras ao buffer
                    if marker_id not in marker_readings:
                        marker_readings[marker_id] = {
                            "tvec": deque(maxlen=5),
                            "rvec": deque(maxlen=5)
                        }

                    marker_readings[marker_id]["tvec"].append(relative_tvec)
                    marker_readings[marker_id]["rvec"].append(relative_rvec)

                    # Média das últimas 5 leituras
                    if len(marker_readings[marker_id]["tvec"]) == 5:
                        avg_tvec = np.mean(marker_readings[marker_id]["tvec"], axis=0)
                        avg_rvec = np.mean(marker_readings[marker_id]["rvec"], axis=0)

                        marker_data = {
                            "id": marker_id,
                            "rvec": np.round(avg_rvec, 3).tolist(),
                            "tvec": np.round(avg_tvec, 3).tolist(),
                            "inPos" : ""
                        }
                        all_markers_data.append(marker_data)

                        # Desenhar eixos no frame
                        cv2.drawFrameAxes(color_image, intrinsic_camera, distortion, rvec, tvec, 0.01)
            
                # Verificação de posição
                for saved_marker in saved_markers:
                    for detected_marker in all_markers_data:
                        if saved_marker["id"] == detected_marker["id"] and detected_marker["id"] != 4:
                            diff_tvec = np.linalg.norm(np.array(saved_marker["tvec"]) - np.array(detected_marker["tvec"]))
                            diff_rvec = np.linalg.norm(np.array(saved_marker["rvec"]) - np.array(detected_marker["rvec"]))
                            if diff_tvec <= tolerance_tvec and diff_rvec <= tolerance_rvec:
                                print(f"✅ Marker {saved_marker['id']} está aproximadamente na mesma posição!")
                                print(f"id: {saved_marker['id']}, rvec: {saved_marker['rvec']}, tvec: {saved_marker['tvec']}")
                                detected_marker["inPos"] = "true"
                            else:
                                print(f"❌ Marker {saved_marker['id']} se moveu!")
                                print(f"id: {saved_marker['id']}, rvec: {saved_marker['rvec']}, tvec: {saved_marker['tvec']}")
                                detected_marker["inPos"] = "false"
                # Envio UDP
                data_json = json.dumps({"markers": all_markers_data}).encode('utf-8')
                sock.sendto(data_json, (UDP_IP, UDP_PORT))
    
    # Exibir imagem com detecção
    cv2.imshow("Aruco detection", color_image)
    
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('s'):
        save_path = "Aruco/transformData.json"
        with open(save_path, "w", encoding="utf-8") as json_file:
            json.dump({"markers": all_markers_data}, json_file, indent=4)
        print(f"💾 JSON salvo em {save_path}")

# Encerramento
pipe.stop()
cv2.destroyAllWindows()
sock.close()
