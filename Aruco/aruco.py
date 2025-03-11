import cv2
import numpy as np
import cv2.aruco as aruco

cap = cv2.VideoCapture(0)

# Define o dicionário de marcadores (pode ser DICT_4X4_50, DICT_5X5_50, etc.)
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
parameters = aruco.DetectorParameters()
arucoDetector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

markers = []

# Cria 7 arucos com ids diferentes 
for i in range(7):
    marker = aruco.generateImageMarker(aruco_dict, i, 200)
    markers.append(marker)
    cv2.imwrite(f"aruco_marker_{i}.png", marker)  # Salva imagem na pasta
    
    intrinsic_camera = np.array([[641.208, 0, 633.497], [0, 641.208, 367.593],  [0, 0, 1]])

    distortion = np.array([-0.43948, 0.18514, 0, 0, 0])  

while True:
    ret, frame = cap.read()

    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    markerCorners, markerIds, __ = arucoDetector.detectMarkers(frame)

    # Se houver marcadores detectados, desenha-os na tela
    if markerIds is not None:
        aruco.drawDetectedMarkers(frame, markerCorners, markerIds)
        print("IDs detectados:", markerIds.flatten())  # Exibe os IDs no terminal

    if len(markerCorners) > 0:
        for i in range(0, len(markerIds)):
           
            rvec, tvec, markerPoints = cv2.aruco.estimatePoseSingleMarkers(markerCorners[i], 0.025, intrinsic_camera, distortion)
            
            cv2.aruco.drawDetectedMarkers(frame, markerCorners) 

            cv2.drawFrameAxes(frame, intrinsic_camera, distortion, rvec, tvec, 0.01) 


    cv2.imshow("Detecção ArUco", frame)    
    
    if cv2.waitKey(1) == ord('q'): break
    
    
cap.release()
cv2.destroyAllWindows()
