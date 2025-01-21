import numpy as np
import cv2

img = cv2.imread('Assets/shapes.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

corners =cv2.goodFeaturesToTrack(gray, 20, 0.65, 10)
#parâmetros: image, numero de quinas, minimum quality (o quao confiavel é a quina, minima distancia entre quinas(bom para cantos arredondados por exemplo))

corners = np.int_(corners)

for corner in corners:
    x,y = corner.ravel()
    cv2.circle(img, (x,y), 5, (255,0,0), 2) 

for i in range(len(corners)):
    for j in range(i+1, len(corners)):
        corner1 = tuple(corners[i][0])
        corner2 = tuple(corners[j][0])
        cv2.line(img, corner1, corner2, (0,255,0), 1)

cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()