import numpy as np
import cv2

# Carregar a imagem
img = cv2.imread('Assets/formas.png')

# Verifica se a imagem foi carregada corretamente
if img is None:
    print("Erro: Não foi possível carregar a imagem.")
    exit()

# Converter para HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Definir intervalo de azul
lower_blue = np.array([20, 100, 100])
upper_blue = np.array([30, 255, 255])

# Criar máscara para o azul
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Converter a máscara para escala de cinza para a detecção de cantos
gray = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

# Aplicar detecção de cantos
corners = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)

# Verificar se cantos foram detectados antes de continuar
if corners is not None:
    corners = np.int0(corners)  # Converter para inteiros
    for corner in corners:
        x, y = corner.ravel()
        cv2.circle(img, (x, y), 5, (255, 0, 0), 2)  # Desenhar os cantos detectados

# Exibir imagens
cv2.imshow('Image', img)
cv2.imshow('Mask', mask)
#cv2.imshow('Result', result)

cv2.waitKey(0)
cv2.destroyAllWindows()
