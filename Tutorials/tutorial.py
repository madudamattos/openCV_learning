import cv2

img = cv2.imread('Assets/shapes.png', -1)

# -1, cv2.IMREAD_COLOR : loads a color image. Any transparency of image will be neglected. It is the default flag.
# 0, cv2.IMREAD_GRAYSCALE: Loads image in grayscale mode
# 1, cv2.IMREAD_UNCHANGED: Loads image as such including alpha chanel

#cv2.imwrite('new_img.jpg', img)

cv2.imshow('Image', img)
# espera por um tempo infinito ate uma tecla ser pressionada, e ai fecha a janela
cv2.waitKey(0)
cv2.destroyAllWindows()