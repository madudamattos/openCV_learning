import cv2

img = cv2.imread('Assets/formas.png')

gray_img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

_, thresh_img = cv2.threshold(gray_img, 135, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE )

for i, contour in enumerate(contours):
    if i == 0:
        continue 
    
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    
    cv2.drawContours(img, contour,0, (0, 0, 0), 4)
    
    x, y, w, h = cv2.boundingRect(approx)
    x_mid = int(x + w/3)
    y_mid = int(y+h/1.5)

    coords = (x_mid, y_mid)
    colour = (0,0,0)
    font = cv2.FONT_HERSHEY_DUPLEX
    
    if len(approx) == 3:
        cv2.putText(img, "Triangle", coords, font, 1, colour, 1)
    elif len(approx) == 4:
        cv2.putText(img, "Rectangle", coords, font, 1, colour, 1)
    else:
        cv2.putText(img, "Circle", coords, font, 1, colour, 1)


cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()