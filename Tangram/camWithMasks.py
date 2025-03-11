import pyrealsense2 as rs
import numpy as np
import cv2

# Definições
font = cv2.FONT_HERSHEY_COMPLEX

def nothing(x):
    pass

# Função que configura a máscara
def set_mask(mask):
    cv2.setTrackbarPos("L-H", "Trackbars", mask[0])
    cv2.setTrackbarPos("L-S", "Trackbars", mask[1])
    cv2.setTrackbarPos("L-V", "Trackbars", mask[2])
    cv2.setTrackbarPos("U-H", "Trackbars", mask[3])
    cv2.setTrackbarPos("U-S", "Trackbars", mask[4])
    cv2.setTrackbarPos("U-V", "Trackbars", mask[5])

# Cores das máscaras
blue_mask = np.array([99, 92, 61, 102, 255, 184]) # otimo
red_mask = np.array([0, 168, 80, 14, 255, 255]) # ruim, confunde com o rosa
pink_mask = np.array([167, 196, 82, 179, 255, 255])
yellow_mask = np.array([16, 104, 104, 35, 255, 255])
purple_mask = np.array([111, 0, 0, 123, 255, 134]) # ok mas esta confundindo com o amarelo
dark_green_mask = np.array([68, 220, 0, 255, 255, 111])
light_green_mask = np.array([52, 83, 87, 94, 201, 255]) # otimo


#Cores RGB
blue_rgb = (39, 94, 122)
red_rgb = (167, 61, 29)
pink_rgb = (168, 20, 54)
yellow_rgb = (179, 158, 53)
purple_rgb = (34, 37, 67)
dark_green_rgb = (55, 4, 36)
light_green_rgb = (76, 171, 117)

# Abrir a câmera
pipe = rs.pipeline()
cfg = rs.config()
cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipe.start(cfg)

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L-H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L-S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U-S", "Trackbars", 0, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 0, 255, nothing)


while True:
    frame = pipe.wait_for_frames()
    color_frame = frame.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    # Definir a máscara conforme a tecla pressionada
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        set_mask(blue_mask)
    if key == ord('w'):
        set_mask(red_mask)
    if key == ord('e'):
        set_mask(pink_mask)
    if key == ord('r'):
        set_mask(yellow_mask)
    if key == ord('t'):
        set_mask(purple_mask)
    if key == ord('y'):
        set_mask(dark_green_mask)
    if key == ord('u'):
        set_mask(light_green_mask)    

    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")

    lower = np.array([l_h, l_s, l_v])
    upper = np.array([u_h, u_s, u_v])

    mask = cv2.inRange(hsv, lower, upper)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel)

    # Countours detection
    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contour_color = (0,0,0)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)

        print(len(approx))

        if not cv2.isContourConvex(approx):
            continue    

        if area > 400:
            cv2.drawContours(color_image, [cnt], 0, contour_color, 5)
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            box = np.int32(box)
            cv2.drawContours(color_image, [box], 0, (0, 255, 0), 2)

        if len(approx) == 4:
            cv2.putText(color_image, "rectangle", (10,25), font, 1, (255,0,0))    
        elif len(approx) == 3:
            cv2.putText(color_image, "triangle", (10,25), font, 1, (0,255,0))
        
    cv2.imshow("Filtered Image", cv2.bitwise_and(color_image, color_image, mask=mask))
    cv2.imshow("RealSenseRGB", color_image)
    cv2.imshow("Mask", mask)

    # Finalizar se pressionar a tecla Esc
    if key == 27:
        break

cv2.destroyAllWindows()
pipe.stop()
