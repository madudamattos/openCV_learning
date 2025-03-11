import pyrealsense2 as rs
import numpy as np
import cv2

def nothing(x):
    #any operation
    pass

# Abrir a câmera externa 
pipe = rs.pipeline()
cfg  = rs.config()

cfg.enable_stream(rs.stream.color, 640,480, rs.format.bgr8, 30)

pipe.start(cfg)

cv2.namedWindow("Trackbars")

cv2.createTrackbar("L-H", "Trackbars", 94, 255, nothing)
cv2.createTrackbar("L-S", "Trackbars", 92, 255, nothing)
cv2.createTrackbar("L-V", "Trackbars", 61, 255, nothing)
cv2.createTrackbar("U-H", "Trackbars", 105, 255, nothing)
cv2.createTrackbar("U-S", "Trackbars", 255, 255, nothing)
cv2.createTrackbar("U-V", "Trackbars", 212, 255, nothing)

font = cv2.FONT_HERSHEY_COMPLEX

while True:
    frame = pipe.wait_for_frames()
    color_frame = frame.get_color_frame()
    color_image = np.asanyarray(color_frame.get_data())

    hsv = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)

    l_h = cv2.getTrackbarPos("L-H", "Trackbars")
    l_s = cv2.getTrackbarPos("L-S", "Trackbars")
    l_v = cv2.getTrackbarPos("L-V", "Trackbars")    
    u_h = cv2.getTrackbarPos("U-H", "Trackbars")
    u_s = cv2.getTrackbarPos("U-S", "Trackbars")
    u_v = cv2.getTrackbarPos("U-V", "Trackbars")

    lower_red = np.array([l_h,l_s,l_v])
    upper_red = np.array([u_h,u_s,u_v])

    mask = cv2.inRange(hsv, lower_red, upper_red)
    kernel = np.ones((5,5), np.uint8)
    mask = cv2.erode(mask, kernel)

    # Countours detection
    contours,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        approx = cv2.approxPolyDP(cnt, 0.03*cv2.arcLength(cnt, True), True)

        if area > 400:
            cv2.drawContours(color_image, [cnt], 0, (0,0,0), 5)

        if len(approx) == 4:
            cv2.putText(color_image, "rectangle", (10,25), font, 1, (0,0,0))    
        elif len(approx) == 3:
            cv2.putText(color_image, "triangle", (10,25), font, 1, (0,0,0))
        
            
    cv2.imshow("RealSenseRGB", color_image)
    cv2.imshow("Mask", mask)
    cv2.imshow("Kernel", kernel)

    if cv2.waitKey(1) == ord('q'):
        break


cv2.destroyAllWindows()
pipe.stop()
