import cv2
import pyrealsense2
from realsense_depth import *

# Initialize Camera Intel Realsense
dc = DepthCamera()

cv2.namedWindow("Color frame")
cv2.setMouseCallback("Color frame", show_distance)

while True:
    ret, depth_frame, color_frame = dc.get_frame()

    # Show distance for a specific point
    cv2.circle(color_frame, point, 4, (0, 0, 255))
    distance = depth_frame[point[1], point[0]]
    print(distance)

    cv2.putText(color_frame, "{}mm".format(distance), (point[0], point[1] - 20), cv2.FONT_HERSHEY_PLAIN, 2, (0,0,0), 2)

    cv2.imshow("depth frame", depth_frame)
    cv2.imshow("Color frame", color_frame)
    cv2.waitKey(0)
    if key == 27:
        break
