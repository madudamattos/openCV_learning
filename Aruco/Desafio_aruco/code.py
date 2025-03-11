# -*- coding: utf-8 -*-

import numpy as np
import cv2
import matplotlib.pyplot as plt
from imutils.video import VideoStream
import imutils
import time

CAM_ON = 0

if CAM_ON :
    
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)
else:
    
    cap = cv2.VideoCapture('output.avi')


img02 = cv2.imread('hello.jpg')
img02_rgb = cv2.cvtColor(img02, cv2.COLOR_BGR2RGB)


parameters =  cv2.aruco.DetectorParameters()
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
arucoDetector = cv2.aruco.ArucoDetector(dictionary, parameters)


count = 0

while(True):
    
    if CAM_ON :
    

        frame = vs.read()
        frame = imutils.resize(frame, width=400)
    else :
        
        ret, frame = cap.read()
        time.sleep(0.03)
        if not ret:
           break
       
    # Detect the markers in the image
    markerCorners, markerIds, rejectedCandidates = arucoDetector.detectMarkers(frame)

    img01_corners = cv2.aruco.drawDetectedMarkers(frame, markerCorners,markerIds)
    [l,c,ch] = np.shape(img02_rgb)
    print(l,' ',c,' ',' ',ch)

    pts_src = np.array([[0,0],[c,0],[c,l],[0,l]])

    im_out = frame

    for mark in markerCorners:

        pts_dst = np.array(mark[0])

        h, status = cv2.findHomography(pts_src, pts_dst)
        warped_image = cv2.warpPerspective(img02_rgb, h, (frame.shape[1],frame.shape[0]))
        mask = np.zeros([frame.shape[0], frame.shape[1]], dtype=np.uint8)

        cv2.fillConvexPoly(mask, np.int32([pts_dst]), (255, 255, 255), cv2.LINE_AA)

        element = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        mask = cv2.erode(mask, element, iterations=3)

        # Copy the mask into 3 channels.
        warped_image = warped_image.astype(float)
        mask3 = np.zeros_like(warped_image)

        for i in range(0, 3):

            mask3[:,:,i] = mask/255

        warped_image_masked = cv2.multiply(warped_image, mask3)
        frame_masked = cv2.multiply(im_out.astype(float), 1-mask3)
        im_out = cv2.add(warped_image_masked, frame_masked)
        
    cv2.imshow('Image',im_out.astype(np.uint8)) 
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

if CAM_ON:
    vs.stop()
else:
    cap.release()
    
cv2.destroyAllWindows()