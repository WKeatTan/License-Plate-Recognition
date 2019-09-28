import cv2
import numpy as np
import math
import imutils
from scipy import misc, ndimage

import DetectChars

###################################################################################################

def image_preprocessing(gray): # 形态学处理
    gaussian = cv2.GaussianBlur(gray, (3, 3), 0, 0, cv2.BORDER_DEFAULT) # 高斯平滑
    
    median = cv2.medianBlur(gaussian, 5) # 中值滤波
    
    sobel = cv2.Sobel(median, cv2.CV_8U, 1, 0, ksize=3) # Sobel算子 # 梯度方向: x  
    
    ret, binary = cv2.threshold(sobel, 170, 255, cv2.THRESH_BINARY) # 二值化
    
    # 核函数
    element1 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    element2 = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 7))
    
    dilation = cv2.dilate(binary, element2, iterations=1) # 膨胀
    
    erosion = cv2.erode(dilation, element1, iterations=1) # 腐蚀    
    
    dilation2 = cv2.dilate(erosion, element2, iterations=3) # 膨胀
    
    return dilation2
# end function image_processing

###################################################################################################

def GetRegion(prc):
    regions = []
    
    _, contours, hierarchy = cv2.findContours(prc, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) # 查找轮廓
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < 2000: # if pixels area < 2000 skip
            continue
        
        eps = 0.01 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        height = abs(box[0][1] - box[2][1])
        width = abs(box[0][0] - box[2][0])
        ratio = float(width) / float(height)        
        
        if (ratio < 5 and ratio > 1.8): # if ratio > 1.8 and < 5 add into regions
            regions.append(box) # this will be a rectangle
            
    return regions
# end function get region

###################################################################################################

def deskew(img_crop): # adjust the img angle 
    gaussian = cv2.GaussianBlur(img_crop, (3, 3), 0, 0, cv2.BORDER_DEFAULT)
    
    black_char = cv2.bitwise_not(img_crop)
    
    thresh = cv2.threshold(black_char,0,255,cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    
    edges = cv2.Canny(thresh,50,150,apertureSize = 3)
    
    lines = cv2.HoughLines(edges,1,np.pi/180,0) #霍夫变换 # a line will rotate 180 angle
    
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
    
    if (x2-x1) != 0 and (y2-y1) != 0:
        t = float(y2-y1)/(x2-x1)
        rotate_angle = math.degrees(math.atan(t))
    
        if rotate_angle > 45:
            rotate_angle = -90 + rotate_angle
        elif rotate_angle < -45:
            rotate_angle = 90 + rotate_angle
    
        rotate_img = ndimage.rotate(thresh, rotate_angle)
    
    return rotate_img
# end function deskew

###################################################################################################

def detect_possible_plate(frame): # start point of the lpr sys
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 灰度化 # convert to gray color
    gray = gray * 255 # a color range 0 to 255 # 0 = black # 255 = white
    
    prc = image_preprocessing(gray) # image processing 
    
    regions = GetRegion(prc) # get region of possible plate
    sX,sY,eX,eY = 0,0,600,210
    cv2.line(frame, (sX, eY), (eX, eY), (0, 0xFF, 0), 2) # draw a line to crop & recognition
    sX,sY,eX,eY = 0,0,600,350
    cv2.line(frame, (sX, eY), (eX, eY), (0, 0xFF, 0), 2) # draw a line to crop & recognition
    
    for box in regions:
        cv2.drawContours(frame, [box], 0, (0, 255, 0), 2) # draw a rectangle on object detected
        
        (x,y,w,h) = cv2.boundingRect(box)
        pointYU = box[1][1] # get box top point
        pointYL = box[0][1] # get box bottom point
        
        if pointYU > 220 and pointYL < 320: # if box located at between 
            img_crop = gray[y:y+h,x:x+w] # crop the possible plate    
            rotate_img = deskew(img_crop) # deskew 
        
            # check image color
            white_px = cv2.countNonZero(img_crop) # total white pixel
            total_px = y * x # total img pixel
            black_px = total_px - white_px # total black pixel
            
            if black_px > white_px: # if black > white 
                rotate_img = cv2.bitwise_not(rotate_img) # convert white char to black char
            
            # detect possible char
            listOfMatchingChars = DetectChars.findPossibleCharsInPlate(rotate_img) # detect char
            final_list = DetectChars.removeInnerOverlappingChars(listOfMatchingChars) # remove inner overlapping char
            
            if len(final_list) >= 3: # if a plate consist at least 3 char
                print('[INFO]:Detect %d license plates' % len(regions))
                cv2.imshow('Plate Detected',rotate_img) # display a plate with black char
                
                result = DetectChars.recognizeCharsInPlate(rotate_img, final_list)
                print(result)
                result_frame = frame
                cv2.putText(result_frame, result, (x,(pointYL+10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                cv2.imshow('Character Recognized', result_frame)
                
    cv2.imshow('Camera', frame) # display video camera capture
#end function detect possible plate
    
###################################################################################################
    
if __name__ == '__main__':
    try:
        print('[INFO] Start Capture')
        cap = cv2.VideoCapture(0) # camera start capture
        
        while(cap.isOpened()): # while camera work
            ret,frame = cap.read() # read frame
            frame = imutils.resize(frame, width = 600) # resize frame width to 600
            frame = imutils.resize(frame, height = 400) # resize frame height to 400
            
            detect_possible_plate(frame) # execute detect possible plate
    
            key = cv2.waitKey(27) & 0xFF # press q to exit program
            if key == ord('q'):
                break
            
    except Exception as e:
        print(e)
        
    cv2.destroyAllWindows() # close display windows
    cap.release() # close camera
