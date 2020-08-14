import cv2 as cv
import numpy as np
# import matplotlib.pyplot as plt

def draw_circle(event, x, y, flags, params):
    if event == cv.EVENT_LBUTTONDOWN:
        cv.circle(img,(x,y),25,(0,0,255),thickness=5)


img = cv.imread('samples/cat-sample-02.png')
# img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
windowname = "Image Window"

cv.namedWindow(windowname)
cv.setMouseCallback(windowname,draw_circle)
while True:
    cv.imshow(windowname,img)
    
    if cv.waitKey(1) & 0xFF == 27:
        break

cv.destroyAllWindows()