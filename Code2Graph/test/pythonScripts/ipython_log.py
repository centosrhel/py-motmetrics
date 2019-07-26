# IPython log file

get_ipython().run_line_magic('logstart', '')
import cv2 as cv
flags = [i for i in dir(cv) if i.startswith('COLOR_')]
print(flags)
import numpy as np
'''cap = cv.VideoCapture(0)
while(True):
    _, frame = cap.read()
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    mask = cv.inRange(hsv, lower_blue, upper_blue)
    res = cv.bitwise_and(frame, frame, mask=mask)
    cv.imshow('frame', frame)
    cv.imshow('mask', mask)
    cv.imshow('res', res)
    if (cv.waitKey(5) & 0xFF) == 27:
        break
cv.destroyAllWindows()'''
green = np.uint8([[0,255,0]])
hsv_green = cv.cvtColor(green, cv.COLOR_BGR2HSV)
green = np.uint8(np.array([[0,255,0]]))
hsv_green = cv.cvtColor(green, cv.COLOR_BGR2HSV)
green.shape
green = np.uint8([[[0,255,0 ]]])
green.shape
hsv_green = cv.cvtColor(green, cv.COLOR_BGR2HSV)
hsv_green
red = np.uint8([[[0,0,255]]])
hsv_red = cv.cvtColor(red, cv.COLOR_BGR2HSV)
hsv_red
blue = np.uint8([[[255,0,0]]])
hsv_blue = cv.cvtColor(blue, cv.COLOR_BGR2HSV)
hsv_blue
img = cv.imread('messi5.jpg')
res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
img.shape
img = cv.imread('/Users/hu/Pictures/caffe_2.jpeg')
res = cv.resize(img, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
height, width = img.shape[:2]
res = cv.resize(img,(2*width, 2*height), interpolation = cv.INTER_CUBIC)
img = cv.imread('/Users/hu/Pictures/caffe_2.jpeg',0)
rows,cols = img.shape
img.shape
M = np.float32([[1,0,100], [0,1,50]])
M
dst = cv.warpAffine(img, M, (cols,rows))
cv.imwrite('img.jpg', dst)
dst = cv.warpAffine(img, M, (cols+100,rows+50))
cv.imwrite('img.jpg', dst)
M = cv.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), 90, 1)
dst = cv.warpAffine(img, M, (cols,rows))
cv.imwrite('img.jpg', dst)
dst = cv.warpAffine(img, M, (rows,cols))
cv.imwrite('img.jpg', dst)
dst = cv.warpAffine(img, M, (cols,rows))
cv.imwrite('img2.jpg', dst)
pts1 = np.float32([[50,50],[200,50],[50,200]])
pts2 = np.float32([[10,100],[200,50],[100,250]])
M = cv.getAffineTransform(pts1, pts2)
dst = cv.warpAffine(img, M, (cols,rows))
cv.imwrite('img.jpg', dst)
pts1 = np.float32([[56,65],[368,52],[28,387],[389,390]])
pts2 = np.float32([[0,0],[300,0],[0,300],[300,300]])
pts1.shape
type(pts1)
M = cv.getPerspectiveTransform(pts1, pts2)
M
M.shape
M.dtype
M.stype
dst = cv.warpPerspective(img, M, (300, 300))
cv.imwrite('img.jpg', dst)
exit()
