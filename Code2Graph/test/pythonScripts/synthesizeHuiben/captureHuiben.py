import cv2
import sys
import os
#cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture('http://admin:@192.168.13.125/media/?action=stream')
if not cap:
    sys.exit('cannot open camera! exit!')
#retval = cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
#if not retval:
#    sys.exit('cannot set width! exit!')
#retval = cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#if not retval:
#    sys.exit('cannot set height! exit!')
img_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        continue
    cv2.imshow('test', frame)
    k = cv2.waitKey(100)
    if k == 27:
        break
    elif k == 99: # ord('c')
        filename = os.path.join('Huiben_captured', 'RJ_YY_6_B-%d.png'%img_idx)
        cv2.imwrite(filename, frame)
        print(filename)
        img_idx += 1
print('totally captured %d images' % img_idx)
