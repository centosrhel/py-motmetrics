import numpy as np
import cv2
import sys
import matplotlib.pyplot as plt

img = cv2.imread(sys.argv[1],0)
h,w = img.shape[:2]
img = cv2.resize(img,(w//5,h//5),interpolation=cv2.INTER_AREA)
_,binImg = cv2.threshold(img,128,255,cv2.THRESH_BINARY) # 二值化
horizontal_sum = np.sum(binImg, axis=1) # 水平投影
plt.plot(horizontal_sum, range(horizontal_sum.shape[0]))
plt.gca().invert_yaxis()
plt.show()



import cv2
import numpy as np
from matplotlib import pyplot as plt
import json
import sys
import subprocess
import os

class detectTable(object):
    def __init__(self, src_img):
        self.src_img = src_img

    def run(self):
        if len(self.src_img.shape) == 2:
            gray_img = self.src_img
        elif len(self.src_img.shape) ==3:
            gray_img = cv2.cvtColor(self.src_img, cv2.COLOR_BGR2GRAY)

        # 对暗底亮字图像进行二值化
        # ~array == cv2.bitwise_not(array)
        thresh_img = cv2.adaptiveThreshold(~gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,15,-2)
        h_img = thresh_img.copy()
        v_img = thresh_img.copy()
        scale = 15

        h_size = int(h_img.shape[1] / scale) # 水平结构元素的宽度
        h_structure = cv2.getStructuringElement(cv2.MORPH_RECT,(h_size,1))
        h_erode_img = cv2.erode(h_img,h_structure)
        h_dilate_img = cv2.dilate(h_erode_img,h_structure)

        v_size = int(v_img.shape[0] / scale) # 垂直结构元素的高度
        v_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_size))
        v_erode_img = cv2.erode(v_img, v_structure)
        v_dilate_img = cv2.dilate(v_erode_img, v_structure)

        mask_img = h_dilate_img+v_dilate_img
        joints_img = cv2.bitwise_and(h_dilate_img,v_dilate_img)
        joints_img = cv2.dilate(joints_img,None,iterations=3)

img = cv2.imread(sys.argv[1])
detectTable(img).run()


image = cv2.imread('/Users/hu/Public/ocr/data/huoyan/code/sample/deskew_2.png')
# convert the image to grayscale and flip the foreground
# and background to ensure foreground is now "white" and
# the background is "black"
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray = cv2.bitwise_not(gray) # ~gray
 
# threshold the image, setting all foreground pixels to
# 255 and all background pixels to 0
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

# grab the (x, y) coordinates of all pixel values that
# are greater than zero, then use these coordinates to
# compute a rotated bounding box that contains all
# coordinates
coords = np.column_stack(np.where(thresh > 0))
angle = cv2.minAreaRect(coords)[-1]

# the `cv2.minAreaRect` function returns values in the
# range [-90, 0); as the rectangle rotates clockwise the
# returned angle trends to 0 -- in this special case we
# need to add 90 degrees to the angle
if angle < -45:
    angle = -(90 + angle)
 
# otherwise, just take the inverse of the angle to make
# it positive
else:
    angle = -angle

# rotate the image to deskew it
(h, w) = image.shape[:2]
center = (w // 2, h // 2)
# M = cv2.getRotationMatrix2D(center, angle, 1.0)
M = cv2.getRotationMatrix2D(center, -(90+angle), 1.0)
rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

""" cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle), \
    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2) """

cv2.imwrite('/Users/hu/Public/ocr/data/huoyan/code/sample/deskew_2.png', rotated)