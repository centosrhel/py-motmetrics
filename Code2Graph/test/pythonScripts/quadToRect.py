# python3
# encoding = utf-8
import cv2
import numpy as np
import math

height = 32
y_max = height - 1

img = cv2.imread('/Users/hu/Public/ocr_nlp/data/huoyan/code/sample/00000006.jpg',1)
gt_file = open('/Users/hu/Public/ocr_nlp/data/huoyan/code/sample/00000006.txt', 'r')
gt_labels = gt_file.readlines()
gt_file.close()

a = gt_labels[11]
a = a.strip()
b = a.split(',')
c = b[:-1]
c = [float(i) for i in c]
d = np.array(c)
d = d.reshape((-1,2))
pts1 = d
pts1 = pts1.astype(np.float32)

p0_3 = pts1[0] - pts1[3]
p1_2 = pts1[1] - pts1[2]
p0_1 = pts1[0] - pts1[1]
p3_2 = pts1[3] - pts1[2]

hl = math.sqrt(p0_3[0]*p0_3[0]+p0_3[1]*p0_3[1])
hr = math.sqrt(p1_2[0]*p1_2[0]+p1_2[1]*p1_2[1])
wu = math.sqrt(p0_1[0]*p0_1[0]+p0_1[1]*p0_1[1])
wb = math.sqrt(p3_2[0]*p3_2[0]+p3_2[1]*p3_2[1])

wh_ratio = (wu+wb) / (hl+hr)

width = round(height * wh_ratio)
x_max = width -1
pts2 = np.float32([[0,0], [x_max,0],[x_max,y_max],[0,y_max]])
M = cv2.getPerspectiveTransform(pts1,pts2)
dst = cv2.warpPerspective(img, M, (width,height))
cv2.imwrite('/Users/hu/Public/ocr_nlp/data/huoyan/code/sample/00000006_11.jpg',dst)