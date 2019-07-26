# -*- coding: utf-8 -*-
import cv2, numpy as np
import os
import shutil

cv2.namedWindow('image')
cv2.moveWindow('image',250,50)
cv2.namedWindow('controls')
cv2.moveWindow('controls',250,50)

controls = np.zeros((50,1500),np.uint8)
cv2.putText(controls, "0: next, 9: pre, 1: A(0~10%), 2: B(11%~30%), 3: C(31%~50%), 4: D(51%~70%), 5: E(71%~90%), 6: F(91%~100%), 7: G(unknown), Esc: Exit",
    (40,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
cv2.imshow("controls",controls)
cv2.waitKey()

pp=os.getcwd()
image_folder = os.path.join(pp,"image")
output_folder = os.path.join(pp,'output')

count = 0
for i in os.listdir(image_folder):
    if os.path.isfile(os.path.join(image_folder,i)):
        count += 1
print(count)

dirA_path = os.path.join(output_folder ,'A(0~10%)')
if not os.path.exists(dirA_path):
    os.mkdir(dirA_path)
dirB_path = os.path.join(output_folder ,'B(11%~30%)')
if not os.path.exists(dirB_path):
    os.mkdir(dirB_path)
dirC_path = os.path.join(output_folder ,'C(31%~50%)')
if not os.path.exists(dirC_path):
    os.mkdir(dirC_path)
dirD_path = os.path.join(output_folder ,'D(51%~70%)')
if not os.path.exists(dirD_path):
    os.mkdir(dirD_path)
dirE_path = os.path.join(output_folder ,'E(71%~90%)')
if not os.path.exists(dirE_path):
    os.mkdir(dirE_path)
dirF_path = os.path.join(output_folder ,'F(91%~100%)')
if not os.path.exists(dirF_path):
    os.mkdir(dirF_path)
dirG_path = os.path.join(output_folder ,'G(unknown)')
if not os.path.exists(dirG_path):
    os.mkdir(dirG_path)

f = os.listdir(image_folder)
imageIdx = 0

while True:
    try:
        dstimage = os.path.join(image_folder,f[imageIdx])
        im = cv2.imread(dstimage)  
        cv2.imshow('image', im)
        status =   {ord('0'):'next', 
                    ord('9'):'prev',  
                    ord('1'):'A', 
                    ord('2'):'B', 
                    ord('3'):'C', 
                    ord('4'):'D', 
                    ord('5'):'E', 
                    ord('6'):'F', 
                    ord('7'):'G', 
                    27: 'exit'}[cv2.waitKey()]
        if status == 'next':
            imageIdx += 1
        elif status == 'prev':
            imageIdx -= 1
        elif status == 'A':
            shutil.copy(os.path.join(image_folder,f[imageIdx]),os.path.join(dirA_path,f[imageIdx]))
            imageIdx = imageIdx+1
        elif status == 'B':
            shutil.copy(os.path.join(image_folder,f[imageIdx]),os.path.join(dirB_path,f[imageIdx]))
            imageIdx = imageIdx+1
        elif status == 'C':
            shutil.copy(os.path.join(image_folder,f[imageIdx]),os.path.join(dirC_path,f[imageIdx]))
            imageIdx = imageIdx+1
        elif status == 'D':
            shutil.copy(os.path.join(image_folder,f[imageIdx]),os.path.join(dirD_path,f[imageIdx]))
            imageIdx = imageIdx+1
        elif status == 'E':
            shutil.copy(os.path.join(image_folder,f[imageIdx]),os.path.join(dirE_path,f[imageIdx]))
            imageIdx = imageIdx+1
        elif status == 'F':
            shutil.copy(os.path.join(image_folder,f[imageIdx]),os.path.join(dirF_path,f[imageIdx]))
            imageIdx = imageIdx+1
        elif status =='G':
            shutil.copy(os.path.join(image_folder,f[imageIdx]),os.path.join(dirG_path,f[imageIdx]))
            imageIdx = imageIdx+1
        elif status == 'exit':
            break
    except KeyError:
        print("Invalid Key was pressed!!")
cv2.destroyAllWindows()