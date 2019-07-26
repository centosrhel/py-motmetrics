from __future__ import absolute_import, division, print_function
import os, sys, shutil
import cv2
import numpy as np

'''
src_root = '/workspace/cedu/Projects/data/RJ_YW_5X/RJ_YW_5X_Gallery_original'
dst_root = '/workspace/cedu/Projects/data/RJ_YW_5X/RJ_YW_5X_Gallery'
if not os.path.exists(dst_root):
    os.makedirs(dst_root)
items = os.listdir(src_root)
items = sorted(items)
for item in items:
    if not item.endswith('.jpg'):
        continue
    img = cv2.imread(os.path.join(src_root, item))
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.blur(gray_img, (7,7))
    center_y, center_x = img.shape[0] // 2, img.shape[1] // 2
    for y in range(20, center_y):
        if gray_img[y, center_x] > 50:
            break
    for y2 in range(img.shape[0]-20, center_y, -1):
        if gray_img[y2, center_x] > 50:
            break
    for x in range(10, center_x):
        if gray_img[center_y, x] > 50:
            break
    crop_width, crop_height = img.shape[1]-x-10, y2-y-60
    print(center_y, center_x, crop_height, crop_width)
    break
save_idx = 0
for item in items:
    if not item.endswith('.jpg'):
        continue
    img = cv2.imread(os.path.join(src_root, item))
    print(item)
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.blur(gray_img, (7,7))
    for y in range(20, center_y):
        if gray_img[y, center_x] > 50:
            break
    for x in range(10, center_x):
        if gray_img[center_y, x] > 50:
            break
    cv2.imwrite(os.path.join(dst_root, 'RJ_YW_5X_%03d.jpg'%save_idx),img[y+40:y+41+crop_height, x+10:x+11+crop_width])
    save_idx += 1



src_root = '/workspace/cedu/Projects/data/RJ_YW_5X/RJ_YW_5X_Query_original'
dst_root = '/workspace/cedu/Projects/data/RJ_YW_5X/RJ_YW_5X_Query'
if not os.path.exists(dst_root):
    os.makedirs(dst_root)
items = [item for item in os.listdir(src_root) if item.endswith('.jpg')]
items = sorted(items)
counts = len(items) // 4
for i in range(1, counts):
    print(i)
    j = i - 1
    os.system('cp '+os.path.join(src_root, items[4*i])+' '+
              os.path.join(dst_root, ('RJ_YW_5X_%03d_%03d_%02d.jpg' %(2*j, 2*j+1,1))))
    os.system('cp ' + os.path.join(src_root, items[4 * i+1]) + ' ' +
              os.path.join(dst_root, ('RJ_YW_5X_%03d_%03d_%02d.jpg' % (2 * j, 2 * j + 1, 2))))
    os.system('cp ' + os.path.join(src_root, items[4 * i + 2]) + ' ' +
              os.path.join(dst_root, ('RJ_YW_5X_%03d_%03d_%02d.jpg' % (2 * j, 2 * j + 1, 3))))
    os.system('cp ' + os.path.join(src_root, items[4 * i + 3]) + ' ' +
              os.path.join(dst_root, ('RJ_YW_5X_%03d_%03d_%02d.jpg' % (2 * j, 2 * j + 1, 4))))
'''

src_root = '/mnt/DATA1/DATASETS/books/RJ_YW_Query'
dst_root = '/workspace/cedu/Projects/data/RJ_YW_Query_700'
DOWNSAMPLE_RATIO = 0.25
if not os.path.exists(dst_root):
    os.makedirs(dst_root)
def process(root_path):
    items = [item for item in os.listdir(root_path) if item.startswith('RJ_YW')]
    for item in items:
        full_pathname = os.path.join(root_path, item)
        print(full_pathname)
        if os.path.isdir(full_pathname):
            os.makedirs(os.path.join(dst_root, item))
            process(full_pathname)
        elif os.path.isfile(full_pathname) and item.endswith('.jpg'):
            img = cv2.imread(full_pathname)
            img = cv2.resize(img, (0,0), fx=DOWNSAMPLE_RATIO, fy=DOWNSAMPLE_RATIO, interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(dst_root, os.path.basename(root_path), item), img)
if __name__ == '__main__':
    process(src_root)
