#!/usr/bin/env python3
# encoding = utf-8
import cv2
import numpy as np
import random
import os
import sys

uniform_width = 1000.0 # first make width of muben to be uniform_width
border_width = 280
#max_disturb = border_width-10
maxAngle = 30
inputDir = '/Users/hu/Public/ocr/data/huoyan/Huiben_original/'
outDir = '/Users/hu/Public/ocr/data/huoyan/Huiben_synthed2/'
augMultiplier = 10

imgFilenames = [i for i in os.listdir(inputDir) if os.path.splitext(i)[1].lower() in ('.jpg', '.png', '.jpeg')]
result_idx = 7350

for img_name in imgFilenames:
    huiben = cv2.imread(os.path.join(inputDir, img_name), 1)
    stem_name = os.path.splitext(img_name)[0]
    gt_file = open(os.path.join(inputDir, stem_name+'.txt'), 'r')
    raw_gt_labels = gt_file.readlines()
    gt_file.close()
    gt_labels = []
    for a_label in raw_gt_labels:
        a_label = a_label.strip()
        items = a_label.split(',')
        tmp_items = [int(i) for i in items[:8]]
        tmp_items.append(','.join(items[9:]))
        gt_labels.append(tmp_items)

    huiben_size = huiben.shape
    scaling_factor = uniform_width / huiben_size[1]
    interpolation_method = cv2.INTER_LINEAR
    if scaling_factor < 1.0:
        interpolation_method = cv2.INTER_AREA
    resized_img = cv2.resize(huiben, (0,0), fx=scaling_factor, fy=scaling_factor, interpolation=interpolation_method)
    padded_img = cv2.copyMakeBorder(resized_img, border_width, border_width, border_width, border_width,\
        cv2.BORDER_CONSTANT, value=(255, 255, 255))
    for a_label in gt_labels:
        tmp_label = a_label[:-1]
        a_label[:-1] = [i*scaling_factor+border_width for i in tmp_label]

    rows, cols, _ = padded_img.shape
    np_gt_labels = np.array(gt_labels)

    xmax = border_width+uniform_width-1
    ymax = rows-border_width-1
    pts1 = np.float32([[border_width, border_width], [xmax, border_width], \
        [xmax, ymax], [border_width, ymax]])

    for k in range(augMultiplier):
        #M = cv2.getRotationMatrix2D(((cols-1)/2.0, (rows-1)/2.0), round(random.uniform(-maxAngle, maxAngle)), 1)
        pts2 = np.float32([[border_width*random.uniform(0,1), border_width*random.uniform(0,1)], \
            [xmax+random.uniform(-0.5,1)*border_width, border_width*random.uniform(0,1)], \
            [xmax+random.uniform(-0.5,1)*border_width, ymax+random.uniform(-0.5,1)*border_width], \
            [border_width*random.uniform(0,1), ymax+random.uniform(-0.5,1)*border_width]])
        M = cv2.getPerspectiveTransform(pts1, pts2)
        #dst = cv2.warpAffine(padded_img, M, (cols, rows), borderValue=(255, 255, 255))
        dst = cv2.warpPerspective(padded_img, M, (cols, rows), borderValue=(255, 255, 255))

        #cv2.imwrite('/Users/hu/Public/ocr/data/huoyan/code/sample/id_871853_pos_3_1.jpg',dst)
        
        gt_labels = np_gt_labels.tolist()
        for a_label in gt_labels:
            tmp_label = a_label[:-1]
            tmp_label = [float(i) for i in tmp_label]
            np_label = np.array(tmp_label).reshape([-1,2])
            for i, a_point in enumerate(np_label):
                homo_point = list(a_point)
                homo_point.append(1)
                homo_point = np.array(homo_point)
                homo_point = homo_point.reshape((3,1))
                dst_point = np.matmul(M, homo_point)
                dst_point = dst_point / dst_point[2][0]
                dst_point = dst_point[:-1].reshape((-1,))
                np_label[i] = dst_point
            np_label = np_label.reshape((-1,))
            a_label[:-1] = list(np_label)

        '''
        a_label = a_label[:-1]
        np_label = np.array(a_label)
        np_label = np_label.reshape((-1,2))
        np_label = np_label.astype(np.int32)
        np_label = np_label.reshape((-1,1,2))
        cv2.polylines(dst,[np_label], True, (0,0,255))
        cv2.imwrite('/Users/hu/Public/ocr/data/huoyan/code/sample/1.jpg', dst)
        '''
        out_name = '{:08d}'.format(result_idx)
        cv2.imwrite(os.path.join(outDir, out_name+'.jpg'),dst)
        outLabelFile = open(os.path.join(outDir, out_name+'.txt'), 'w')
        for a_label in gt_labels:
            outLabelFile.write('{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{:.1f},{}\n'.format(
                a_label[0],a_label[1],a_label[2],a_label[3],a_label[4],a_label[5],a_label[6],a_label[7],a_label[8]))
        outLabelFile.close()
        result_idx += 1

print(result_idx)
# if __name__ == '__main__':
#     main()