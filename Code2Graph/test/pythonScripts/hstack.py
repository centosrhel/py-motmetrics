#!/usr/bin/env python3
# encoding = utf-8
import cv2
import numpy as np
import random
import os
import sys
import math

def hconcat_folders(folder1, folder2, folder3):
    leftImageFilenames = [i for i in os.listdir(folder1) \
        if os.path.splitext(i)[1].lower() in ('.jpg', '.png', '.jpeg')]
    for imageFilename in leftImageFilenames:
        print(imageFilename)
        leftImage = cv2.imread(os.path.join(folder1, imageFilename), 1)
        rightImage = cv2.imread(os.path.join(folder2, imageFilename), 1)
        jointImage = cv2.hconcat([leftImage, rightImage])
        cv2.imwrite(os.path.join(folder3, imageFilename), jointImage)


def main():
    hconcat_folders(sys.argv[1], sys.argv[2], sys.argv[3])

if __name__ == '__main__':
    main()