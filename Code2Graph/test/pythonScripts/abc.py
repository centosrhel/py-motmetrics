from __future__ import absolute_import, division, print_function
import os, sys, shutil
import glob, math, random
src_root = ''
dst_root = '/Users/hu/Documents/FaceProject/Attractiveness/validation_227'
items = os.listdir(dst_root)
for item in items:
    words = item.split('_')
    if words[3] == '中' and words[5] == '中' and words[6] == '80':
        src_pathname = os.path.join(src_root, item)
        dst_pathname = os.path.join(dst_root, item)
        shutil.copy(src_pathname, dst_pathname)
        #shutil.move
