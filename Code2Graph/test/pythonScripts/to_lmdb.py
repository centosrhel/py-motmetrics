import lmdb
import cv2
import numpy as np
import os

def checkImageIsValid(imageBin):
    if imageBin is None:
        return False
    try:
        imageBuf = np.frombuffer(imageBin, dtype=np.uint8)
        img = cv2.imdecode(imageBuf, cv2.IMREAD_COLOR)
        imgH, imgW = img.shape[:2]
    except:
        return False
    if imgH * imgW == 0:
        return False
    return True


def writeCache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def createDataset(outputPath, imagePathList, labelList, lexiconList=None, checkValid=True):
    """
    Create LMDB dataset for CRNN training.
    ARGS:
        outputPath    : LMDB output path
        imagePathList : list of image path
        labelList     : list of corresponding groundtruth texts
        lexiconList   : (optional) list of lexicon lists
        checkValid    : if true, check the validity of every image
    """
    assert len(imagePathList) == len(labelList)
    nSamples = len(imagePathList)
    env = lmdb.open(outputPath, map_size=1099511627776)
    cache = {}
    cnt = 1
    for i in range(nSamples):
        imagePath = imagePathList[i].split()[0]
        label = labelList[i]

        with open('sample/' + imagePath, 'rb') as f:
            imageBin = f.read()

        if checkValid:
            if not checkImageIsValid(imageBin):
                print('%s is not a valid image' % imagePath)
                continue
        imageKey = ('image-%09d' % cnt).encode('utf-8')
        labelKey = ('label-%09d' % cnt).encode('utf-8')
        cache[imageKey] = imageBin
        cache[labelKey] = label.encode('utf-8')
        if lexiconList:
            lexiconKey = 'lexicon-%09d' % cnt
            cache[lexiconKey] = ' '.join(lexiconList[i])
        if cnt % 1000 == 0:
            writeCache(env, cache)
            cache = {}
            print('Written %d / %d' % (cnt, nSamples))
        cnt += 1
    nSamples = cnt - 1
    cache['num-samples'.encode('utf-8')] = str(nSamples).encode('utf-8')
    writeCache(env, cache)
    print('Created dataset with %d samples' % nSamples)

OUT_PATH = 'sample/crnn_train_lmdb'
IN_PATH = 'sample/train.txt'

if __name__ == '__main__':
    outputPath = OUT_PATH
    if not os.path.exists(OUT_PATH):
        os.mkdir(OUT_PATH)
    img_data = open(IN_PATH)
    imgPathList = list(img_data)

    labelList = []
    for line in imgPathList:
        word = line.split()
        word = ','.join(word[1:])
        labelList.append(word)
    createDataset(outputPath, imgPathList, labelList)