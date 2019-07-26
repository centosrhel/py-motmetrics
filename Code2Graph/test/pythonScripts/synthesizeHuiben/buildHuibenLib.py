inputFolder = '/Users/hu/Public/ocr/data/huoyan/Huiben_original/'
outFilename = 'HuibenLib.txt'
import os
labelFilenames = [i for i in os.listdir(inputFolder) if os.path.splitext(i)[1] == '.txt'] # os.walk
allChars = ''
for labelFilename in labelFilenames:
    f = open(os.path.join(inputFolder, labelFilename))
    gt_labels = f.readlines()
    f.close()
    for singleLine in gt_labels:
        allFields = singleLine.strip().split(',')
        transcript = (','.join(allFields[9:]))[1:-1]
        allChars += transcript

idx_to_char = sorted(list(set(allChars)))
char_to_idx = dict([(char, i) for i,char in enumerate(idx_to_char)])

with open(outFilename, 'w') as f:
    for labelFilename in labelFilenames:
        with open(os.path.join(inputFolder, labelFilename)) as ff:
            gt_labels = ff.readlines()

        chars = ''
        for singleLine in gt_labels:
            allFields = singleLine.strip().split(',')
            transcript = (','.join(allFields[9:]))[1:-1]
            chars += transcript

        uniqueChars = sorted(list(set(chars)))
        uniqueIndices = [char_to_idx[i] for i in uniqueChars]
        uniqueIndices = [str(i) for i in uniqueIndices]

        f.write(os.path.splitext(labelFilename)[0]+','+','.join(uniqueIndices)+'\n')