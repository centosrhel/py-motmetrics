'''
get filename list in a folder
'''
import os
import sys
listing = []

if __name__ == '__main__':
    allItems = os.listdir(sys.argv[1])
    subfolder = os.path.basename(sys.argv[1])
    for item_a in allItems:
        if item_a.endswith('.jpg'):
            listing.append(item_a)
            os.makedirs(os.path.join('/Users/hu/Movies', subfolder, item_a.split('.')[0]))

    filename = os.path.basename(sys.argv[1]) + '.txt'
    with open(filename, 'w') as f:
        for item_a in listing:
            f.write(os.path.join(sys.argv[1], item_a) + '\n')