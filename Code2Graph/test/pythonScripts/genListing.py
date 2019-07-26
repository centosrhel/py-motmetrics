import os, sys
import shutil
listing = []
mappings = {'one':0, 'two':1, 'three':2, 'four':3, 'five':4}
dst_root = '/Users/hu/Documents/FaceProject/Attractiveness/validation_227'
def search_file(path):
	items = os.listdir(path)
	for item in items:
		fullpath = os.path.join(path,item)
		if os.path.isdir(fullpath):
			search_file(fullpath)
		elif os.path.splitext(item)[1].lower() in ('.jpg', '.jpeg', '.png'):
			#listing.append(fullpath + '\t' + str(mappings[os.path.basename(path)]))
			src = os.path.join('/Users/hu/Documents/FaceProject/Attractiveness/preprocessedImages_227', item)
			dst = os.path.join(dst_root, os.path.basename(path))
			shutil.copy(src, dst)
			
if __name__ == '__main__':
	args = sys.argv
	search_file(args[1])
	'''with open('./listing.txt', 'w') as f:
		for filename in listing:
			#f.write(filename + '\t' + os.path.splitext(os.path.split(filename)[1])[0] + '\n')
			f.write(filename + '\n')'''
