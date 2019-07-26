import os, sys
listing = []
def search_file(path):
    items = os.listdir(path)
    for item in items:
        fullpath = os.path.join(path,item)
        if os.path.isdir(fullpath):
            search_file(fullpath)
        elif os.path.splitext(item)[1].lower() in ('.jpg', '.jpeg', '.png'):
            a = item.split('_')
            listing.append(fullpath + '\t' + a[-2] + '\t' + a[1])
            
if __name__ == '__main__':
	args = sys.argv
	search_file(args[1])
	with open('./genData.txt', 'w') as f:
		for item in listing:
			#f.write(filename + '\t' + os.path.splitext(os.path.split(filename)[1])[0] + '\n')
			f.write(item + '\n')
