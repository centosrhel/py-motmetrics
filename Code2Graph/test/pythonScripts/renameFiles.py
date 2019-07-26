'''rename all files in a folder'''
import os, sys
import shutil
def search_file(path, dst_path):
	i = 0
	items = os.listdir(path)
	#items.sort()
	for item in items:
		fullpath = os.path.join(path,item)
		if os.path.isdir(fullpath):
			#search_file(fullpath)
                        pass
		elif os.path.splitext(item)[1]=='.jpg':
			#newname = os.path.join(dst_path,os.path.split(path)[1]+'_'+str(i).zfill(4)+'.jpg')
			newname = os.path.join(dst_path, str(i)+'.jpg')
			#os.rename(fullpath,newname)
			shutil.copy(fullpath, newname)
			i += 1
			
if __name__ == '__main__':
	args = sys.argv
	search_file(args[1], args[2])
