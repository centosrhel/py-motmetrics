import numpy as np
import json
import pathlib
import cv2

json_file_encoding = 'gb2312'

def fordict(contents,img):
    write_lines = ''
    for key in contents.keys():
        # print(key)
        #print(contents[key])
        if key == 'shapes' and isinstance(contents[key],list):
            for dd in range(len(contents[key])):
                sub_dict = contents[key][dd]
                #print(type(sub_dict['points']))
                ploypoints = []
                for point in sub_dict['points']:
                    write_lines = write_lines + str(point[0]) + ',' + str(point[1]) + ','
                    ploypoints.append(point)

                ploypoints = np.array(ploypoints,np.int32)
                cv2.polylines(img,[ploypoints],True,(0,0,255),2)

                write_lines = write_lines + sub_dict['label'] + '\n'
            break

    return write_lines , img

if __name__ == '__main__':

    image_dir = pathlib.Path('/Users/hu/Public/ocr/data/huoyan/Huiben_diandu')

    valid_path = pathlib.Path('/Users/hu/Public/ocr/data/huoyan/Huiben_diandu_visual')

    img_pathnames = image_dir.glob('*.jpg')

    for image_fn in img_pathnames:

        img_name = image_fn.as_posix()

        json_name = image_dir / image_fn.with_suffix('.json').name
        print(json_name)

        gt_save_name = image_fn.with_suffix('.txt')
        
        img = cv2.imread(img_name)
        
        f = open(json_name,'r',encoding=json_file_encoding)
    
        load_dict = json.load(f,encoding=json_file_encoding)
        f.close()
    
        write_str ,img = fordict(load_dict,img)

        valid_img_save_name = valid_path / image_fn.name

        cv2.imwrite(str(valid_img_save_name),img)

        gt_txt = open(gt_save_name,'w')

        gt_txt.writelines(write_str)
        gt_txt.close()