# -*- coding: utf-8 -*-

import json
import urllib.request
from urllib.parse import quote
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    '--csv_files',
    nargs='+',
    default=['zuowen_2000.csv'],
    help="List of .csv to be parsed and downloaded."
)
parser.add_argument(
    '--csv_root',
    default='/Users/hu/Public/ocr_nlp/data/zuowen/',
    help="root path to csv files"
)

def check_pth(pth_in):
    if not os.path.exists(pth_in):
        os.makedirs(pth_in)

# test .csv:
args = parser.parse_args()
csv_root = args.csv_root
csv_lst = args.csv_files
csv_lst_2use = list()
out_dst_lst = list()
print("Given csv path: {}\n"
      "Given csv list to be processed:\n".format(csv_root))
for csv in csv_lst:
    print(csv)
    out_dst = csv_root + csv[:-4] + '/'
    if os.path.isdir(out_dst):
        print("Dir {} already exist! Will skipping downloaded images..."
              "If want to re-download, please remove dir first.".format(out_dst))
        out_dst_lst.append(out_dst)
        csv_lst_2use.append(csv)
        # continue
    else:
        out_dst_lst.append(out_dst)
        csv_lst_2use.append(csv)
# out_dst = csv_root.split('/')[-1][:-4] + '/'

for idx, out_dst in enumerate(out_dst_lst):
    print("Current csv {}...".format(csv_lst_2use[idx]))
    image_dst = out_dst + 'images/'
    gt_dst = out_dst + 'annotations/'
    check_pth(image_dst)
    check_pth(gt_dst)
    res_dict = {}

    # missed_lst = list()
    with open(csv_root + csv_lst_2use[idx], 'r') as csv_in:
        line0 = csv_in.readline()  # skip header
        headers = line0.split(',')
        # headers[-1] = headers[-1][:-1]
        headers[-1] = headers[-1].strip()

        try:
            if '7118' in csv_lst_2use[idx]:
                url_idx = headers.index('"ossvalidurl"')
            else:
                url_idx = headers.index('"url"')
        except ValueError:
            try:
                url_idx = headers.index('"ossvalidurl"')
            except ValueError:
                try:
                    url_idx = headers.index('"URL"')
                except ValueError:
                    url_idx = headers.index('"image_url"')

        uid_idx = 0
        try:
            result_idx = headers.index('"result"')
        except ValueError:
            result_idx = headers.index('"sd_result"')

        f = csv_in.readlines()
        img_cnt = 0
        img_skipped = 0
        print("Found total {} lines in csv file".format(len(f)))
        for row in f:
            items = row.split(',')
            url = items[url_idx][1:-1]
            # img_name = url.split('/')[-1]
            # img_name2use = img_name[:-4] + '_{}.jpg'.format(items[id_idx][1:-1])
            # # image name in 7118 is super long containing "%2F". Parse for 3 more levels:
            # if '7118' in csv_lst_2use[idx] or '7530' in csv_lst_2use[idx]:
            #     img_name2use = img_name2use.split('%2F')[-1][:-4]
            #     img_name2use = img_name2use.split('?')[0]
            #     img_name2use = img_name2use.split('_')[-1]
            img_name2use = items[uid_idx][1:-1] + '.jpg'
            # load labels and bboxes to dictionary:
            # if "20190123_92302" in csv_lst_2use[idx] or '20190214-92303-92515-99321' in csv_lst_2use[idx]:
            #     # in 20190123_92302.92304.csv, there are two extra columns after the last result column
            #     str2json = str(','.join(items[result_idx:-2]))
            # else:
            #     str2json = str(','.join(items[result_idx:]))
            str2json = str(','.join(items[result_idx:-3]))
            res = json.loads(json.loads(str2json))

            # check if image already exist:
            if os.path.isfile(image_dst + img_name2use):
                img_skipped += 1
                print("Skipped {} already downloaded images...".format(img_skipped))
                res_dict[img_name2use] = [None] * len(res['items'])
                for r, obj in enumerate(res['items']):
                    res_dict[img_name2use][r] = {
                        'labels': obj['labels']['transfer'],
                        'bbox': obj['meta']['geometry']  # in order: xmin, ymin, xmax, ymax
                    }
                #if img_skipped % 100 == 0:
                continue
            try:
                urllib.request.urlretrieve(url, image_dst + img_name2use)
            except UnicodeEncodeError:
                print("Found UnicodeEncodeError in url: {}".format(url))
                url2use = quote(url, safe='/:?=')
                urllib.request.urlretrieve(url2use, image_dst + img_name2use)

            res_dict[img_name2use] = [None] * len(res['items'])
            for r, obj in enumerate(res['items']):
                res_dict[img_name2use][r] = {
                    'labels': obj['labels']['transfer'],
                    'bbox': obj['meta']['geometry']  # in order: xmin, ymin, xmax, ymax
                }

            img_cnt += 1
            if img_cnt % 50 == 0 or img_cnt == 0:
                print("Downloaded {} images...".format(img_cnt))
            # if img_cnt > 5:
            #     break

        print("Downloaded {} images".format(img_cnt))

        with open(gt_dst + 'annotation.json', 'w') as j_out:
            json.dump(res_dict, j_out)
