#!/usr/bin/env python3
# encoding = utf-8
# python3 xml2txt.py <a_dir>
from lxml import etree, objectify
import os
import sys # sys.argv, sys.exit
if __name__ == '__main__':
    xmlFilenames = [filename for filename in os.listdir(sys.argv[1]) if os.path.splitext(filename)[1].lower() == '.xml']
    for xmlFilename in xmlFilenames:
        fullName = os.path.join(sys.argv[1], xmlFilename)
        xml_root = etree.parse(fullName)
        outName = os.path.join(sys.argv[1], xml_root.xpath('/annotation/filename')[0].text+'.txt')
        f = open(outName, 'w')
        objects = xml_root.xpath('/annotation/object')
        for a_object in objects:
            subnodes = a_object.getchildren()
            for a_subnode in subnodes:
                if a_subnode.tag == 'bndbox':
                    boundaries = a_subnode.getchildren()
                    a_list = []
                    for boundary in boundaries:
                        # print(boundary.tag, boundary.text)
                        a_list.append(boundary.text)
                    break
            xmin, ymin, xmax, ymax = a_list
            f.write('{},{},{},{},{},{},{},{},{},"{}"\n'.format(xmin, ymin, xmax, ymin, xmax, ymax, xmin, ymax, subnodes[3].text,\
                subnodes[0].text))
        f.close()