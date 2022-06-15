#!/usr/bin/env python3

import os
import argparse
import glob

import xml.etree.ElementTree as ET

parser = argparse.ArgumentParser()
parser.add_argument('-i',
                    '--input_dir',
                    type=str,
                    help='Directory where xml files are',
                    required=True)


class_dict = dict()
total_classes = 0

def analyze_xml(xml_file):
    global total_classes, class_dict
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = list(root.getiterator('object'))

    for obj in objects:
        label = obj.findtext("name")
        if label in class_dict:
            class_dict[label] += 1
        else:
            class_dict[label] = 1

        total_classes += 1


def main():
    global total_classes, class_dict
    
    args = parser.parse_args()
    input_dir = os.path.normpath(args.input_dir)
    os.chdir(input_dir)

    xml_files = glob.glob("*.xml")
    
    for xml in xml_files:
        analyze_xml(xml)
    
    print(f"Total classes: {total_classes}")
    for k, v in class_dict.items():
        percent = v / total_classes * 100
        print("{} -> {:.2f}% ({})".format(k, percent, v))


if __name__ == "__main__":
    main()