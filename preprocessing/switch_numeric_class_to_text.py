#!/usr/bin/env python3

import os
import argparse
import glob

import xml.etree.ElementTree as ET

from object_detection.utils import label_map_util


parser = argparse.ArgumentParser()
parser.add_argument('-i',
                    '--input_dir',
                    type=str,
                    help='Directory where xml files are',
                    required=True)
parser.add_argument('-l',
                    '--label_map',
                    type=str,
                    help='Path to label map file',
                    required=True)
parser.add_argument('--increment_number_class_in_1',
                    type=bool,
                    help='Weather to increment class from xml in 1',
                    default=False,
                    required=False)
                    
def update_xml_class(xml_file, label_map, increment_in_1=False):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = list(root.getiterator('object'))

    for obj in objects:
        name_elem = obj.find("name")
        label = int(name_elem.text)

        if increment_in_1:
            label += 1

        name_elem.text = label_map[label]["name"]
        print(f"Processed: {os.path.basename(xml_file)}")

    tree.write(xml_file)


def main():
    args = parser.parse_args()
    input_dir = os.path.normpath(args.input_dir)
    label_map_path = os.path.normpath(args.label_map)
    increment_class = args.increment_number_class_in_1

    label_map = label_map_util.create_category_index_from_labelmap(label_map_path)

    xml_files = glob.glob(os.path.join(input_dir, "*.xml"))
        
    for xml in xml_files:
        update_xml_class(xml, label_map, increment_class)


if __name__ == "__main__":
    main()