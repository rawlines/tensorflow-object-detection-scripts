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

parser.add_argument('-n',
                    '--new_class',
                    type=str,
                    help='New class name for merged classes',
                    required=True)

parser.add_argument('-c',
                    '--classes_to_merge',
                    type=list,
                    nargs='+',
                    help='Class names to merge, in format: "class 1" "class 2" "class n"',
                    required=True)


def update_xml_class(xml_file, classes_to_merge, new_class):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = list(root.getiterator('object'))

    for obj in objects:
        name_elem = obj.find("name")
        label = name_elem.text

        if label in classes_to_merge:
            name_elem.text = new_class

        print(f"Processed: {os.path.basename(xml_file)}")

    tree.write(xml_file)


def main():
    args = parser.parse_args()
    input_dir = os.path.normpath(args.input_dir)
    new_class = args.new_class
    classes_to_merge = list(map(lambda x: "".join(x), args.classes_to_merge))

    xml_files = glob.glob(os.path.join(input_dir, "*.xml"))
        
    for xml in xml_files:
        update_xml_class(xml, classes_to_merge, new_class)


if __name__ == "__main__":
    main()