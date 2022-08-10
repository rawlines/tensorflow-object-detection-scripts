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
parser.add_argument('-c',
                    '--old_class',
                    type=str,
                    help='Class to update',
                    required=True)
parser.add_argument('-n',
		    '--new_class',
                    type=str,
                    help='New class to use',
                    required=True)

def update_xml_class(xml_file, old_class, new_class):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = list(root.getiterator('object'))

    for obj in objects:
        name_elem = obj.find("name")

        if name_elem.text == old_class:
            name_elem.text = new_class

        print(f"Processed: {os.path.basename(xml_file)}")

    tree.write(xml_file)


def main():
    args = parser.parse_args()
    input_dir = os.path.normpath(args.input_dir)
    old_class = args.old_class
    new_class = args.new_class

    xml_files = glob.glob(os.path.join(input_dir, "*.xml"))

    for xml in xml_files:
        update_xml_class(xml, old_class, new_class)


if __name__ == "__main__":
    main()
