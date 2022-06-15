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
                    
def analyze_xml(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = list(root.getiterator('object'))

    labels = set()
    for obj in objects:
        label = obj.findtext("name")
        labels.add(label)

    print(f"{xml_file}: {labels}")


def main():
    args = parser.parse_args()
    input_dir = os.path.normpath(args.input_dir)
    os.chdir(input_dir)

    xml_files = glob.glob("*.xml")
        
    for xml in xml_files:
        analyze_xml(xml)


if __name__ == "__main__":
    main()