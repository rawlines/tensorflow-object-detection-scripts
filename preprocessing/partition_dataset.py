#!/usr/bin/env python3

from multiprocessing.dummy import current_process
import os
import re
import argparse
import math
import random

import os.path as path
import xml.etree.ElementTree as ET

from shutil import copyfile

from scipy import rand


parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
                                formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    '-a', '--annotation_dir',
    help='Path to the folder where the annotations are stored.',
    type=str,
    required=True
)
parser.add_argument(
    '-i', '--image_dir',
    help='Path to the folder where the image dataset is stored.',
    type=str,
    required=True
)
parser.add_argument(
    '-o', '--output_dir',
    help='Path to the output folder where the train and test dirs should be created.',
    type=str,
    required=True
)
parser.add_argument(
    '-r', '--train_ratio',
    help='Ratio of train images all over total number of images, default is 0.7.',
    default=0.7,
    type=float)

parser.add_argument(
    '-b', '--balance_train',
    help='Will try to balance the dataset as much as data permits it, it might modify the train/test ratio',
    default=False,
    required=False,
    action='store_true')

parser.add_argument(
    '--min_balanced_samples',
    help='How much samples as minumim should be used to balance the dataset',
    type=int,
    default=-1,
    required=False)


def analyze_xml(xml_file):
    class_dict = {}
    
    if not path.exists(xml_file):
        return dict()
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    objects = list(root.getiterator('object'))

    for obj in objects:
        label = obj.findtext("name")
        if label in class_dict:
            class_dict[label] += 1
        else:
            class_dict[label] = 1

    return class_dict

def copy(img_path, img_source, a_source, dest):
    image_filename = path.join(img_source, img_path)
    annotation_filename = path.join(a_source, f'{img_path.split(".")[0]}.xml')

    if path.exists(annotation_filename):
        copyfile(image_filename, path.join(dest, path.basename(image_filename)))
        copyfile(annotation_filename, path.join(dest, path.basename(annotation_filename)))
    else:
        print(f"SKIPPING {image_filename}, annotation not exists")


def partition_dir(annotation_source, image_source, dest, train_ratio, **kwargs):
    annotation_source = path.normpath(annotation_source)
    image_source = path.normpath(image_source)
    dest = path.normpath(dest)

    train_dir = path.join(dest, 'train')
    test_dir = path.join(dest, 'test')

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    all_images = [i for i in os.listdir(image_source)
        if re.search(r'([a-zA-Z0-9\s_\\.\-\(\):])+(?i)(.jpg|.jpeg|.png)$', i)]

    #Shuffle the dataset
    random.shuffle(all_images)

    if kwargs["balanced"]:
        copy_balanced(all_images, train_ratio, image_source, annotation_source, test_dir, train_dir, **kwargs)
    else:
        num_images = len(all_images)
        train_images = math.ceil(num_images * train_ratio)
        
        #train
        print(f"Processing images for training: {train_dir}")
        for img in all_images[:train_images]:
            print(img)
            copy(img, image_source, annotation_source, train_dir)

        #test
        print(f"Processing images for test: {test_dir}")
        for img in all_images[train_images:]:
            print(img)
            copy(img, image_source, annotation_source, test_dir)


    #TODO: asegurar el balaceo de los datos

def copy_balanced(all_images, train_ratio, image_source, annotation_source, test_dir, train_dir, **kwargs):
    print("Copyng dataset in balanced mode")
    
    all_annotations = [os.path.join(annotation_source, f'{img.split(".")[0]}.xml') for img in all_images]
    class_dict_stats = {}
    class_individual_stats = []
    
    #Analyze all the dataset
    for a in all_annotations:
        result = analyze_xml(a)
        class_individual_stats.append(result)
        for k, v in result.items():
            if k in class_dict_stats:
                class_dict_stats[k] += v
            else:
                class_dict_stats[k] = v
    
    print(class_dict_stats)
    num_images = len(all_images)
    train_images = math.ceil(num_images * train_ratio)
    min_b_samples = kwargs["min_balanced_samples"] 
    min_samples = min_b_samples if min_b_samples != -1 else min(result.values())
    print(f"Min samples for balanced is {min_samples}")
    # if min_samples < train_images:
    #     train_images = min_b_samples if min_b_samples != -1 else min_samples
    
    #Start copying balancing the dataset
    class_dict_stats.clear()
    for i, img in enumerate(all_images):
        copy_to_train_votes = 0
        copy_to_test_votes = 0
        
        #Check how many of each class are in each sample
        class_stats = class_individual_stats[i]
        for k, v in class_stats.items():
            current_classes = class_dict_stats[k] if k in class_dict_stats else v
            if current_classes < min_samples:
                copy_to_train_votes += 1
            else:
                copy_to_test_votes += 1
                
        #copy to train or test
        if copy_to_train_votes > copy_to_test_votes:
            copy(img, image_source, annotation_source, train_dir)
            for k, v in class_stats.items():
                if k in class_dict_stats:
                    class_dict_stats[k] += v
                else:
                    class_dict_stats[k] = v
        else:
            copy(img, image_source, annotation_source, test_dir)

def main():
    args = parser.parse_args()
    partition_dir(args.annotation_dir,
                  args.image_dir,
                  args.output_dir,
                  args.train_ratio,
                  balanced=args.balance_train,
                  min_balanced_samples=args.min_balanced_samples)


if __name__ == '__main__':
    main()