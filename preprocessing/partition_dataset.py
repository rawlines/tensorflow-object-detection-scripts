#!/usr/bin/env python3

import os
import re
import argparse
import math
import random

import os.path as path
import xml.etree.ElementTree as ET
import utils.genetic_balanced as gen

from functools import cmp_to_key
from numpy import diff
from tokenize import group
from shutil import copyfile
from cv2 import sort
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

parser.add_argument(
    '--balance_with_genetic',
    help='Will use a genetic algorithm to balance the dataset, CAUTION, WILL REQUIRE A POWERFUL COMPUTER',
    default=False,
    required=False,
    action='store_true')


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
        if kwargs["genetic_balance"]:
            copy_balanced_genetic(all_images, train_ratio, image_source, annotation_source, test_dir, train_dir, **kwargs)
        else:
            copy_balanced(all_images, train_ratio, image_source, annotation_source, test_dir, train_dir, **kwargs)
    else:
        num_images = len(all_images)
        train_images = math.ceil(num_images * train_ratio)
        
        #train
        print(f"Processing images for training: {train_dir}")
        for img in all_images[:train_images]:
            copy(img, image_source, annotation_source, train_dir)

        #test
        print(f"Processing images for test: {test_dir}")
        for img in all_images[train_images:]:
            copy(img, image_source, annotation_source, test_dir)

def copy_balanced(all_images, train_ratio, image_source, annotation_source, test_dir, train_dir, **kwargs):
    print("Copyng dataset in balanced mode")
    
    all_annotations = [os.path.join(annotation_source, f'{img.split(".")[0]}.xml') for img in all_images]
    class_dict_stats = {}
    class_individual_stats = []
    
    #Analyze all the dataset
    for i, a in enumerate(all_annotations):
        result = analyze_xml(a)
        class_individual_stats.append([all_images[i], result])
        for k, v in result.items():
            if k in class_dict_stats:
                class_dict_stats[k] += v
            else:
                class_dict_stats[k] = v
    
    classes_sorted = list(map(lambda x: x[0], sorted(class_dict_stats.items(), key=lambda x: x[1])))
    
    #sort classes from less items to more
    def sort_func(a: list, b: list):
        stats_a: dict = a[1]
        stats_b: dict = b[1]
        
        num_keys_a = len(stats_a.keys())
        num_keys_b = len(stats_b.keys())
        
        total_classes_a = sum(stats_a.values())
        total_classes_b = sum(stats_b.values())
        
        difference_a = total_classes_a - num_keys_a
        difference_b = total_classes_b - num_keys_b
        
        def fit(stats: dict):
            _fit = 0
            for k, v in stats.items():
                _fit += classes_sorted.index(k) * v / class_dict_stats[k]
            return _fit
        
        fit_a = fit(stats_a)
        fit_b = fit(stats_b)
        
        fit_a += total_classes_a + difference_a + num_keys_a + total_classes_a
        fit_b += total_classes_b + difference_b + num_keys_b + total_classes_b
        return fit_a - fit_b

        
    sorted_list = sorted(class_individual_stats, key=cmp_to_key(sort_func))
    
    # Select the minimum number of samples
    min_train_samples = 0
    if kwargs["min_balanced_samples"] != -1:
        min_train_samples = kwargs["min_balanced_samples"]
    else:
        min_train_samples = float('inf')
        for v in class_dict_stats.values():
            ## how many samples of each class we should take for train using the ratio
            rationed_train_samples = math.floor(v * train_ratio)
            if rationed_train_samples < min_train_samples:
                min_train_samples = rationed_train_samples
        
    
    print(f"Min samples for balanced train dataset is {min_train_samples}")
    
    #Start copying trying to balance the dataset
    current_copy_stats = {}
    for img, result in sorted_list:
        must_copy = False
        
        #Check how many of each class are in each sample
        class_stats: dict = result
        for k, v in class_stats.items():
            current_copies = current_copy_stats[k] if k in current_copy_stats else v
            available_copies = class_dict_stats[k]
            print(k, available_copies, current_copies)
            if current_copies < min_train_samples and current_copies < available_copies:
                must_copy = True
                if k in current_copy_stats:
                    current_copy_stats[k] += v
                else:
                    current_copy_stats[k] = v
                
        #copy to train or test
        if must_copy:
            copy(img, image_source, annotation_source, train_dir)
        else:
            copy(img, image_source, annotation_source, test_dir)
    
    
def copy_balanced_genetic(all_images, train_ratio, image_source, annotation_source, test_dir, train_dir, **kwargs):
    print("Copyng dataset in balanced mode using a gentic algorithm")
    
    all_annotations = [os.path.join(annotation_source, f'{img.split(".")[0]}.xml') for img in all_images]
    class_dict_stats = {}
    class_individual_stats = []
    
    #Analyze all the dataset
    for i, a in enumerate(all_annotations):
        result = analyze_xml(a)
        class_individual_stats.append([all_images[i], result])
        for k, v in result.items():
            if k in class_dict_stats:
                class_dict_stats[k] += v
            else:
                class_dict_stats[k] = v
    
    num_images = len(all_images)
    train_images = math.ceil(num_images * train_ratio)
    
    genetic = gen.Genetic(class_individual_stats,
                      train_images,
                      iterations=1000,
                      living_things=10,
                      mutations=1,
                      reproductions=2,
                      n_threads=8,
                      selection='tournament',
                      probabilistic_repoblation=False,
                      tournament_percent=0.3,
                      probabilistic_mutation=True)

    genetic.optimize()
    print('Best score: ', genetic.best_score)
    print('Best genotype: ', genetic.best_genotype)
    
    test_images = genetic.best_genotype.get_chain()
    for img, result in test_images:
        try:
            all_images.remove(img)
            copy(img, image_source, annotation_source, train_dir)
        except Exception as e:
            print(f"{img} already copied")
        
    for img in all_images:
        copy(img, image_source, annotation_source, test_dir)
        

def main():
    args = parser.parse_args()
    partition_dir(args.annotation_dir,
                  args.image_dir,
                  args.output_dir,
                  args.train_ratio,
                  balanced=args.balance_train,
                  min_balanced_samples=args.min_balanced_samples,
                  genetic_balance=args.balance_with_genetic)


if __name__ == '__main__':
    main()
