#!/usr/bin/env python3

import os
import re
import argparse

import pandas as pd
import xml.etree.ElementTree as ET
import tensorflow as tf

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util
from collections import namedtuple

# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"    # Suppress TensorFlow logging (1)


parser = argparse.ArgumentParser(
    description="Convert TensorFlow XML-to-TFRecord converter")
parser.add_argument("-i",
                    "--input_dir",
                    help="Path where images and annotations in xml are stored.",
                    type=str,
                    required=True)
parser.add_argument("-o",
                    "--output_path",
                    help="Path to output TFRecord (.record) file.",
                    type=str,
                    required=True)
parser.add_argument("-l",
                    "--label_map",
                    help="Path to label map.",
                    type=str,
                    required=True)



def annotations_to_pandas(input_dir):
    annotation_files = [i for i in os.listdir(input_dir)
        if re.search(r".+\.xml$", i)]

    xml_list = []
    for xml_file in annotation_files:
        tree = ET.parse(os.path.join(input_dir, xml_file))
        filename = tree.find("filename").text

        size = tree.find("size")
        width = int(size.find("width").text)
        height = int(size.find("height").text)

        for obj in tree.findall("object"):
            bndbox = obj.find("bndbox")
            value = (filename,
                     width,
                     height,
                     obj.find("name").text,
                     int(bndbox.find("xmin").text),
                     int(bndbox.find("ymin").text),
                     int(bndbox.find("xmax").text),
                     int(bndbox.find("ymax").text))

            xml_list.append(value)

    column_name = ["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"]
    xml_df = pd.DataFrame(xml_list, columns=column_name)

    data = namedtuple("data", ["filename", "object"])
    gb = xml_df.groupby("filename")
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]


def create_tf_example(input_dir, example, label_map):
    with tf.io.gfile.GFile(os.path.join(input_dir, example.filename), "rb") as fid:
        encoded_jpg = fid.read()

    width = int(example.object.iloc[0].width)
    height = int(example.object.iloc[0].height)

    filename = example.filename.encode("utf8")
    image_format = "jpg".encode("utf-8")
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for _, row in example.object.iterrows():
        str_class = row["class"]

        xmins.append(row["xmin"] / width)
        xmaxs.append(row["xmax"] / width)
        ymins.append(row["ymin"] / height)
        ymaxs.append(row["ymax"] / height)

        classes_text.append(str_class.encode("utf8"))
        classes.append(int(label_map[str_class]))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        "image/height": dataset_util.int64_feature(height),
        "image/width": dataset_util.int64_feature(width),
        "image/filename": dataset_util.bytes_feature(filename),
        "image/source_id": dataset_util.bytes_feature(filename),
        "image/encoded": dataset_util.bytes_feature(encoded_jpg),
        "image/format": dataset_util.bytes_feature(image_format),
        "image/object/bbox/xmin": dataset_util.float_list_feature(xmins),
        "image/object/bbox/xmax": dataset_util.float_list_feature(xmaxs),
        "image/object/bbox/ymin": dataset_util.float_list_feature(ymins),
        "image/object/bbox/ymax": dataset_util.float_list_feature(ymaxs),
        "image/object/class/text": dataset_util.bytes_list_feature(classes_text),
        "image/object/class/label": dataset_util.int64_list_feature(classes)
    }))
    return tf_example


def main():
    args = parser.parse_args()
    
    writer = tf.io.TFRecordWriter(os.path.normpath(args.output_path))

    input_dir = os.path.normpath(args.input_dir)
    label_map_path = os.path.normpath(args.label_map)
    label_map = label_map_util.create_categories_from_labelmap(label_map_path)
    label_map = { d["id"]: d["name"] for d in label_map }

    examples = annotations_to_pandas(input_dir)

    for example in examples:
        tf_example = create_tf_example(input_dir, example, label_map)
        writer.write(tf_example.SerializeToString())

    writer.close()

    print("Successfully created the TFRecord file: {}".format(args.output_path))

if __name__ == "__main__":
    main()