import os

import numpy as np
import tensorflow as tf

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

# Set memmory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def load_category_index(label_map_path):
    return label_map_util.create_category_index_from_labelmap(
        os.path.normpath(label_map_path))


def write_boxes_in_image(image_np, detections, category_index, threshold=0.5, tensorrt_out=False):
    image_np_with_annotations = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations,
        detections[1 if tensorrt_out else "detection_boxes"][0].numpy(),
        detections[2 if tensorrt_out else "detection_classes"][0].numpy().astype(np.int),
        detections[4 if tensorrt_out else "detection_scores"][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=threshold)

    return image_np_with_annotations


def get_input_tensor(img):
    return tf.convert_to_tensor(img, dtype=tf.float32)
