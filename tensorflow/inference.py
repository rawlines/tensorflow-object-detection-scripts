#!/usr/bin/env python3

import argparse
import os

import numpy as np
import tensorflow as tf

from six import BytesIO
from PIL import Image
from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder


parser = argparse.ArgumentParser(description="Partition dataset of images into training and testing sets",
                                formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument(
    '-i', '--image',
    help='Image file where to apply the inference.',
    type=str,
    required=True
)
parser.add_argument(
    '-o', '--output',
    help='Output file for store the image with inference.',
    type=str,
    required=True
)
parser.add_argument(
    '-p', '--pipeline',
    help='Path to pipeline.',
    type=str,
    required=True
)
parser.add_argument(
    '-m', '--model',
    help='Path to model.',
    type=str,
    required=True
)
parser.add_argument(
    '-c', '--checkpoint',
    help='Path to checkpoint.',
    type=str,
    required=True
)
parser.add_argument(
    '-l', '--label_map',
    help='Path to label map.',
    type=str,
    required=True
)

def load_model(pipeline_path, checkpoint_path, model_dir):
    pipeline_path = os.path.normpath(pipeline_path)
    model_dir = os.path.normpath(model_dir)

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(pipeline_path)
    model_config = configs['model']
    detection_model = model_builder.build(
        model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.normpath(checkpoint_path)).expect_partial()

    def get_model_detection_function(model):
        """Get a tf.function for detection."""

        @tf.function
        def detect_fn(image):
            """Detect objects in image."""

            image, shapes = model.preprocess(image)
            prediction_dict = model.predict(image, shapes)
            return model.postprocess(prediction_dict, shapes)

        return detect_fn

    return get_model_detection_function(detection_model)


def load_image(image_path):
    """ Loads image from path and stores and retruns it as a numpy array """
    img_data = tf.io.gfile.GFile(image_path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def load_category_index(label_map_path):
    return label_map_util.create_category_index_from_labelmap(
        os.path.normpath(label_map_path))


def main():
    args = parser.parse_args()
    model = load_model(args.pipeline, args.checkpoint, args.model)

    image_np = load_image(args.image)
    input_tensor = tf.convert_to_tensor(image_np, dtype=tf.float32)

    detections = model(input_tensor[None])

    category_index = load_category_index(args.label_map)
    
    image_np_with_annotations = image_np.copy()
    viz_utils.visualize_boxes_and_labels_on_image_array(
        image_np_with_annotations,
        detections["detection_boxes"][0].numpy(),
        detections["detection_classes"][0].numpy().astype(np.int),
        detections["detection_scores"][0].numpy(),
        category_index,
        use_normalized_coordinates=True,
        min_score_thresh=0.5)

    img = Image.fromarray(image_np_with_annotations)
    img.save(os.path.normpath(args.output))
    print(detections["detection_scores"][0].numpy())


if __name__ == '__main__':
    main()