import os

import tensorflow as tf

from object_detection.utils import config_util
from object_detection.builders import model_builder

# Set memmory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


class Model(tf.Module):
    def __init__(self, detection_model):
        super(Model, self).__init__()
        self.detection_model = detection_model

    @tf.function
    def __call__(self, input_tensor):
        image, shapes = self.detection_model.preprocess(input_tensor)
        prediction_dict = self.detection_model.predict(image, shapes)
        return self.detection_model.postprocess(prediction_dict, shapes)


def _load_model(args):
    assert args.pipeline is not None
    assert args.checkpoint is not None

    pipeline_path = os.path.normpath(args.pipeline)
    checkpoint_path = os.path.normpath(args.checkpoint)

    # Load pipeline config and build a detection model
    configs = config_util.get_configs_from_pipeline_file(pipeline_path)
    model_config = configs['model']

    # with tf.Graph().as_default() as g:
    detection_model = model_builder.build(
        model_config=model_config, is_training=False)

    # Restore checkpoint
    ckpt = tf.train.Checkpoint(model=detection_model)
    ckpt.restore(os.path.normpath(checkpoint_path)).expect_partial()

    return detection_model


def load_model(args):
    detection_model = _load_model(args)
        
    return Model(detection_model)


def load_saved_model(args):
    assert args.model_path is not None

    model = tf.saved_model.load(os.path.normpath(args.model_path))
    return model


def save_model(args):
    assert args.output is not None

    detection_model = _load_model(args)

    model = Model(detection_model)
    call = model.__call__.get_concrete_function(tf.TensorSpec([None, None, None, 3], tf.float32))

    tf.saved_model.save(model, os.path.normpath(args.output), signatures=call)
