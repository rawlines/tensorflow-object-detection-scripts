import os

import tensorflow as tf
import numpy as np

from tensorflow.python.compiler.tensorrt import trt_convert as trt

# Set memmory growth
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)


def load_tensorrt_model(args):
    assert args.tensorrt_model is not None

    tensorrt_model = os.path.normpath(args.tensorrt_model)

    saved_model_loaded = tf.saved_model.load(tensorrt_model, tags=[trt.tag_constants.SERVING])
    
    graph_func = saved_model_loaded.signatures[trt.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    frozen_func = trt.convert_to_constants.convert_variables_to_constants_v2(graph_func)

    return frozen_func


def conver_to_tensorrt_model(args):
    assert args.model_dir is not None
    assert args.output_model_dir is not None

    model_dir = os.path.normpath(args.model_dir)
    output_model_dir = os.path.normpath(args.output_model_dir)

    conversion_params = trt.DEFAULT_TRT_CONVERSION_PARAMS
    conversion_params = conversion_params._replace(max_workspace_size_bytes=(1<<30))
    conversion_params = conversion_params._replace(precision_mode="FP16")
    conversion_params = conversion_params._replace(maximum_cached_engines=100)

    converter = trt.TrtGraphConverterV2(input_saved_model_dir=model_dir, conversion_params=conversion_params)
    converter.convert() #Dynamic op true default, because of TF 2.x

    def input_gen():
        # Input for a single inference call, for a network that has two input tensors:
        inp = np.random.normal(size=(1, 1920, 1080, 3), loc=127.5, scale=127.5).astype(np.float32)
        yield (inp),

    converter.build(input_fn=input_gen)
    converter.save(output_model_dir)
