import cv2
import argparse
import os

import pyzed.sl as sl
import utils.tensorrt_utils as tu
import utils.model_utils as mu
import utils.inference_utils as iu


parser = argparse.ArgumentParser(description="runs inference on SVO files",
                                formatter_class=argparse.RawTextHelpFormatter)

parser.add_argument(
    '-s', '--svo_file',
    help='Path to svo file for inference.',
    type=str,
    default=None,
    required=False
)
parser.add_argument(
    '-o', '--output',
    help='Output file for store the video with inference.',
    type=str,
    default=None,
    required=False
)
parser.add_argument(
    '--model_path',
    help='Path to saved model dir.',
    type=str,
    default=None,
    required=False
)
parser.add_argument(
    '-l', '--label_map',
    help='Path to label map.',
    type=str,
    default=None,
    required=False
)
parser.add_argument(
    '--num_fps',
    help="Number of fps to grab, if not set will infer the full svo file",
    type=int,
    default=0,
    required=False
)
parser.add_argument(
    '--use_tensorrt',
    help="Use this to inference with a tensorrt model",
    default=False,
    required=False,
    action='store_true'
)
parser.add_argument(
    '--tensorrt_model',
    type=str,
    default=None,
    required=False
)

def main():
    args = parser.parse_args()
    svo_path = os.path.normpath(args.svo_file)

    category_index = iu.load_category_index(args.label_map)
    print("Loading TF model as: ", end="")
    #Load saved model as default or as trt
    if args.tensorrt_model:
        print("TensorRT Model")
        detect_fn = tu.load_tensorrt_model(args)
    else:
        print("Normal SavedModel")
        detect_fn = mu.load_saved_model(args)

    print("Initializing ZED SVO Loader")
    params = sl.InitParameters()
    params.set_from_svo_file(svo_path)
    params.svo_real_time_mode = True
    params.depth_mode = sl.DEPTH_MODE.QUALITY
    params.coordinate_units = sl.UNIT.METER
    params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
    params.depth_stabilization = True

    cam = sl.Camera()
    if not cam.is_opened():
        print("Opening SVO File...")

    status = cam.open(params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    left = sl.Mat()
    depth = sl.Mat()
    point_cloud = sl.Mat()
    depth_img = sl.Mat()
    resolution = sl.Resolution(1280, 720)

    fourcc = cv2.VideoWriter_fourcc('X', 'V', 'I', 'D')
    video_writer = cv2.VideoWriter(os.path.normpath(args.output), fourcc, 20.0, (1280, 720))

    num_fps = args.num_fps
    eternal_grab = True if num_fps <= 0 else False
    current_fps = 0
    print("Running inference")
    while eternal_grab or current_fps <= num_fps:
        err_code = cam.grab()
        if err_code != sl.ERROR_CODE.SUCCESS:
            break
        
        # Retrieve data each frame
        cam.retrieve_image(left, view=sl.VIEW.LEFT, resolution=resolution)
        cam.retrieve_image(depth_img, view=sl.VIEW.DEPTH, resolution=resolution)

        cam.retrieve_measure(depth, measure=sl.MEASURE.DEPTH, resolution=resolution)
        cam.retrieve_measure(point_cloud, measure=sl.MEASURE.XYZ, resolution=resolution) # -> (X, Y ,Z, not used)

        #pt_cloud_np = point_cloud.get_data()

        img = cv2.cvtColor(left.get_data(), cv2.COLOR_RGBA2RGB)
        input_tensor = iu.get_input_tensor(img)

        predictions = detect_fn(input_tensor[None])
        img_with_boxes = iu.write_boxes_in_image(img, predictions, category_index, tensorrt_out=args.tensorrt_model)
        #video_writer.write(img_with_boxes)

        print(f"Inference frame: {current_fps}/{'Infinity' if eternal_grab else num_fps}", end="\r", flush=True)

        current_fps += 1

    print("Inference finished!!")

    video_writer.release()
    cam.close()


if __name__ == "__main__":
    main()
