import argparse
import os
import cv2

import pyzed.sl as sl
import matplotlib.pyplot as plt
import numpy as np

from skimage import io
from skimage import filters
from skimage.morphology import white_tophat, black_tophat, disk, erosion
from skimage.metrics import structural_similarity, mean_squared_error
from skimage.color import rgb2gray
from skimage.color import rgba2rgb

parser = argparse.ArgumentParser(
    description="Extract frames from from SVO file")
parser.add_argument("-i",
                    "--input",
                    help="Path SVO file.",
                    type=str,
                    required=True)
parser.add_argument("-o",
                    "--output_path",
                    help="Path to output exported frames.",
                    type=str,
                    required=True)
parser.add_argument("--frames",
                    help="Num of frames to export",
                    type=int,
                    required=False,
                    default=-1)
parser.add_argument("--init_frame",
                    help="Frame to start counting from",
                    type=int,
                    required=False,
                    default=0)
parser.add_argument("--norm_coef",
                    help="Computes Calculates an absolute difference norm or a relative difference norm if coef is greater than computed, then image will be ignored. this metric will be ignored if value is 0",
                    type=float,
                    required=False,
                    default=0)
parser.add_argument("--crop_image",
                    help="Data to crop de image, selects the area to compute norm_coef: <x1,x2,y1,y2>",
                    nargs=4,
                    required=False,
                    default=None)
parser.add_argument("--reference_image",
                    help="Image to use as reference for norm_coef, if none, the previous frame will be used",
                    type=str,
                    required=False,
                    default='')
parser.add_argument("--norm_type",
                    help="Norm type to use. See cv2 NormTypes: https://docs.opencv.org/3.4/d2/de8/group__core__array.html#gad12cefbcb5291cf958a85b4b67b6149f",
                    type=int,
                    required=False,
                    default=4)
parser.add_argument("--show_computed_norm",
                    help="Display computed norm",
                    action='store_true',
                    default=False)
parser.add_argument("--inverse_coef_comparision",
                    help="When used this flag, the comparison will be if the coef is lower than computed, instead of greater than",
                    action='store_true',
                    default=False)

args = parser.parse_args()

num_frames = args.frames
init_frame = args.init_frame
svo_file = args.input
output_path = args.output_path

norm_coef = args.norm_coef
show_computed_norm = args.show_computed_norm
norm_type = args.norm_type
inverse_coef_comparision = args.inverse_coef_comparision
do_compute_norm = norm_coef != 0

crop_x1 = 0
crop_x2 = 0
crop_y1 = 0
crop_y2 = 0
do_crop_image = args.crop_image != None
if do_crop_image:
    crop_x1, crop_x2, crop_y1, crop_y2 = [int(i) for i in args.crop_image]

reference_image = args.reference_image
use_reference_image = args.reference_image != ''


every_frames = 1
total_frames = num_frames * every_frames if num_frames > 0 else -1

if not os.path.exists(output_path):
    os.mkdir(output_path)

params = sl.InitParameters()

params.set_from_svo_file(svo_file)
params.svo_real_time_mode = False
params.depth_mode = sl.DEPTH_MODE.QUALITY
params.coordinate_units = sl.UNIT.METER
params.coordinate_system = sl.COORDINATE_SYSTEM.IMAGE
params.depth_stabilization = True

cam = sl.Camera()
if not cam.is_opened():
    print('Opening ZED Camera...')

status = cam.open(params)
if status != sl.ERROR_CODE.SUCCESS:
    print(repr(status))
    exit()

left = sl.Mat()

resolution = sl.Resolution(1920, 1080)

print('Reading frames')
if init_frame > 0:
    cam.set_svo_position(init_frame)

def save_image(frame_num, img_data):
    plt.imsave(f'{output_path}/frame_{frame_num}.jpg', img_data)

prev_frame = None
if use_reference_image:
    prev_frame = io.imread(reference_image, as_gray=True)
    if do_crop_image:
        prev_frame = prev_frame[crop_y1:crop_y2,crop_x1:crop_x2]
    prev_frame = filters.roberts(prev_frame)

frame = 0
while frame <= total_frames or total_frames == -1:
    err_code = cam.grab()

    if err_code != sl.ERROR_CODE.SUCCESS:
        print('No more frames: ', repr(err_code))
        break

    if frame % every_frames != 0:
        continue

    # Retrieve data each frame
    cam.retrieve_image(left, view=sl.VIEW.LEFT, resolution=resolution)
    left_np = left.get_data()
    left_np[..., :3] = left_np[..., 2::-1]

    if do_compute_norm:
        current_frame = left_np[crop_y1:crop_y2,crop_x1:crop_x2] if do_crop_image else left_np
        current_frame = rgb2gray(rgba2rgb(current_frame))
        
        if prev_frame is not None:
            norm = cv2.norm(current_frame, prev_frame, norm_type)
            if show_computed_norm:
                print(f'norm {init_frame + frame}: {norm}')

            elif norm > norm_coef if inverse_coef_comparision else norm:
                print(f'norm {init_frame + frame}: {norm}')
                save_image(init_frame + frame, left_np)
        else:
            save_image(init_frame + frame, left_np)

        if not use_reference_image:
            prev_frame = current_frame
    else: 
        save_image(init_frame + frame, left_np)
    
    print(f'Frame: {frame}/{total_frames}', end='\r')
    frame += 1

print('Finish')
cam.close()
