import argparse
import os
import cv2

import pyzed.sl as sl
import numpy as np

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
parser.add_argument("--motion_coef",
                    help="Coef to motion threshold detection from 0 to 255",
                    type=float,
                    required=False,
                    default=0)
parser.add_argument("--crop_image",
                    help="Data to crop de image, selects the area to compute the motion detection: <x1,x2,y1,y2>",
                    nargs=4,
                    required=False,
                    default=None)
parser.add_argument("--save_motion_detections",
                    help="only save detected motion maps",
                    action='store_true',
                    default=False)

args = parser.parse_args()

num_frames = args.frames
init_frame = args.init_frame
svo_file = args.input
output_path = args.output_path

motion_coef = args.motion_coef
save_motion_detections = args.save_motion_detections
motion_kernel = np.ones((5, 5))
do_motion_detection = motion_coef != 0

crop_x1 = 0
crop_x2 = 0
crop_y1 = 0
crop_y2 = 0
do_crop_image = args.crop_image != None
if do_crop_image:
    crop_x1, crop_x2, crop_y1, crop_y2 = [int(i) for i in args.crop_image]

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
    cv2.imwrite(f'{output_path}/frame_{frame_num}.jpg', img_data)

prev_frame = None
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

    if do_motion_detection:
        current_frame = left_np[crop_y1:crop_y2,crop_x1:crop_x2] if do_crop_image else left_np
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_BGRA2GRAY)
        
        if prev_frame is not None:
            diff_frame = cv2.absdiff(src1=prev_frame, src2=current_frame)
            diff_frame = cv2.dilate(diff_frame, motion_kernel, 1)
            thresh = cv2.threshold(src=diff_frame, thresh=motion_coef, maxval=255, type=cv2.THRESH_BINARY)[1]

            if save_motion_detections:
                save_image(init_frame + frame, thresh)
            elif thresh.any():
                print(f'Detected movement on frame {init_frame + frame}')
                save_image(init_frame + frame, left_np)
        else:
            save_image(init_frame + frame, left_np)

        prev_frame = current_frame
    else: 
        save_image(init_frame + frame, left_np)
    
    print(f'Frame: {frame}/{total_frames}', end='\r')
    frame += 1

print('Finish')
cam.close()
