import argparse
import os

import pyzed.sl as sl
import matplotlib.pyplot as plt

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
                    help="Frame to estart counting from",
                    type=int,
                    required=False,
                    default=0)

args = parser.parse_args()

num_frames = args.frames
init_frame = args.init_frame
svo_file = args.input
output_path = args.output_path

every_frames = 1
total_frames = num_frames * every_frames if num_frames > 0 else -1

if not os.path.exists(output_path):
    os.mkdir(output_path)

params = sl.InitParameters()

params.set_from_svo_file(svo_file)
params.svo_real_time_mode = True
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
    plt.imsave(f'{output_path}/frame_{init_frame + frame}.jpg', left_np)
    print(f'Frame: {frame}/{total_frames}', end='\r')
    frame += 1

print('Finish')
cam.close()
