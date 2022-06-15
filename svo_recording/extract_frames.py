import pyzed.sl as sl
import matplotlib.pyplot as plt

num_frames = 6000
every_frames = 1
total_frames = num_frames * every_frames

params = sl.InitParameters()

params.set_from_svo_file('merge.svo')
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
for frame in range(total_frames):
    err_code = cam.grab()

    if err_code != sl.ERROR_CODE.SUCCESS: 
        print('No more frames: ', repr(err_code))
        break

    if frame % every_frames != 0:
        continue

    # Retrieve data each frame
    cam.retrieve_image(left, view=sl.VIEW.LEFT, resolution=resolution)

    left_np = left.get_data()
    plt.imsave(f'frames/frame_{frame}.jpg', left_np)

print('Finish')
cam.close()
