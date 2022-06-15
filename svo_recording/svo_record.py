#! /usr/bin/env 

import argparse

import pyzed.sl as sl

parser = argparse.ArgumentParser(description="Records video from ZED Camera")
parser.add_argument("-o", metavar="output_file", type=str, help="Output directory", required=True)
parser.add_argument("-n", metavar="fps_grab", type=int, help="How many fps to grab", required=True)
parser.add_argument("--resolution",
                    type=str,
                    default="HD1080",
                    choices=["HD2K", "HD1080", "HD720", "VGA"],
                    help="Recording resolution")
parser.add_argument("--fps",
                    type=int,
                    default=30,
                    choices=[15, 30, 60, 100],
                    help="Recording FPS")
parser.add_argument("--compression",
                    type=str,
                    default="H264",
                    choices=["LOSSLESS", "H264", "H265"],
                    help="Compression mode")
parser.add_argument("--cycles",
                    type=int,
                    default=1,
                    help="Recording cycles")

def main():
    args = parser.parse_args()
    
    output_path = args.o
    resolution = args.resolution
    fps= args.fps
    compression = args.compression
    num_grabs = args.n
    num_cycles = args.cycles

    zed = sl.Camera()

    init_params = sl.InitParameters()
    init_params.camera_resolution = getattr(sl.RESOLUTION, resolution)
    init_params.camera_fps = fps
    init_params.coordinate_units = sl.UNIT.METER

    err = zed.open(init_params)

    if err != sl.ERROR_CODE.SUCCESS:
        print("Error opening camera: ", repr(err))
        exit(1)

    for cycle in range(num_cycles):
        o_with_cycles = "c{}__{}".format(cycle, output_path)

        record_params = sl.RecordingParameters(o_with_cycles, getattr(sl.SVO_COMPRESSION_MODE, compression))
        err = zed.enable_recording(record_params)

        if err != sl.ERROR_CODE.SUCCESS:
            print("Error init recording: ", repr(err))
            exit(1)

        print(f"recording... {cycle+1}/{num_cycles}")
        for i in range(num_grabs):
            # Each new frame is added to the SVO file
            zed.grab()
            if i % 50 == 0:
                print(f"Recorded: {i+1}/{num_grabs}", flush=True, end='\r')

        # Disable recording
        print()
        print(f"...done {cycle+1}/{num_cycles}")
        zed.disable_recording()


if __name__ == "__main__":
    main()
