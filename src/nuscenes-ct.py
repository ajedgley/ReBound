"""
nuscenes-ct.py

Conversion tool to bring nuscenes dataset into LVT. 
"""

import getopt
import sys
import os
from lidar.src.utils import create_ego_directory

from nuscenes.nuscenes import NuScenes

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_classes import Quaternion

from utils import create_lct_directory

from nuscenes.nuscenes import NuScenes


# Parse CLI args and validate input
def parse_options():

    input_path = ""
    output_path = ""
    parse_options = ""
    
    # Read in flags passed in with command line argument
    # Make sure that options which need an argument (namely -f for input file path and -o for output file path) have them
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:o:s:", "help")
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("use -f to specify directory of nuScenes dataset")
            print("use -o to specify the path where the LVT dataset will go")
            print("use -s to specify the name of the scene")
            sys.exit(2)
        elif opt == "-f": #and len(opts) == 2:
            input_path = arg
        elif opt == "-o": #and len(opts) == 2:
            output_path = arg
        elif opt == "-s":
            scene_name = arg
        else:
            # Only reach here if you were passed in a single option; consider this invalid input since we need both file paths
            print("Invalid set of arguments entered. Please refer to -h flag for more information.")
            sys.exit(2)

    return (input_path, output_path, scene_name)

# Used to check if file is valid nuScenes file
def validate_io_paths(input_path, output_path):

    # First check that the input path (1) exists, and (2) is a valid nuScenes database
    try:
        nusc = NuScenes(version='v1.0-mini', dataroot=input_path, verbose=True)
    except AssertionError as error:
        print("Invalid argument passed in as nuScenes file.")
        print("DEBUG: stacktrace is as follows.", str(error))

    # Output directory path is validated in utils.create_lct_directory()
    create_lct_directory(os.getcwd(), output_path)

def extract_ego(nusc, sample, frame_num, output_path):
    poserecord = nusc.get('ego_pose', sample['ego_pose_token'])

    full_path = os.path.join(os.getcwd(), output_path)
    create_ego_directory(full_path, frame_num, poserecord['translation'], poserecord['rotation'])

# Driver for nuscenes conversion tool
if __name__ == "__main__":

    # Read in input database and output directory paths
    (input_path, output_path, scene_name) = parse_options()

    # Debug print statement to check that they were read in correctly
    # print(input_path, output_path)

    # Validate whether the database path passed in is valid and if the output directory path is valid
    # If the output directory exists, then use that directory. Otherwise, create a new directory at the
    # specified path. 
    

    #Hard code scene name for now:
    scene_name = "scene-0061"

    validate_io_paths(input_path, output_path)
    nusc = NuScenes('v1.0-mini', input_path, True)
    nusc.list_scenes()
    scene_token = nusc.field2token('scene', 'name', scene_name)[0]
    scene = nusc.get('scene', scene_token)
    sample = nusc.get('sample', scene['first_sample_token'])
    frame_num = 0
    while sample['next'] != '':
        #CALL FUNCTIONS HERE. the variable 'sample' is the frame
        extract_ego(nusc, sample, frame_num, output_path)
        frame_num += 1
        sample = nusc.get('sample', sample['next'])

    