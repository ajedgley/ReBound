"""
argoverse-ct.py

Conversion tool for bringing data from the argoverse dataset into our generic data format
"""
from __future__ import annotations
from utils import dataformat_utils
import getopt
import os
import pyarrow.feather as feather
import sys

# Name of all the cameras
camera_list = ["ring_front_center",
                "ring_front_left",
                "ring_front_right",
                "ring_rear_left",
                "ring_rear_right",
                "ring_side_left",
                "ring_side_right",
                "stereo_front_left",
                "stereo_front_right"]

# Command line parsing
def parse_options():
    """Read in user command line input to get directory paths which will be used for input and output.
    """
    input_path = ""
    output_path = ""
    scene_names = []

    try:
        opts, _ = getopt.getopt(sys.argv[1:], "hf:o:", "help")
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("REQUIRED: -f to specify the path to the Argoverse file.")
            print("REQUIRED: -o to specify the name of the directory where the LVT format will go. Will be a folder in the current directory.")
            sys.exit(2)
        elif opt == "-f":
            input_path = arg
            input_string = input("Please enter a comma-delinated list of scene names. Remove any whitespace. To convert all scenes, leave blank.\n")
            if len(input_string) != 0:
                list_of_scenes = input_string.split(",")
                scene_names.extend(list_of_scenes)
        elif opt == "-o":
            output_path = arg
        else:
            sys.exit(2)

    return (input_path, output_path, scene_names)

# Process camera sub dir
def extract_rgb(frame_num, output_path, input_path):
    # TODO: need correct frame_num
    for camera in camera_list:
        # Hardcoded path
        try:
            dataformat_utils.add_rgb_frame_from_jpg(output_path, camera, frame_num, input_path+camera+"/315969335099927219.jpg")
        except:
            pass

# Process pointcloud sub dir
def extract_lidar(frame_num, output_path, input_path):
    # TODO: need correct frame_num, clarify lidar_up vs lidar_down
    # Hardcoded path
    try:
        lidar = feather.read_feather(input_path+"315969335159945000.feather")
        up = lidar[lidar["laser_number"] < 32]
        points = list(zip(up.x, up.y, up.z))
        dataformat_utils.add_lidar_frame(output_path, "lidar_up", frame_num, points)
    except:
        pass

# Process ego sub dir
def extract_ego(frame_num, output_path, input_path):
    # TODO: need correct frame_num and translation
    # Temporarily took from 0th index, look into how to get translation
    egovehicle = feather.read_feather(input_path+"city_SE3_egovehicle.feather")
    rotation=list(egovehicle.loc[0][["qw","qx","qy","qz"]])
    translation=list(egovehicle.loc[0][["tx_m","ty_m","tz_m"]])
    dataformat_utils.create_ego_directory(output_path, frame_num, translation, rotation)

# Process bounding sub dir
def extract_bounding(frame_num, output_path, input_path):
    # TODO: need correct frame_num, origins, and rotations
    origins=[]
    sizes=[]
    rotations=[]
    annotation_names=[]
    confidences=[]

    # Look into how to get origins and rotations
    annotations = feather.read_feather(input_path+'annotations.feather')
    for i in annotations.index:
        row = annotations.loc[i]
        origins.append(list(row[["tx_m","ty_m","tz_m"]]))
        sizes.append(list(row[["length_m","width_m","height_m"]]))
        rotations.append(list(row[["qw","qx","qy","qz"]]))
        annotation_names.append(row["category"])
        confidences.append(100)

    dataformat_utils.create_frame_bounding_directory(output_path, frame_num, origins, sizes, rotations, annotation_names, confidences)

# Main method for converting datasets
def convert_dataset(input_path, output_path, scene_name):
    # TODO: need a way to match timestamps with frame_num
    # Look into how to get translation, rotation, and intrinsic
    for camera in camera_list:
        translation=None
        rotation=None
        intrinsic=None
        dataformat_utils.create_rgb_sensor_directory(output_path+scene_name, camera, translation, rotation, intrinsic)

    # Create dir for each lidar sensor
    dataformat_utils.create_lidar_sensor_directory(output_path+scene_name, "lidar_up")
    dataformat_utils.create_lidar_sensor_directory(output_path+scene_name, "lidar_down")

    # Temporary frame_num
    for frame_num in range(1):
        extract_rgb(frame_num, output_path+scene_name, input_path+scene_name+"/sensors/cameras/")
        extract_lidar(frame_num, output_path+scene_name, input_path+scene_name+"/sensors/lidar/")
        extract_ego(frame_num, output_path+scene_name, input_path+scene_name+"/")
        extract_bounding(frame_num, output_path+scene_name, input_path+scene_name+"/")

if __name__ == "__main__":
    (input_path, output_path, scene_names) = parse_options()
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")

    if not os.path.exists(input_path):
        sys.exit("Invalid input path. Please check paths entered and try again.")

    if len(scene_names) == 0:
        print("Converting All Scenes...")
        for scene in os.scandir(input_path):
            if scene.is_dir():
                scene_names.append(scene.name)
        print(scene_names)
    
    # Create all sub dirs
    for scene_name in scene_names:
        dataformat_utils.create_lct_directory(output_path, scene_name)
    
    # Convert all the scenes
    for scene_name in scene_names:
        convert_dataset(input_path, output_path, scene_name)