"""
argoverse-ct.py

Conversion tool for bringing data from the argoverse dataset into our generic data format
"""
import getopt
import os
import pandas as pd
import re
import sys
from utils import dataformat_utils
from pyquaternion import Quaternion

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

def parse_options():
    ''' Read in user command line input to get directory paths which will be used for input and output.
    Args:
        None
    Returns:
        input_path: Path to argoverse dataset being read into LVT
        output_path: Path where user wants LVT to generate generic data format used in program
    '''
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

def extract_rgb(frame_num, timestamp, output_path, input_path):
    ''' Extracts the RGB data from an argoverse frame and puts it in the lct file system
    Args:
        frame_num: frame number synchronized with timestamp
        timestamp: timestamp
        input_path: path to argoverse data
        output_path: path where to generate generic data format
    Returns:
        None
    '''
    for camera in camera_list:
        # For synchronization, find image closest to timestamp
        timestamps = []
        for file in os.scandir(input_path+camera):
            res = re.match(r"(\d+)[.]jpg",file.name)
            if not res:
                continue
            timestamps.append(res[1])
        m=min(timestamps, key=lambda x:abs(int(x)-int(timestamp)))
        dataformat_utils.add_rgb_frame_from_jpg(output_path, camera, frame_num, input_path+camera+"/"+m+".jpg")

def extract_lidar(frame_num, timestamp, output_path, input_path):
    ''' Extracts LiDAR data from an argoverse frame and puts it in the lct file system
    Args:
        frame_num: frame number synchronized with timestamp
        timestamp: timestamp
        input_path: path to argoverse data
        output_path: path where to generate generic data format
    Returns:
        None
    '''
    lidar = pd.read_feather(input_path+timestamp+".feather")
    points = list(zip(lidar.x, lidar.y, lidar.z))
    dataformat_utils.add_lidar_frame(output_path, "lidar", frame_num, points)

def extract_ego(frame_num, timestamp, output_path, input_path):
    ''' Extracts ego data from an argoverse frame and puts it in the lct file system
    Args:
        frame_num: frame number synchronized with timestamp
        timestamp: timestamp
        input_path: path to argoverse data
        output_path: path where to generate generic data format
    Returns:
        None
    '''
    egovehicle = pd.read_feather(input_path+"city_SE3_egovehicle.feather")
    ego = egovehicle[egovehicle["timestamp_ns"] == int(timestamp)]
    # Get rotation and translation of egovehicle
    rotation = ego[["qw","qx","qy","qz"]].values[0].tolist()
    translation = ego[["tx_m","ty_m","tz_m"]].values[0].tolist()
    dataformat_utils.create_ego_directory(output_path, frame_num, translation, rotation)

def extract_bounding(annotations, frame_num, timestamp, output_path):
    ''' Extracts the bounding data from a nuScenes frame and converts it into our intermediate format
    Args:
        annotations: dataframe containing annotations
        frame_num: frame number synchronized with timestamp
        timestamp: timestamp
        output_path: path where to generate generic data format
    Returns:
        None
    '''
    origins = []
    sizes = []
    rotations = []
    annotation_names = []
    confidences = []
    ids = []
    internal_pts = []

    # Get annotation, rotation, confidence level, quaternion, center, diminensions, uuids, and interior points of each bounding box in frame
    annotations = annotations[annotations["timestamp_ns"] == int(timestamp)]
    for annotation in annotations.itertuples():
        origins.append([annotation.tx_m, annotation.ty_m, annotation.tz_m])
        sizes.append([annotation.width_m, annotation.length_m, annotation.height_m])
        annotation_names.append(annotation.category)
        quat = Quaternion(annotation.qw, annotation.qx, annotation.qy, annotation.qz)
        rotations.append(quat.q.tolist())
        # Confidence set to 100 by default for ground truth data
        confidences.append(100)
        ids.append(annotation.track_uuid)
        internal_pts.append(annotation.num_interior_pts)
    dataformat_utils.create_frame_bounding_directory(output_path, frame_num, origins, sizes, rotations, annotation_names, confidences, ids, internal_pts)

# Main method for converting datasets
def convert_dataset(input_path, output_path):
    int_df = pd.read_feather(input_path + "calibration/intrinsics.feather")
    ext_df = pd.read_feather(input_path + "calibration/egovehicle_SE3_sensor.feather")
    int_df.set_index("sensor_name", inplace=True, drop=True)
    ext_df.set_index("sensor_name", inplace=True, drop=True)

    # Store metadata
    dataformat_utils.add_metadata(output_path, 'Argoverse', ['timestamps.csv'])
    
    # Extrinsic and intrinsic data for each camera
    for camera in camera_list:
        translation=ext_df.loc[camera][["tx_m","ty_m","tz_m"]].values.tolist()
        rotation=ext_df.loc[camera][["qw","qx","qy","qz"]].values.tolist()
        intrinsic= [
                [int_df["fx_px"][camera],0,int_df["cx_px"][camera]],
                [0, int_df["fy_px"][camera], int_df["cy_px"][camera]],
                [0,0,1]
        ]
        dataformat_utils.create_rgb_sensor_directory(output_path, camera, translation, rotation, intrinsic)

    # Create dir for each lidar sensor
    dataformat_utils.create_lidar_sensor_directory(output_path, "lidar")

    # Read in annotation.feather file
    annotations = pd.read_feather(input_path+"annotations.feather")

    # Frame number and timestamp synchronization
    frame_num = 0
    frame_count = 0
    timestamps = []
    # Create a frame for each timestamp in lidar sweeps
    for file in os.scandir(input_path+"sensors/lidar"):
        res = re.match(r"(\d+)[.]feather",file.name)
        if not res:
            continue
        timestamps.append(res[1])
        frame_count += 1
    timestamps.sort()
    dataformat_utils.print_progress_bar(frame_num, frame_count)

    # Loop through each frame
    for timestamp in timestamps:
        extract_rgb(frame_num, timestamp, output_path, input_path+"sensors/cameras/")
        extract_lidar(frame_num, timestamp, output_path, input_path+"sensors/lidar/")
        extract_ego(frame_num, timestamp, output_path, input_path)
        extract_bounding(annotations, frame_num, timestamp, output_path)
        frame_num += 1
        dataformat_utils.print_progress_bar(frame_num, frame_count)

    # Store timestamps
    pd.DataFrame(timestamps, columns=['timestamps']).to_csv(output_path + 'timestamps.csv')

if __name__ == "__main__":
    (input_path, output_path, scene_names) = parse_options()
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")

    # Verify input path exists
    if not os.path.exists(input_path):
        sys.exit("Invalid input path. Please check paths entered and try again.")
    input_path += ("" if input_path[-1] == "/" else "/")
    output_path += ("" if output_path[-1] == "/" else "/")

    # Verify scenes exists
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
        convert_dataset(input_path+scene_name+"/", output_path+scene_name+"/")