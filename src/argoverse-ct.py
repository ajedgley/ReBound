"""
argoverse-ct.py

Conversion tool for bringing data from the argoverse dataset into our generic data format
"""
import getopt
import os
import pandas as pd
import numpy as np
import re
import sys
from utils import dataformat_utils
from utils import geometry_utils
from pyquaternion import Quaternion

'''class CurrentState:
    def __init__(self, cam_timestamp, ann_timestamp, frame_num):
        self.cam_timestamp = cam_timestamp
        self.ann_timestamp = ann_timestamp
        self.frame_num = frame_num'''

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
def extract_rgb(frame_num, timestamp, output_path, input_path):
    # TODO: need correct frame_num
    for camera in camera_list:
        timestamps = []
        for file in os.scandir(input_path+camera):
            res = re.match(r"(\d+)[.]jpg",file.name)
            if not res:
                continue
            timestamps.append(res[1])
        m=min(timestamps, key=lambda x:abs(int(x)-int(timestamp)))
        dataformat_utils.add_rgb_frame_from_jpg(output_path, camera, frame_num, input_path+camera+"/"+m+".jpg")

# Process pointcloud sub dir
def extract_lidar(frame_num, timestamp, output_path, input_path):
    # TODO: need correct frame_num, clarify lidar_up vs lidar_down
    # # Hardcoded path
    lidar = pd.read_feather(input_path+timestamp+".feather")
    points = list(zip(lidar.x, lidar.y, lidar.z))
    dataformat_utils.add_lidar_frame(output_path, "lidar", frame_num, points)

# Process ego sub dir
def extract_ego(frame_num, timestamp, output_path, input_path):
    # TODO: need correct frame_num and translation
    # Temporarily took from 0th index, look into how to get translation
    egovehicle = pd.read_feather(input_path+"city_SE3_egovehicle.feather")
    ego = egovehicle[egovehicle["timestamp_ns"] == int(timestamp)]
    rotation = ego[["qw","qx","qy","qz"]].values[0].tolist()
    translation = ego[["tx_m","ty_m","tz_m"]].values[0].tolist()
    dataformat_utils.create_ego_directory(output_path, frame_num, translation, rotation)

# This function will extract and convert the bounding boxes from Argoverse 2's annotations.feather file into the LVT format. 
def extract_bounding(annotations, frame_num, timestamp, output_path, input_path):
    ''' Each row in the annotations.feather file in the Argoverse 2 dataset has the values of 

        Index: The current annotation

        timestamp_ns: The timestamp from when the annotation was created (This annotation is not synced with camera timestamps)

        track_uuid: idk

        category: can be any of "ANIMAL, ARTICULATED_BUS, BICYCLE, BICYCLIST, BOLLARD, BOX_TRUCK, BUS, CONSTRUCTION_BARREL, CONSTRUCTION_CONE, DOG, 
        LARGE_VEHICLE, MESSAGE_BOARD_TRAILER, MOBILE_PEDESTRIAN_CROSSING_SIGN, MOTORCYCLE, MOTORCYCLIS, OFFICIAL_SIGNALER, PEDESTRIAN, RAILED_VEHICLE, 
        REGULAR_VEHICLE, SCHOOL_BUS, SIGN, STOP_SIGN, STROLLER, TRAFFIC_LIGHT_TRAILER, TRUCK, TRUCK_CAB, VEHICULAR_TRAILER, WHEELCHAIR, WHEELED_DEVICE, 
        WHEELED_RIDER

        length_m: length of bounding box
        width_m: width of bounding box
        height_m: height of bounding box

        qw: quaternion w value
        qx: quaternion x value
        qy: quaternion y value
        qz: quaternion z value

        tx_m: translation x from car
        ty_m: translation y from car
        tz_m: translation z from car

        num_interior_pts: idk
    '''
    origins = []
    sizes = []
    rotations = []
    annotation_names = []
    confidences = []

    # Initializes an object containing an invalid timestamps and an initial index of 0
    #state = CurrentState(-1, -1, 0)
    # Loops through every row in annotations.feather
    annotations = annotations[annotations["timestamp_ns"] == int(timestamp)]
    for annotation in annotations.itertuples():
        ''' MAY NEED TO WORK WITH LENGTH, WIDTH, AND HEIGHT TO FIT PROPERLY'''
        origins.append([annotation.tx_m, annotation.ty_m, annotation.tz_m])
        ''' THESE SHOULD BE FINE'''
        sizes.append([annotation.width_m, annotation.length_m, annotation.height_m])
        ''' THESE SHOULD BE FINE'''
        annotation_names.append(annotation.category)
        ''' IDK HOW TO WORK WITH QUATERNIONS, MAY BE INCORRECT/CORRECT'''
        quat = Quaternion(annotation.qw, annotation.qx, annotation.qy, annotation.qz)
        rotations.append(quat.q.tolist())
        ''' I BELIEVE THESE ANNOTATIONS ARE GROUND TRUTHS'''
        # Confidence set to 100 by default for ground truth data
        confidences.append(100)
    
    dataformat_utils.create_frame_bounding_directory(output_path, frame_num, origins, sizes, rotations, annotation_names, confidences)

# Main method for converting datasets
def convert_dataset(input_path, output_path, scene_name):
    # TODO: need a way to match timestamps with frame_num
    # Look into how to get translation, rotation, and intrinsic
    camera_data_int = {}
    camera_data_ext = {}
    int_df = pd.read_feather(input_path + scene_name + '/calibration/intrinsics.feather')
    ext_df = pd.read_feather(input_path + scene_name + '/calibration/egovehicle_SE3_sensor.feather')
    int_df.set_index('sensor_name', inplace=True, drop=True)
    ext_df.set_index('sensor_name', inplace=True, drop=True)
    
    for camera in camera_list:
        translation=ext_df.loc[camera][["tx_m","ty_m","tz_m"]].values.tolist()
        rotation=ext_df.loc[camera][["qw","qx","qy","qz"]].values.tolist()
        intrinsic= [
                [int_df['fx_px'][camera],0,int_df['cx_px'][camera]],
                [0, int_df['fy_px'][camera], int_df['cy_px'][camera]],
                [0,0,1]]
        dataformat_utils.create_rgb_sensor_directory(output_path+scene_name, camera, translation, rotation, intrinsic)

    # Create dir for each lidar sensor
    dataformat_utils.create_lidar_sensor_directory(output_path+scene_name, "lidar")

    # Read in annotation.feather file
    annotations = pd.read_feather(input_path+scene_name+'/annotations.feather')

    frame_num = 0
    frame_count = 0
    timestamps = []
    for file in os.scandir(input_path+scene_name+"/sensors/lidar"):
        res = re.match(r"(\d+)[.]feather",file.name)
        if not res:
            continue
        timestamps.append(res[1])
        frame_count += 1
    timestamps.sort()
    dataformat_utils.print_progress_bar(frame_num, frame_count)
    for timestamp in timestamps:
        extract_rgb(frame_num, timestamp, output_path+scene_name, input_path+scene_name+"/sensors/cameras/")
        extract_lidar(frame_num, timestamp, output_path+scene_name, input_path+scene_name+"/sensors/lidar/")
        extract_ego(frame_num, timestamp, output_path+scene_name, input_path+scene_name+"/")
        extract_bounding(annotations, frame_num, timestamp, output_path+scene_name, input_path+scene_name+"/")
        frame_num += 1
        dataformat_utils.print_progress_bar(frame_num, frame_count)

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
            

# Synchronization:
# Create a frame for each lidar sweep. Get timestamp.
# For each camera, find the closest timestamp to the lidar timestamp.
# For the annotations, the timestamp should have an exact match.