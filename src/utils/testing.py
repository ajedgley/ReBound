"""
testing.py

Testing functions to validate LVT directories

"""
import os
import sys
import json
import PIL
import numpy as np
from shutil import copyfile
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R
import math

def is_lct_directory(path):
    """Tests to see if specified directory conforms to LCT spec
    Args:
        path: path to LCT directory
    Returns:
        is_verified: True if LCT directory is valid or False if not
    """

    # Individual verification bools
    cameras_exist = os.path.exists(os.path.join(path, "cameras"))
    inside_cameras_valid = check_inside_cameras(os.path.join(path, "cameras"))
    pointcloud_exists = os.path.exists(os.path.join(path, "pointcloud"))
    inside_pointcloud_valid = check_inside_pointcloud(os.path.join(path, "pointcloud"))
    bounding_exists = os.path.exists(os.path.join(path, "bounding"))
    inside_bounding_valid = check_inside_bounding(os.path.join(path, "bounding"))
    ego_exists = os.path.exists(os.path.join(path, "ego"))
    inside_ego_valid = check_inside_ego(os.path.join(path, "ego"))
    predicted_exists = os.path.exists(os.path.join(path, "pred_bounding"))


    # Overall verification bool
    is_verified = True

    # Provides feedback to the user
    if not cameras_exist:
        print("There is no directory named \"cameras\" at the selected path.\n")
        is_verified = False
    if not inside_cameras_valid:
        is_verified = False
    if not pointcloud_exists:
        print("There is no directory named \"pointcloud\" at the selected path. \n")
        is_verified = False
    if not inside_pointcloud_valid:
        is_verified = False
    if not bounding_exists:
        print("There is no directory named \"bounding\" at the selected path. \n")
        is_verified = False
    if not inside_bounding_valid:
        is_verified = False
    if not ego_exists:
        print("There is no directory named \"ego\" at the selected path. \n")
        is_verified = False
    if not inside_ego_valid:
        is_verified = False
    if not predicted_exists:
        print("There is no directory named \"pred_bounding\" at the selected path. \n")
        is_verified = False

    return is_verified



def check_inside_cameras(path):
    """Checks to make sure that all the subdirectories of cameras only have Extrinsic.json, Intrinsic.json and .jpg files
    Prints out reason for invalidity if one exists
    Args:
        path: path to cameras directory
    Returns:
        is_verified: false if not valid and true otherwise
    """

    is_verified = True
    
    # Cameras
    for dir in os.listdir(path):
        has_in = False
        has_ex = False
        has_jpg = True
        # Files in cameras
        for file in os.listdir(os.path.join(path, dir)):
            if file == "Extrinsics.JSON": 
                if not has_ex: 
                    has_ex = True
                else:
                    is_verified = False
                    print("directory " + dir + " has multiple Extrinsics.JSON files")
            elif file == "Intrinsics.JSON":
                if not has_in:
                    has_in = True
                else:
                    is_verified = False
                    print("directory " + dir + " has multiple Intrinsics.JSON files")
            else:
                extension = file[-4:]
                if extension != ".jpg":
                    is_verified = False
                    print("There are files in " + dir + "that are not Intrinsics.JSON, Extrinsics.JSON or .jpgs")
    
    return is_verified


def check_inside_pointcloud(path):
    """Checks to make sure that all the subdirectories of pointcloud only have .pcd files
    Prints out reason for invalidity if one exists
    Args:
        path: path to pointcloud directory
    Returns:
        is_verified: false if not valid and true otherwise
    """

    is_verified = True

    for dir in os.listdir(path):
        for file in os.listdir(os.path.join(path, dir)):
            extension = file[-4:]
            if extension != ".pcd":
                is_verified = False
                print("There is a file in " + dir + " that is not a .pcd file")
    
    return is_verified

def check_inside_bounding(path):
    """Checks to make sure that all the subdirectories of bounding are properly formatted
       Prints out the reason for invalidity if one exists
       Args:
           path: path to bounding directory
       Returns:
           is_verified: false if not valid and true otherwise    
    """
    
    
    is_verified = True

    # Loop through frames
    for dir in os.listdir(path):
        has_description = False
        has_boxes = False
        # Loop through files in frame
        for file in os.listdir(os.path.join(path, dir)):
            if has_boxes and has_description:
                is_verified = False
                print("directory " + dir + " has multiple description.JSON/boxes.json files")
            elif file == "description.json":
                has_description = True
            elif file == "boxes.json":
                has_boxes = True
            else:
                is_verified = False
        
        if not has_boxes and not has_description:
            is_verified = False
            print("There is not a description.json and a boxes.json file in the " + dir + " directory")

    return is_verified

def check_inside_ego(path):
    """Checks to makes sure that all the subdirectories of ego are properly formatted
       Prints out the reason for invalidity if one exists
       Args:
            path: path to the ego directory
       Returns: 
            is_verified: false if not valid and true otherwise 
    """

    is_verified = True

    for file in os.listdir(path):
        if not file[-4:] == ".JSON":
            is_verified = False
            print("There is a file in the ego directory that is not a json file")

    return is_verified