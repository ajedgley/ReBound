import unittest
import os
import re
import pandas as pd
import json
import numpy as np
import sys
import getopt
from deepdiff import DeepDiff # pip install deepdiff
from nuscenes.nuscenes import NuScenes

# These tests are to be run after the nuscenes -> LVT -> nuscenes pipeline, 
# without any edits, to ensure data is retained

global input_path
global output_path
global scene_names
global pred_path
global ver_name

def parse_options():
    """Read in user command line input to get directory paths which will be used for input and output.
    Args:
        None
    Returns:
        input_path: Path to NuScenes dataset being read into LVT
        output_path: Path where user wants LVT to generate generic data format used in program
        scene_name: Name of the scene in NuScenes
        pred_path: Path to data based on a model's predictions
        """
    # We can make scene_names a list so that it either includes the name of one scene or every scene a user wants to batch process 
    input_path = ""
    output_path = ""
    scene_names = []
    pred_path = ""
    ver_name = ""


    # Read in flags passed in with command line argument
    # Make sure that options which need an argument (namely -f for input file path and -o for output file path) have them
    # User is able to specify -h, -f, -o, -s, and -r options
    # -h brings up help menu
    # -f is used to specify the path to the Waymo file you want to read in and requires one arg. If -r is specified then this arg
    # corresponds to a directory containing all the .tfrecord files you'd like to read in
    # -o is used to specify the path to the directory where the LVT format will go.
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "hf:o:s:p:rv:", "help")
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("REQUIRED: -f to specify directory of nuScenes dataset")
            print("REQUIRED: -o to specify the path where the LVT dataset will go")
            print("OPTIONAL: -p to specify a path to projected data")
            print("REQUIRED: -v to specify the version of this dataset eg: v1.0-mini")
            sys.exit(2)
        elif opt == "-f":
            input_path = arg
            input_string = input("Please enter a comma-delinated list of scene names. Remove any whitespace. To convert all scenes, leave blank\n")
            if len(input_string) != 0:
                list_of_scenes = input_string.split(",")
                scene_names.extend(list_of_scenes)
        elif opt == "-o":
            output_path = arg
        elif opt == "-p":
            pred_path = arg
        elif opt == "-v":
            ver_name = arg

        else:
            print("Invalid set of arguments entered. Please refer to -h flag for more information.")
            sys.exit(2)

    return (input_path, output_path, scene_names, pred_path, ver_name)

# Used to check if file is valid nuScenes file
def validate_input_path(input_path, ver_name):
    """Verify that input path given to nuscenes database is valid input
    Args:
        input_path: Path to check before reading into LVT generic format
    Returns:
        True on valid input and False on invalid input
        """

    # Check that the input path exists and is a valid nuScenes database
    try:
        # If input path is invalid as nuScenes database, this constructor will throw an AssertationError
        NuScenes(version=ver_name, dataroot=input_path, verbose=True)
        return True
    except AssertionError as error:
        print("Invalid argument passed in as nuScenes file.")
        return False

class TestNuscenesConversion(unittest.TestCase):
    def test_attribute(self):
        in_str = str(input_path + "/attribute.json")
        out_str = str(output_path + "/attribute.json")

        with open(in_str) as in_file:
            json_dict1 = json.load(in_file)
        with open(out_str) as out_file:
            json_dict2 = json.load(out_file)

        diff = DeepDiff(json_dict1, json_dict2, ignore_order=True)
        self.assertTrue((diff == {}), diff)

    def test_calibrated_sensor(self):
        in_str = str(input_path + "/calibrated_sensor.json")
        out_str = str(output_path + "/calibrated_sensor.json")

        with open(in_str) as in_file:
            json_dict1 = json.load(in_file)
        with open(out_str) as out_file:
            json_dict2 = json.load(out_file)

        diff = DeepDiff(json_dict1, json_dict2, ignore_order=True)
        self.assertTrue((diff == {}), diff)

    def test_category(self):
        in_str = str(input_path + "/category.json")
        out_str = str(output_path + "/category.json")

        with open(in_str) as in_file:
            json_dict1 = json.load(in_file)
        with open(out_str) as out_file:
            json_dict2 = json.load(out_file)

        diff = DeepDiff(json_dict1, json_dict2, ignore_order=True)
        self.assertTrue((diff == {}), diff)
    
    def test_ego_pose(self):
        in_str = str(input_path + "/ego_pose.json")
        out_str = str(output_path + "/ego_pose.json")

        with open(in_str) as in_file:
            json_dict1 = json.load(in_file)
        with open(out_str) as out_file:
            json_dict2 = json.load(out_file)

        diff = DeepDiff(json_dict1, json_dict2, ignore_order=True)
        self.assertTrue((diff == {}), diff)
    
    def test_instance(self):
        in_str = str(input_path + "/instance.json")
        out_str = str(output_path + "/instance.json")

        with open(in_str) as in_file:
            json_dict1 = json.load(in_file)
        with open(out_str) as out_file:
            json_dict2 = json.load(out_file)

        diff = DeepDiff(json_dict1, json_dict2, ignore_order=True)
        self.assertTrue((diff == {}), diff)
    
    def test_log(self):
        in_str = str(input_path + "/log.json")
        out_str = str(output_path + "/log.json")

        with open(in_str) as in_file:
            json_dict1 = json.load(in_file)
        with open(out_str) as out_file:
            json_dict2 = json.load(out_file)

        diff = DeepDiff(json_dict1, json_dict2, ignore_order=True)
        self.assertTrue((diff == {}), diff)
    
    def test_map(self):
        in_str = str(input_path + "/map.json")
        out_str = str(output_path + "/map.json")

        with open(in_str) as in_file:
            json_dict1 = json.load(in_file)
        with open(out_str) as out_file:
            json_dict2 = json.load(out_file)

        diff = DeepDiff(json_dict1, json_dict2, ignore_order=True)
        self.assertTrue((diff == {}), diff)
    
    def test_sample_annotation(self):
        in_str = str(input_path + "/sample_annotation.json")
        out_str = str(output_path + "/sample_annotation.json")

        with open(in_str) as in_file:
            json_dict1 = json.load(in_file)
        with open(out_str) as out_file:
            json_dict2 = json.load(out_file)

        diff = DeepDiff(json_dict1, json_dict2, ignore_order=True)
        self.assertTrue((diff == {}), diff)
    
    def test_sample_data(self):
        in_str = str(input_path + "/sample_data.json")
        out_str = str(output_path + "/sample_data.json")

        with open(in_str) as in_file:
            json_dict1 = json.load(in_file)
        with open(out_str) as out_file:
            json_dict2 = json.load(out_file)

        diff = DeepDiff(json_dict1, json_dict2, ignore_order=True)
        self.assertTrue((diff == {}), diff)
    
    def test_sample(self):
        in_str = str(input_path + "/sample.json")
        out_str = str(output_path + "/sample.json")

        with open(in_str) as in_file:
            json_dict1 = json.load(in_file)
        with open(out_str) as out_file:
            json_dict2 = json.load(out_file)

        diff = DeepDiff(json_dict1, json_dict2, ignore_order=True)
        self.assertTrue((diff == {}), diff)
    
    def test_scene(self):
        in_str = str(input_path + "/scene.json")
        out_str = str(output_path + "/scene.json")

        with open(in_str) as in_file:
            json_dict1 = json.load(in_file)
        with open(out_str) as out_file:
            json_dict2 = json.load(out_file)

        diff = DeepDiff(json_dict1, json_dict2, ignore_order=True)
        self.assertTrue((diff == {}), diff)
    
    def test_sensor(self):
        in_str = str(input_path + "/sensor.json")
        out_str = str(output_path + "/sensor.json")

        with open(in_str) as in_file:
            json_dict1 = json.load(in_file)
        with open(out_str) as out_file:
            json_dict2 = json.load(out_file)

        diff = DeepDiff(json_dict1, json_dict2, ignore_order=True)
        self.assertTrue((diff == {}), diff)
    
    def test_visibility(self):
        in_str = str(input_path + "/visibility.json")
        out_str = str(output_path + "/visibility.json")

        with open(in_str) as in_file:
            json_dict1 = json.load(in_file)
        with open(out_str) as out_file:
            json_dict2 = json.load(out_file)

        diff = DeepDiff(json_dict1, json_dict2, ignore_order=True)
        self.assertTrue((diff == {}), diff)

if __name__ == '__main__':
    # Read in input database and output directory paths
    (input_path, output_path, scene_names, pred_path, ver_name) = parse_options()
    
    # Validate whether the database path passed in is valid and if the output directory path is valid
    # If the output directory exists, then use that directory. Otherwise, create a new directory at the
    # specified path. 
    if ver_name == "":
        sys.exit("No version name given")
    
    if not validate_input_path(input_path, ver_name):
        sys.exit("Invalid input path or version name specified. Please check paths or version name entered and try again")
    
    pred_data = {}
    if pred_path != "":
        pred_data = json.load(open(pred_path))

    nusc = NuScenes(ver_name, input_path, True)

    path = os.getcwd()

    sys.argv = sys.argv[:1]  
    unittest.main()