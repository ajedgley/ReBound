import unittest
import os
import re
import pandas as pd
import json
import numpy as np
import sys
import getopt

# These tests are to be run after through the Argo -> LVT -> Argo pipeline, 
# without any edits, to ensure data is retained

input_path = ""
output_path = ""
scene_names = []

def parse_options():
    ''' Read in user command line input to get directory paths for the input and output of the conversion pipeline.
    Args:
        None
    Returns:
        input_path: Path to original argoverse dataset
        output_path: Path to "converted" argoverse dataset, which should have the same information
    '''

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
            input_string = input("Please enter a comma-delinated list of scene names. Remove any whitespace. To use all scenes, leave blank.\n")
            if len(input_string) != 0:
                list_of_scenes = input_string.split(",")
                scene_names.extend(list_of_scenes)
        elif opt == "-o":
            output_path = arg
        else:
            sys.exit(2)

    return (input_path, output_path, scene_names)

class TestArgoConversion(unittest.TestCase):
    # tests that all camera extrinsic data is retained
    def test_cameras_extrinsics_retained(self):
        for scene_name in scene_names:
            original_df = pd.read_feather(input_path + scene_name + '/calibration/egovehicle_SE3_sensor.feather')
            new_df = pd.read_feather(output_path + scene_name + '/calibration/egovehicle_SE3_sensor.feather')
            
            if not original_df.equals(new_df):
                self.assertTrue(False, output_path + scene_name + " did not retain data from " + input_path + scene_name)
        self.assertTrue(True, "camera extrinsics not retained")
    
    # tests that all camera intrinsic data is retained
    def test_cameras_intrinsics_retained(self):
        for scene_name in scene_names:
            original_df = pd.read_feather(input_path + scene_name + '/calibration/intrinsics.feather')
            new_df = pd.read_feather(output_path + scene_name + '/calibration/intrinsics.feather')

            if not original_df.equals(new_df):
                self.assertTrue(False, output_path + scene_name + " did not retain data from " + input_path + scene_name)
        self.assertTrue(True, "camera intrinsics not retained")
    
    # tests that all data from map folder json files are retained
    def test_map_json_retained(self):
        for scene_name in scene_names:
            input_files = []
            for file in os.scandir(input_path+scene_name+"/map"):
                res = re.match(r"(a|log_map_archive).*[.]json",file.name)
                if not res:
                    continue
                else:
                    input_files.append(file.name)
            input_files.sort()

            output_files = []
            for file in os.scandir(output_path+scene_name+"/map"):
                res = re.match(r"(a|log_map_archive).*[.]json",file.name)
                if not res:
                    continue
                else:
                    output_files.append(file.name)
            output_files.sort()
        
            for i in range(len(input_files)):
                if input_files[i] == output_files[i]:
                    in_str = str(input_path + scene_name + '/map/' + input_files[i])
                    out_str = str(output_path + scene_name + '/map/' + output_files[i])

                    with open(in_str) as in_file:
                        json_dict1 = json.load(in_file)
                    with open(out_str) as out_file:
                        json_dict2 = json.load(out_file)
                    
                    if sorted(json_dict1.items()) != sorted(json_dict2.items()):
                        self.assertTrue(False, out_str + " did not retain data from " + in_str)
                else:
                    self.assertTrue(False, scene_name + " map json file names do not match")

        self.assertTrue(True, "map imgSim2 not retained")

    # tests that all data from the map npy file is retained
    def test_map_npy_retained(self):
        for scene_name in scene_names:
            input_files = []
            for file in os.scandir(input_path+scene_name+"/map"):
                res = re.match(r"a.*[.]npy",file.name)
                if not res:
                    continue
                else:
                    input_files.append(file.name)
            input_files.sort()

            output_files = []
            for file in os.scandir(output_path+scene_name+"/map"):
                res = re.match(r"a.*[.]npy",file.name)
                if not res:
                    continue
                else:
                    output_files.append(file.name)
            output_files.sort()
        
            for i in range(len(input_files)):
                if input_files[i] == output_files[i]:
                    in_str = str(input_path + scene_name + '/map/' + input_files[i])
                    out_str = str(output_path + scene_name + '/map/' + output_files[i])
            
                    if not np.array_equal(np.load(in_str), np.load(out_str), equal_nan=True):
                        self.assertTrue(False, out_str + " did not retain data from " + in_str)
                else:
                    self.assertTrue(False, scene_name + " map npy file names do not match")

        self.assertTrue(True, "map ground_height_surface not retained")        
    
    # tests that all data from the lidar files are retained
    # takes the longest if multiple scenes are given
    def test_lidar_retained(self):
        for scene_name in scene_names:
            timestamps = []
            # Create a frame for each timestamp in lidar sweeps
            for file in os.scandir(input_path+ scene_name +"/sensors/lidar"):
                res = re.match(r"(\d+)[.]feather",file.name)
                if not res:
                    continue
                timestamps.append(res[1])
            timestamps.sort()
            for timestamp in timestamps:
                original_df = pd.read_feather(input_path + scene_name + '/sensors/lidar/' + timestamp + '.feather')
                new_df = pd.read_feather(output_path + scene_name + '/sensors/lidar/' + timestamp + '.feather')
                
                if not original_df.equals(new_df):
                    self.assertTrue(False, scene_name + " " + timestamp + " lidar data not retained")
        self.assertTrue(True, "lidar data not retained")
    
    # tests that all data from the annotation file is retained
    def test_annotations_retained(self):
        for scene_name in scene_names:
            original_df = pd.read_feather(input_path + scene_name + '/annotations.feather')
            new_df = pd.read_feather(output_path + scene_name + '/annotations.feather')
            original_df.set_index("track_uuid", inplace=True, drop=True)
            new_df.set_index("track_uuid", inplace=True, drop=True)
            original_df.sort_index()
            new_df.sort_index()

            if not original_df.equals(new_df):
                self.assertTrue(False, output_path + scene_name + " annotations not retained from " + input_path + scene_name)
        
        self.assertTrue(True, "annotations not retained")

    # tests that all data from the egovehicle file is retained
    def test_egovehicle_retained(self):
        for scene_name in scene_names:
            original_df = pd.read_feather(input_path + scene_name + '/city_SE3_egovehicle.feather')
            new_df = pd.read_feather(output_path + scene_name + '/city_SE3_egovehicle.feather')
        
            if not original_df.equals(new_df):
                self.assertTrue(False, output_path + scene_name + " egovehicle data not retained from " + input_path + scene_name)
        
        self.assertTrue(True, "egovehicle data not retained")

if __name__ == '__main__':
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
        print("Testing All Scenes...")
        for scene in os.scandir(input_path):
            if scene.is_dir():
                scene_names.append(scene.name)
        print(scene_names)

    sys.argv = sys.argv[:1]  
    unittest.main()