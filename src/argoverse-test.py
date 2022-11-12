import unittest
import os
import re
import argparse
import pandas as pd
import json
import shutil
import numpy as np

# TODO: unfinished, this is essentially a sketch of a program
# These tests are to be run directly through the Argo -> LVT -> Argo pipeline, 
# without any edits, to ensure data is retained

# executed from LVT, TODO: will change to cmd line input later
input_path = '../../mini-sensor/'
output_path = '../../test-out/'
scene_names = [
    'a33a44fb-6008-3dc2-b7c5-2d27b70741e8',
    'a7636fca-4d9e-3052-bef2-af0ce5d1df74',
    'a91d4c7b-bf55-3a0e-9eba-1a43577bcca8',
    'adf9a841-e0db-30ab-b5b3-bf0b61658e1e']

# TODO: change to work for all scene names
scene_name = scene_names[0]

# TODO: change all functions to test for equality regardless of order?
class TestArgoConversion(unittest.TestCase):
    # tests that all camera extrinsic data is retained
    def test_cameras_extrinsics_retained(self):
        original_df = pd.read_feather(input_path + scene_name + '/calibration/egovehicle_SE3_sensor.feather')
        new_df = pd.read_feather(output_path + scene_name + '/calibration/egovehicle_SE3_sensor.feather')

        self.assertTrue(original_df.equals(new_df), "camera extrinsics not retained")
    
    # tests that all camera intrinsic data is retained
    def test_cameras_intrinsics_retained(self):
        original_df = pd.read_feather(input_path + scene_name + '/calibration/intrinsics.feather')
        new_df = pd.read_feather(output_path + scene_name + '/calibration/intrinsics.feather')

        self.assertTrue(original_df.equals(new_df), "camera intrinsics not retained")
    
    # tests that all data from the imgSim2 map json file is retained (i don't actually know what this is)
    def test_map_imgSim2_city_retained(self):
        in_str = str(input_path + scene_name + '/map/' + scene_name + '___img_Sim2_city.json')
        with open(in_str) as in_file:
            json_dict1 = json.load(in_file)
        
        out_str = str(output_path + scene_name + '/map/' + scene_name + '___img_Sim2_city.json')
        with open(out_str) as out_file:
            json_dict2 = json.load(out_file)
  
        self.assertTrue(sorted(json_dict1.items()) == sorted(json_dict2.items()), "map imgSim2 not retained")
    
    # tests that all data from the height_surface_PIT map npy file is retained (i don't actually know what this is)
    def test_map_ground_height_surface_PIT_retained(self):
        in_str = str(input_path + scene_name + '/map/' + scene_name + '_ground_height_surface____PIT.npy')
        out_str = str(output_path + scene_name + '/map/' + scene_name + '_ground_height_surface____PIT.npy')
        
        self.assertTrue(np.array_equal(np.load(in_str), np.load(out_str), equal_nan=True), "map ground_height_surface not retained")
    
    # tests that all data from the archive_PIT_city map json file is retained (i don't actually know what this is)
    def test_log_map_archive_PIT_city_retained(self):
        in_str = str(input_path + scene_name + '/map/' + 'log_map_archive_' + scene_name + '____PIT_city_62578.json')
        with open(in_str) as in_file:
            json_dict1 = json.load(in_file)
        
        out_str = str(output_path + scene_name + '/map/' + 'log_map_archive_' + scene_name + '____PIT_city_62578.json')
        with open(out_str) as out_file:
            json_dict2 = json.load(out_file)
  
        self.assertTrue(sorted(json_dict1.items()) == sorted(json_dict2.items()), "map imgSim2 not retained")
    
    # tests that all data from the lidar files are retained
    def test_lidar_retained(self):
        # TODO: change to be for all timestamps
        timestamp = '315967267360053000'
        original_df = pd.read_feather(input_path + scene_name + '/sensors/lidar/' + timestamp + '.feather')
        new_df = pd.read_feather(output_path + scene_name + '/sensors/lidar/' + timestamp + '.feather')

        self.assertTrue(original_df.equals(new_df), "annotations not retained")
    
    # tests that all data from the annotation file is retained
    def test_annotations_retained(self):
        original_df = pd.read_feather(input_path + scene_name + '/annotations.feather')
        new_df = pd.read_feather(output_path + scene_name + '/annotations.feather')

        self.assertTrue(original_df.equals(new_df), "annotations not retained")

    # tests that all data from the egovehicle file is retained
    def test_egovehicle_retained(self):
        original_df = pd.read_feather(input_path + scene_name + '/city_SE3_egovehicle.feather')
        new_df = pd.read_feather(output_path + scene_name + '/city_SE3_egovehicle.feather')
        
        self.assertTrue(original_df.equals(new_df), "annotations not retained")

if __name__ == '__main__':
    # TODO: change input_path and output_path to command line inputs

    # copies the original file directory; TODO: take out once above is implemented
    if not os.path.isdir(output_path):
        shutil.copytree(input_path, output_path)

    unittest.main()