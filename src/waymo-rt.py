"""
waymo-ct.py

Conversion tool for bringing data from the waymo dataset into our generic data format
"""
import sys
import getopt
import os
from utils import geometry_utils
from utils import dataformat_utils
from waymo_open_dataset.utils import frame_utils
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
from waymo_open_dataset import label_pb2 as open_label
import numpy as np
import PIL
import io
import json
from pyquaternion import Quaternion
import concurrent.futures
import uuid
import open3d as o3d
import google.protobuf
from google.protobuf.json_format import MessageToDict

# This should be fine for now
def parse_options():
    ''' Read in user command line input to get directory paths which will be used for input and output.
    Args:
        None
    Returns:
        input_path: Path to LVT dataset being exported into argoverse
        output_path: Path to existing argoverse dataset
    '''
    input_path = ""
    output_path = ""
    original_path = ""
    scene_names = []

    try:
        opts, _ = getopt.getopt(sys.argv[1:], "hf:o:v:", "help")
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("REQUIRED: -f to specify the path to the LVT file.")
            print("REQUIRED: -o to specify the path to the argoverse file.")
            print("REQUIRED: -v to specify the path to the original file.")
            sys.exit(2)
        elif opt == "-f":
            input_path = arg
            input_string = input("Please enter a comma-delinated list of scene names. Remove any whitespace. To convert all scenes, leave blank.\n")
            if len(input_string) != 0:
                list_of_scenes = input_string.split(",")
                scene_names.extend(list_of_scenes)
        elif opt == "-o":
            output_path = arg
        elif opt == "-v":
            original_path = arg
        else:
            sys.exit(2)

    return (original_path, input_path, output_path, scene_names)


def clear_labels(labels):
        a = 0
        for label in labels:
            a+=1
        for i in range(0, a):
            labels.pop()
        return labels

def extract_bounding(frame, frame_num, input_path):
    annotations = json.load(open(input_path + "/bounding/" + str(frame_num) + "/boxes.json"))


    new_laser_labels = clear_labels(frame.laser_labels)

    for box in annotations["boxes"]:

        #construct new laser labels (this doesn't match up with for label in frame.label_laser HELP)
        new_label = open_label.Label()

        # constructs a new box, heading may be wrong
        new_box = open_label.Label.Box()
        new_box.center_x = box["origin"][0]
        new_box.center_y = box["origin"][1]
        new_box.center_z = box["origin"][2]
        new_box.width = box["size"][0]
        new_box.length = box["size"][1]
        new_box.height = box["size"][2]
        quat = Quaternion(box["rotation"][0], box["rotation"][1], box["rotation"][2], box["rotation"][3])
        new_box.heading = quat.radians

        new_metadata = open_label.Label.Metadata()
        new_metadata.speed_x = 0
        new_metadata.speed_y = 0
        new_metadata.accel_x = 0
        new_metadata.accel_y = 0

        new_label.box.CopyFrom(new_box)
        new_label.metadata.CopyFrom(new_metadata)
        
        if box["annotation"] == "Vehicle":
            new_type = 1
        elif box["annotation"] == "Pedestrian":
            new_type = 2
        elif box["annotation"] == "Sign":
            new_type = 3
        elif box["annotation"] == "Cyclist":
            new_type = 4
        else:
            new_type = 0
        
        new_label.type = new_type
       
        new_label.id = "0000000000000000000000"
        new_label.num_lidar_points_in_box = 0

        fp1 = open("label.txt", "w")
        fp1.write(repr(new_label))
        new_laser_labels.append(new_label)

    return frame
        
        





def count_frames(dataset):
    """counts frames in dataset to use for progress bar
    Args:
        dataset: waymo dataset
    Returns:
        frame_count: number of frames
        """
    frame_count = 0
    # dataset is a tfrecord, so no "length" exists.
    for frame in dataset:
        frame_count += 1
    return frame_count

if __name__ == "__main__":
    (original_path, input_path, output_path, scene_names) = parse_options()

    dataset = tf.data.TFRecordDataset(original_path)

    writer = tf.io.TFRecordWriter(output_path)
    for frame_num, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        
        #returns a new frame with edited annotations

        newframe = extract_bounding(frame, frame_num, input_path)

        #print("uhhh", newframe)
        # do I need to set this?s
        output = newframe.SerializeToString()
        #AHAHAHAH THIS IS THE IMPORATNT PART
        writer.write(output)

        #dataset = tf.data.TFRecordDataset("/home/royce/Documents/CMSC435/LVT/waymo/idk.tfrecord")
        #print(repr(dataset))
        # remove this later, easier for testing (one frame)


