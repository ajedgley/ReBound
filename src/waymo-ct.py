#Conversion Script for waymo Dataset
import sys
import getopt
import os
from utils import create_frame_bounding_directory, create_lct_directory
from waymo_open_dataset.utils import frame_utils
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset

#Get command line options
def parse_options():
    waymo_path = ""
    folder_name = ""
    custom_path = ""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:o:p:", "help")
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("required: -f to specify the path to the Waymo file")
            print("required: -o to specify the name of the directory where the LVT format will go. Will be a folder in the current directory")
            print("optional: -p to specify a custom path where the LVT format dataset will go")
            sys.exit(2)
        elif opt == "-f":
            waymo_path = arg
        elif opt == "-o":
            folder_name = arg
        elif opt == '-p':
            custom_path = arg
        else:
            sys.exit(2)

    return (waymo_path, folder_name, custom_path)

def extract_bounding(frame, frame_num, lct_path):
    origins = []
    sizes = []
    rotations = []
    annotation_names = []
    annotation_dict = {1: "Vehicle", 2: "Pedestrian", 3: "Sign", 4:"Cyclist"}
    confidences = []
    for label in frame.laser_labels:
        origins.append([label.box.center_x, label.box.center_y, label.box.center_z])
        sizes.append([label.box.width, label.box.length, label.box.height])
        annotation_names.append(annotation_dict[label.type])
        rotations.append([0,0,0,0])
        confidences.append(100)
    create_frame_bounding_directory(folder_name, frame_num, origins, sizes,rotations,annotation_names,confidences)
if __name__ == "__main__":
    (waymo_path, folder_name, custom_path) = parse_options()

    #Check if path specified is a Waymo dataset file
    if os.path.splitext(waymo_path)[1] != ".tfrecord":
        print("The file specified is not a tfrecord")
        sys.exit(2)

    if len(custom_path) != 0:
        create_lct_directory(os.getcwd().join(custom_path), folder_name)
    else:
        create_lct_directory(os.getcwd(), folder_name)

    #Extract data from TFRecord File
    dataset = tf.data.TFRecordDataset(waymo_path,'')

    #Loop through each frame
    counter = 0
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        #At this point have one frame imported as 'frame'
        extract_bounding(frame,counter,folder_name)
        counter += 1
        

    