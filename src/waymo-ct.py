#Conversion Script for waymo Dataset
import sys
import getopt
import os
from utils import create_frame_bounding_directory, create_lct_directory
from waymo_open_dataset.utils import frame_utils
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
import numpy as np
import PIL
import io


# Name of all the cameras
Name  = {
    0:"UNKNOWN",
    1:'FRONT',
    2:'FRONT_LEFT',
    3:'FRONT_RIGHT',
    4:'SIDE_LEFT',
    5:'SIDE_RIGHT'
  }

def extract_rgb(output_path, waymo_path):

    """Extracts the RGB data from a waymo tfrecord and converts it into our intermediate format
    Args:
        output_path: path to LCT directory
        waymo_path: path to waymo data
    Returns:
        None
        """

    #Extract data from TFRecord File
    dataset = tf.data.TFRecordDataset(waymo_path,'')

    frame_num = 0
    #Loop through each frame
    for data in dataset:
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        #At this point have one frame imported as 'frame'

        #For the data that's the same in each frame:
        if(frame_num == 0):
            camera_data = {}
            # We get the camera names and their intrinsic data
            frame_dict = frame_utils.convert_frame_to_dict(frame)
            for intrinsic_data in frame_dict.keys():

                # If we've gotten this far, that means intrinsic_data holds the intrinsic data (and name) of a camera
                if(intrinsic_data[-10:] == "_INTRINSIC"):
                    camera_data[intrinsic_data[:-10]] = frame_dict[intrinsic_data].reshape((3, 3)).tolist()

            #frame.pose.transform holds a 4x4 rotation/translation matrix. Here we extract the translation vector:
            translation = (frame.pose.transform[3],frame.pose.transform[7],frame.pose.transform[11])

            #...and the 3x3 rotation matrix:
            rot_matrix = [[frame.pose.transform[i*4 + j] for j in range(3)] for i in range(3)]
            
            matrix_numpy = np.array(rot_matrix)

            #Next, convert the 3x3 matrix into quaternions:
            (eigenvalues, eigenvectors) = np.linalg.eig(matrix_numpy)
            
            for i in range(3):
                if(eigenvalues[i] == 1):
                    eigenu = eigenvectors[:,i]
            
            trace_matrix = matrix_numpy.diagonal().sum()
            cos_theta = (trace_matrix - 1) / 2

            sine_half_theta = ((1 - cos_theta)/ 2)**0.5

            rotation_quats = (((1 + cos_theta)/ 2)**0.5, float(sine_half_theta * eigenu[0]),
            float(sine_half_theta * eigenu[1]), float(sine_half_theta * eigenu[2]))

        # finally, with all that settled, let's create the directory and files:
        for image in frame.images:
            if (frame_num == 0):
                utils.create_rgb_sensor_directory(output_path, Name[image.name], translation, rotation_quats,
                camera_data[Name[image.name]])
            utils.add_rgb_frame(output_path, Name[image.name], PIL.Image.open(io.BytesIO(image.image)), frame_num)
        frame_num += 1

        
    

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

    path = os.getcwd()
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
        

    