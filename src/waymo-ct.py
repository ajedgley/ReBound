"""
waymo-ct.py

Conversion tool for bringing data from the waymo dataset into our generic data format
"""
import sys
import getopt
import os
import utils
from waymo_open_dataset.utils import frame_utils
from waymo_open_dataset.utils import transform_utils
import tensorflow as tf
from waymo_open_dataset import dataset_pb2 as open_dataset
import numpy as np
import PIL
import io
from pyquaternion import Quaternion
import concurrent.futures

# Name of all the RGB cameras
RGB_Name  = {
    0:"UNKNOWN",
    1:'FRONT',
    2:'FRONT_LEFT',
    3:'FRONT_RIGHT',
    4:'SIDE_LEFT',
    5:'SIDE_RIGHT'
  }

# Name of all LiDAR sensors
Lidar_Name = {
    0:"UNKNOWN",
    1:"TOP",
    2:"FRONT",
    3:"SIDE_LEFT",
    4:"SIDE_RIGHT",
    5:"REAR"
}


def parse_options():
    """Read in user command line input to get directory paths which will be used for input and output.
    Args:
        None
    Returns:
        input_path: Path to waymo dataset being read into LVT
        output_path: Path where user wants LVT to generate generic data format used in program
    """
    input_path = ""
    output_path = ""
    batch_processing = False

    # User is able to specify -h, -f, -o, and -r options
    # -h brings up help menu
    # -f is used to specify the path to the Waymo file you want to read in and requires one arg. If -r is specified then this arg
    # corresponds to a directory containing all the .tfrecord files you'd like to read in
    # -o is used to specify the path to the directory where the LVT format will go. If -r is specified then this folder will contain output
    # folders for each .tfrecord file read in
    # -r is used to specify the user is trying to batch process a set of files corresponding to the directory given with the -f flag
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:o:r", "help")
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("required: -f to specify the path to the Waymo file")
            print("required: -o to specify the name of the directory where the LVT format will go. Will be a folder in the current directory")
            sys.exit(2)
        elif opt == "-f":
            input_path = arg
        elif opt == "-o":
            output_path = arg
        elif opt == "-r":
            # Indicates that the user is trying to run batch processing
            batch_processing = True
        else:
            sys.exit(2)

    return (input_path, output_path, batch_processing)

def extract_bounding(frame, frame_num, lct_path):
    """Extracts the bounding data from a waymo frame and converts it into our intermediate format
    Args:
        frame: waymo frame
        frame_num: frame number
        lct_path: path to LCT directory
    Returns:
        None
        """

    origins = []
    sizes = []
    rotations = []
    annotation_names = []
    annotation_dict = {1: "Vehicle", 2: "Pedestrian", 3: "Sign", 4:"Cyclist"}
    confidences = []

    # Get annotation, rotation, confidence level, quaternion, center, and diminensions of each bounding box in frame
    for label in frame.laser_labels:
        origins.append([label.box.center_x, label.box.center_y, label.box.center_z])
        sizes.append([label.box.width, label.box.length, label.box.height])
        annotation_names.append(annotation_dict[label.type])
        quat = Quaternion(axis=[0.0, 0.0, 1.0], radians=label.box.heading)
        rotations.append(quat.q.tolist())
        # Confidence set to 100 by default for ground truth data
        confidences.append(100)
    utils.create_frame_bounding_directory(lct_path, frame_num, origins, sizes, rotations, annotation_names, confidences)

def setup_rgb(frame, lct_path):
    """Sets up the RGB directory with extrinsic data
    Args:
        frame: waymo frame
        lct_path: path to LCT directory
    Returns:
        None
        """

    camera_data_int = {}
    camera_data_ext = {}

    # Get the camera names and their intrinsic data
    for c in frame.context.camera_calibrations:

        # Store intrinsic and extrinsic data for each camera in a dictionary
        matrix = np.array(c.intrinsic, np.float32).tolist()
        camera_data_int[RGB_Name[c.name]] = [[matrix[0],0,matrix[2]],[0, matrix[1], matrix[3]],[0,0,1]]
        camera_data_ext[RGB_Name[c.name]] = np.reshape(np.array(c.extrinsic.transform, np.float32), [4, 4])
    
    # Create directory for each camera in scene
    for image in frame.images:
        # Convert given extrinsic data into format we can use 
        axes_transformation = np.array([
                [0,-1,0,0],
                [0,0,-1,0],
                [1,0,0,0],
                [0,0,0,1]])
        axes_transformation = np.linalg.inv(axes_transformation)
        transform_matrix = np.matmul(camera_data_ext[RGB_Name[image.name]], axes_transformation)
        translation, rotation_quats = utils.translation_and_rotation(transform_matrix.tolist())
        # Create directory for camera
        utils.create_rgb_sensor_directory(lct_path, RGB_Name[image.name], translation, rotation_quats, camera_data_int[RGB_Name[image.name]])

def extract_rgb(frame, frame_num, lct_path):
    """Extracts the RGB data from a waymo frame and converts it into our intermediate format
    Args:
        frame: waymo frame
        frame_num: frame number
        lct_path: path to LCT directory
    Returns:
        None
        """

    # Add image files to respective camera directory
    for image in frame.images:
        utils.add_rgb_frame(lct_path, RGB_Name[image.name], PIL.Image.open(io.BytesIO(image.image)), frame_num)

def setup_lidar(frame, lct_path, translations, rotations):
    """Uses the first frame to initialize LiDAR directory and store extrinsic data
    Args:
        frame: waymo frame
        lct_path: path to LCT directory
        translations: empty translation dictionary
        rotations: empty rotation dictionary
    Returns:
        None
        """

    # Calibrations are the data which go along with each point cloud for a LiDAR sensor
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    for c in calibrations:
        sensor = c.name

        # Set up the folder for each sensor
        utils.create_lidar_sensor_directory(lct_path, Lidar_Name[sensor])
        transform_matrix = c.extrinsic.transform

        # The transaltion matrices are the same for each frame, so this computation is only run once
        translation, rotation = utils.translation_and_rotation(transform_matrix)
        translations[sensor] = translation
        rotations[sensor] = rotation

def extract_lidar(frame, frame_num, lct_path, translations, rotations):
    """Extracts LiDAR data from one frame and puts it in the lct file system
    Args:
        frame: waymo frame
        frame_num: frame number
        lct_path: path to LCT directory
        translations: translation dictionary
        rotations: rotation dictionary
    Returns:
        None
        """

    # Extract the pointclouds as a list of points
    range_images, camera_projections,range_image_top_pose = frame_utils.parse_range_image_and_camera_projection(frame)
    point_clouds, _ = frame_utils.convert_range_image_to_point_cloud(frame,range_images,camera_projections,range_image_top_pose,0,False)

    # There are 5 pointclouds corresponding to the 5 sensors
    for i, points in enumerate(point_clouds):
        # Sensor numbers are indexed from 1 in Waymo
        sensor = i+1
        utils.add_lidar_frame(lct_path, Lidar_Name[sensor], frame_num, points, translations[sensor], rotations[sensor])

def extract_ego(frame, frame_num, lct_path):
    """Extracts ego data from one frame and puts it in the lct file system
    Args:
        frame: waymo frame
        frame_num: frame number
        lct_path: path to LCT directory
    Returns:
        None
        """
    translation, rotation_quats = utils.translation_and_rotation(frame.pose.transform)
    utils.create_ego_directory(lct_path, frame_num, translation, rotation_quats)

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

def convert_dataset(output_path, dataset):
    # Initialize LiDAR camera dictionaries
    translations = {}
    rotations = {}

    frame_count = count_frames(dataset)
    executor = concurrent.futures.ThreadPoolExecutor(os.cpu_count() + 1)
    futures = []

    # start progress bar
    utils.print_progress_bar(0, frame_count)
    # Loop through each frame
    for frame_num, data in enumerate(dataset):
        frame = open_dataset.Frame()
        frame.ParseFromString(bytearray(data.numpy()))

        if frame_num == 0:
            setup_rgb(frame, output_path)
            setup_lidar(frame, output_path, translations, rotations)

        # Each function call is submitted to a thread pool so that they can be run concurrently
        # futures allows us to track when multithreaded functions terminate
        # executor.submit starts a multithreaded proecss corresponding to the functions passed in as the first arg of the function call
        futures.append([executor.submit(extract_bounding, frame, frame_num, output_path),
        executor.submit(extract_rgb, frame, frame_num, output_path),
        executor.submit(extract_lidar, frame, frame_num, output_path, translations, rotations),
        executor.submit(extract_ego, frame, frame_num, output_path)])

    # When each frame is done processing, update progress bar
    frame_num = 0
    for frame in futures:
        concurrent.futures.wait(frame, return_when=concurrent.futures.ALL_COMPLETED)
        frame_num += 1
        utils.print_progress_bar(frame_num, frame_count)

if __name__ == "__main__":
    (input_path, output_path, batch_processing) = parse_options()

    # This list will remain empty if we're not batch processing, but if we're batch processing then it will list all the items being
    # proccessed
    batch_items = []

    # Check if path specified is a Waymo dataset file if not batch processing
    if not batch_processing and os.path.splitext(input_path)[1] != ".tfrecord":
        print("The file specified is not a tfrecord")
        sys.exit(2)
    else:
        # Check if all files in directory specified correspond to Waymo dataset file.
        dir_contents = os.listdir(input_path)
        for item in dir_contents:
            item_name_details = os.path.splitext(item)
            # Fail if any item is not a .tfrecord
            if item_name_details[1] != ".tfrecord":
                print("The directory specified for batch processing has a file which is not a tfrecord")
                print("The file name is", item)
                sys.exit(2)
            batch_items.append(item_name_details[0])

    # Frequently using current work directory; storing a reference
    path = os.getcwd()
    
    # If we're running batch processing, don't want our root folder to have subfolders for the different types of data. We only want subfolders for each scene
    if batch_processing:
        try:
            parent_path = os.path.join(path, output_path)
            os.makedirs(parent_path, exist_ok=True)
        except OSError as error:
            print(error)
            sys.exit(1)
    else:
        utils.create_lct_directory(path, output_path)

    # If we're batch processing, we have to make an output folder for each item we're converting
    # Users can then point to the output folder they want to use when running lct.py
    # Setting our output_path to be the parent directory for all these output folders
    if batch_processing:
        output_path = path + "/" + output_path
        for item_name in batch_items:
            utils.create_lct_directory(output_path, item_name)

    # Extract data from TFRecord File
    datasets = []
    if not batch_processing:
        datasets.append(tf.data.TFRecordDataset(input_path, ''))
    else:
        # Convert each tfrecord file at the input directory location if we are batch processing
        dir_contents = os.listdir(input_path)
        for item in dir_contents:
            datasets.append(tf.data.TFRecordDataset(input_path + "/" + item, ''))

    # Convert data into LVT generic format
    if batch_processing:
        for dataset, item_name in zip(datasets, batch_items):
            convert_dataset(output_path + "/" + item_name, dataset)
    else:
        convert_dataset(output_path, datasets[0])