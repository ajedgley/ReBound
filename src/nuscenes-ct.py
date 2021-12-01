"""
nuscenes-ct.py

Conversion tool for bringing data from the waymo dataset into our generic data format
"""
import getopt
import sys
import os
import utils
import json
from nuscenes.nuscenes import NuScenes

from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.data_classes import Quaternion
from nuscenes.utils.data_classes import Box

import numpy as np

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
    input_path = ""
    output_path = ""
    # We can make scene_names a list so that it either includes the name of one scene or every scene a user wants to batch process
    scene_names = []
    pred_path = ""
    # Read in flags passed in with command line argument
    # Make sure that options which need an argument (namely -f for input file path and -o for output file path) have them
    # User is able to specify -h, -f, -o, -s, and -r options
    # -h brings up help menu
    # -f is used to specify the path to the Waymo file you want to read in and requires one arg. If -r is specified then this arg
    # corresponds to a directory containing all the .tfrecord files you'd like to read in
    # -o is used to specify the path to the directory where the LVT format will go.
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "hf:o:s:p:r", "help")
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("REQUIRED: -f to specify directory of nuScenes dataset")
            print("REQUIRED: -o to specify the path where the LVT dataset will go")
            print("OPTIONAL: -p to specify a path to projected data")
            sys.exit(2)
        elif opt == "-f":
            input_path = arg
            input_string = input("Please enter a comma-delinated list of scene names. Remove any whitespace. To convert all scenes, leave blank\n")
            if len(input_string) != 0:
                list_of_scenes = input_string.split(",")
                scene_names.extend(list_of_scenes)
            # Debug print statement
            print(scene_names)
        elif opt == "-o":
            output_path = arg
        elif opt == "-p":
            pred_path = arg

        else:
            print("Invalid set of arguments entered. Please refer to -h flag for more information.")
            sys.exit(2)

    return (input_path, output_path, scene_names, pred_path)

# Used to check if file is valid nuScenes file
def validate_input_path(input_path):
    """Verify that input path given to nuscenes database is valid input
    Args:
        input_path: Path to check before reading into LVT generic format
    Returns:
        True on valid input and False on invalid input
        """

    # Check that the input path exists and is a valid nuScenes database
    try:
        # If input path is invalid as nuScenes database, this constructor will throw an AssertationError
        NuScenes(version='v1.0-mini', dataroot=input_path, verbose=True)
        return True
    except AssertionError as error:
        print("Invalid argument passed in as nuScenes file.")
        return False

def extract_ego(nusc, sample, frame_num, output_path):
    """Extracts ego data from one frame and puts it in the lct file system
    Args:
        nusc: NuScenes API object used for obtaining data
        sample: Frame of nuScenes data
        frame_num: Number corresponding to sample
        output_path: Path to generic data format directory
    Returns:
        None
        """

    # Get ego pose information. The LIDAR sensor has the ego information, so we can use that.
    sensor = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
    poserecord = nusc.get('ego_pose', sensor['ego_pose_token'])

    full_path = os.path.join(os.getcwd(), output_path)
    utils.create_ego_directory(full_path, frame_num, poserecord['translation'], poserecord['rotation'])

def extract_bounding(nusc, sample, frame_num, output_path):
    """Extracts the bounding data from a nuScenes frame and converts it into our intermediate format
    Args:
        nusc: NuScenes API object used for obtaining data
        sample: Frame of nuScenes data
        frame_num: Number corresponding to sample
        output_path: Path to generic data format directory
    Returns:
        None
        """
    origins = []
    sizes = []
    rotations = []
    annotation_names = []
    confidences = []
    
    # Get translation, rotation, dimensions, and origins for bounding boxes for each annotation
    for i in range(0, len(sample['anns']) - 1):
        token = sample['anns'][i]
        annotation_metadata = nusc.get('sample_annotation', token)
        # Create nuscenes box object so we can easily transform this box to the vehicle frame that our dataset requires
        box = Box(annotation_metadata['translation'], annotation_metadata['size'], Quaternion(annotation_metadata['rotation']))

        # Get ego pose information. The LIDAR sensor has the ego information, so we can use that.
        sensor = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        poserecord = nusc.get('ego_pose', sensor['ego_pose_token'])
        
        box.translate(-np.array(poserecord['translation']))
        box.rotate(Quaternion(poserecord['rotation']).inverse)

        # Store data obtained from annotation
        origins.append(box.center.tolist())
        sizes.append(annotation_metadata['size'])
        rotations.append(box.orientation.q.tolist())
        annotation_names.append(annotation_metadata['category_name'])

        # Confidence for ground truth data is always 100
        confidences.append(100)
        
    utils.create_frame_bounding_directory(output_path, frame_num, origins, sizes, rotations, annotation_names, confidences)

def extract_pred_bounding(pred_path, nusc, scene_token, sample, output_path, pred_data):
    """Similar to extract_bounding, but specifically to read in predicted data given by a user
    Args:
        pred_path: Path to predicated data provided by user
        nusc: NuScenes API object used for obtaining data
        sample: Frame of nuScenes data
        output_path: Path to generic data format directory
    Returns:
        None
        """
    origins = []
    sizes = []
    rotations = []
    annotation_names = []
    confidences = []
    pred_sample_tokens = []
    frame_num = 0
    # Create list of sample_tokens that correspond to the scene we are converting
    for sample_token in pred_data['results']:
        try:
            sample = nusc.get('sample', sample_token)
            if sample['scene_token'] == scene_token:
                pred_sample_tokens.append(sample_token)
        except:
            continue 
    
    


    # Now go through each sample token that corresponds to our scene and import the data taken from pred_data
    for sample_token in pred_sample_tokens:
        origins = []
        sizes = []
        rotations = []
        annotation_names = []
        confidences = []

        # Ego Frame data for conversion
        sample = nusc.get('sample', sample_token)
        sensor = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        poserecord = nusc.get('ego_pose', sensor['ego_pose_token'])
        for data in pred_data['results'][str(sample_token)]:
            box = Box(data['translation'], data['size'], Quaternion(data['rotation']))
            box.translate(-np.array(poserecord['translation']))
            box.rotate(Quaternion(poserecord['rotation']).inverse)
            origins.append(box.center.tolist())
            sizes.append(data['size'])
            rotations.append(box.orientation.q.tolist())
            annotation_names.append(data['detection_name'])
            confidences.append(int(data['detection_score'] * 100))
        utils.create_frame_bounding_directory(output_path, frame_num, origins, sizes, rotations, annotation_names, confidences, True)
        frame_num += 1

def extract_rgb(nusc, sample, frame_num, target_path):
    """Extracts the RGB data from a nuScenes frame and converts it into our intermediate format
    Args:
        nusc: NuScenes API object used for obtaining data
        sample: NuScenes frame
        frame_num: frame number
        output_path: Path to generic data format directory
    Returns:
        None
        """
    camera_list = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
    # For each camera sensor
    for camera in camera_list:
        (path, _, _) = nusc.get_sample_data(sample['data'][camera])
        utils.add_rgb_frame_from_jpg(target_path, camera, frame_num, path)

def extract_lidar(nusc, sample, frame_num, target_path):
    """Used to extract the LIDAR pointcloud information from the nuScenes dataset
    Args:
        nusc: NuScenes api object for getting info related to LiDAR data
        sample: All the sensor information
        frame_num: Frame number
        target_path: Output directory path where data will be written to
    Returns:
        None
        """
    
    # We'll need to get all the information we need to pass to utils.add_lidar_frame()
    # Get the points, translation, and rotation info using our nusc input
    sensor = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
    cs_record = nusc.get('calibrated_sensor', sensor['calibrated_sensor_token'])
    (path, _, _) = nusc.get_sample_data(sample['data']["LIDAR_TOP"])
    points = LidarPointCloud.from_file(path)
    translation = cs_record['translation']
    rotation = cs_record['rotation']

    # Transform points to Vehicle Frame
    points.rotate(Quaternion(rotation).rotation_matrix)
    points.translate(translation)
    
    # Reshape points
    points = np.transpose(points.points[:3, :])

    utils.add_lidar_frame(target_path, "LIDAR_TOP", frame_num, points, translation, rotation)

def count_frames(nusc, sample):
    """Counts frames to use for progress bar
    Args:
        nusc: NuScenes api object
        sample: nuScenes frame, this should be the first frame
    Returns:
        frame_count: number of frames
        """
    frame_count = 0

    # This prevents our function from modifying the sample
    if sample['next'] != '':
        frame_count += 1

        # Don't want to change where sample['next'] points to since it's used later, so we'll create our own pointer
        sample_counter = nusc.get('sample', sample['next'])

        while sample_counter['next'] != '':
            frame_count += 1
            sample_counter = nusc.get('sample', sample_counter['next'])

    return frame_count
   
def convert_dataset(output_path, scene_name, pred_data):
    # Validate the scene name passed in
    try:
        scene_token = nusc.field2token('scene', 'name', scene_name)[0]
    except Exception:
        print("\n Not a valid scene name for this dataset!")
        exit(2)

    scene = nusc.get('scene', scene_token)
    sample = nusc.get('sample', scene['first_sample_token'])
    frame_num = 0
    
    # Set up camera directories
    camera_list = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]
    for camera in camera_list:
        sensor = nusc.get('sample_data', sample['data'][camera])
        cs_record = nusc.get('calibrated_sensor', sensor['calibrated_sensor_token'])
        utils.create_rgb_sensor_directory(output_path, camera, cs_record['translation'], cs_record['rotation'], cs_record['camera_intrinsic'])

    # Set up LiDAR directory
    utils.create_lidar_sensor_directory(output_path, "LIDAR_TOP")

    if pred_data != {}:
        print('Extracting predicted bounding boxes...')
        extract_pred_bounding(pred_path, nusc, scene_token, sample, output_path, pred_data)
    
    # Setup progress bar
    frame_count = count_frames(nusc, sample)
    utils.print_progress_bar(0, frame_count)

    # Extract sample data from scene
    while sample['next'] != '':
        # Extract all the relevant data from the nuScenes dataset for our scene. The variable 'sample' is the frame
        # Note: This is NOT multithreaded for nuScenes data because each scene is small enough that this runs relatively quickly.
        extract_ego(nusc, sample, frame_num, output_path)
        extract_bounding(nusc, sample, frame_num, output_path)
        extract_rgb(nusc, sample, frame_num, output_path)
        extract_lidar(nusc, sample, frame_num, output_path)
        frame_num += 1
        sample = nusc.get('sample', sample['next'])
        utils.print_progress_bar(frame_num, frame_count)

# Driver for nuscenes conversion tool
if __name__ == "__main__":

    # Read in input database and output directory paths
    (input_path, output_path, scene_names, pred_path) = parse_options()
    
    # Validate whether the database path passed in is valid and if the output directory path is valid
    # If the output directory exists, then use that directory. Otherwise, create a new directory at the
    # specified path. 

    if not validate_input_path(input_path):
        print("Invalid input path specified. Please check paths entered and try again")
        sys.exit(2)
    pred_data = {}
    if pred_path != "":
        pred_data = json.load(open(pred_path))

    nusc = NuScenes('v1.0-mini', input_path, True)

    path = os.getcwd()

    #If this was blank, then convert all scenes
    if len(scene_names) == 0:
        print("Converting All Scenes...")
        for scene in nusc.scene:
            scene_names.append(scene['name'])
        print(scene_names)

    # Output directory path is validated in utils.create_lct_directory()
    # utils.create_lct_directory(os.getcwd(), output_path)
    
    # We have to make an output folder for each item we're converting
    # Users can then point to the output folder they want to use when running lct.py
    # Setting our output_path to be the parent directory for all these output folders
    for scene_name in scene_names:
        utils.create_lct_directory(output_path, scene_name)
                
    nusc.list_scenes()
    

    for scene_name in scene_names:
        convert_dataset(output_path + "/" + scene_name, scene_name, pred_data)
