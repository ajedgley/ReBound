#Licensing


#Utils for creating LCT Directory
import os
import sys
import json
import PIL
import numpy as np
from shutil import copyfile
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation as R


def translation_and_rotation(transform_matrix):
    """Converts tranformation matrix to a translation and rotation our conversion scripts can use
    Args:
        transform_matrix: 1x16 list representing a 4x4 transform matrix
    
    Returns:
        translation: 1x3 translation vector
        rotation: 1x4 rotation quaternion
    """

    transform_array = np.array(transform_matrix).reshape(4, 4)

    #Extract the first 3 entries in the last column as the translation
    translation = tuple(transform_array[:3, -1])

    #Extract the 3x3 rotation matrix from the upper left
    rotation_matrix = transform_array[:3, :3]

    #Convert rotation matrix to a quaternion
    quat = R.from_matrix(rotation_matrix)
    rotation = (quat.as_quat()[3], quat.as_quat()[0], quat.as_quat()[1], quat.as_quat()[2])
    
    return translation, rotation

def print_progress_bar(frame_num, total):
    """Prints a progress bar
    Args:
        frame_num: current frame number
        total: number of frames
    Returns:
        None
        """

    length = 40
    if os.get_terminal_size()[0] < 50:
        length = 10
    elif os.get_terminal_size()[0] < 70:
        length = 20
    filled_length = (length * frame_num//total)
    bar = '█' * filled_length + '-' * (length - filled_length)
    print(f'\rConverting: {bar} {frame_num}/{total} frames', end = '\r')
    # Print New Line on Complete
    if frame_num == total: 
        print()


#Creates top level lct directory structure at "path"
def create_lct_directory(path, name):
    """Create LCT directory at specified path
    Args:
        path: target path where directory will be stored
    
    Returns:
        None
    """
    sub_directories = ['cameras', 'pointcloud', 'bounding', 'pred_bounding', 'ego']
    try:
        parent_path = os.path.join(path, name)
        os.makedirs(parent_path, exist_ok=True)
        print("added new folder")
        for directory in sub_directories:
            full_path = os.path.join(parent_path, directory)
            os.makedirs(full_path, exist_ok=True)
    except OSError as error:
        print(error)
        sys.exit(1)

def create_rgb_sensor_directory(path, name, translation, rotation, intrinsic):
    """Adds one RGB sensor directory inside camera directory, and adds extrinsic/intrinsic data for said sensor
    Args:
        path: path to LCT directory
        name: name of RGB sensor
        translation: (x,y,z) tuple representing sensor translation
        rotation: (w,x,y,z) quaternion representing sensor rotation
        intrinsic: [3,3] 3x3 2D List representing intrinsic matrix
    Returns:
        None
        """

    #Create necessary directory: ./Cameras/[name]
    work_dir = os.path.join(path, "cameras", name)

    os.makedirs(work_dir, exist_ok=True)

    #Creates the Extrinsic.json file in the /Cameras/[name] directory from the
    #translation and rotation parameters.
    extrinsics = {}
    extrinsics['translation'] = translation
    extrinsics['rotation'] = rotation

    with open(work_dir + "/extrinsics.json", "w") as extrinsic_file:
        extrinsic_file.write(json.dumps(extrinsics))

    #Creates the Extrinsic.json file in the /Cameras/[name] directory from the intrinsic parameter.
    with open(work_dir + "/intrinsics.json", "w") as intrinsic_file:
        intrinsic_file.write(json.dumps({"matrix" : intrinsic}))


def add_rgb_frame(path, name, image, frame_num):
    """Adds one jpg from one frame to the structure inside the camera directory for a given sensor
    Args:
        path: path to LCT directory
        name: name of RGB sensor
        images: list of buffers containing JPG images (assumed that length of list is also number of frames)
        frame_num: the number corresponding to the frame
    Returns:
        None
        """

    #Assumes directory exists
    image.save(os.path.join(os.path.join(os.path.join(path, "cameras"), name),
    str(frame_num) +".jpg"))

def add_rgb_frame_from_jpg(path, name, frame_num, input_path):
    """Copies existing jpg file into the cameras directory
    Args:
        path: path to LCT directory
        name: name of RGB sensor
        frame_num: the number corresponding to the frame
        input_path: Path to source jpg iamge
    Returns:
        None
        """
    full_path = os.path.join(path, 'cameras', name, str(frame_num) + '.jpg')
    copyfile(input_path, full_path)

def create_lidar_sensor_directory(path, name):
    """Creates directory for one LiDAR sensor
    Args:
        path: path to LCT directory
        name: name of RGB sensor
    Returns:
        None
        """

    full_path = os.path.join(path, 'pointcloud', name)
    os.makedirs(full_path, exist_ok=True)

def add_lidar_frame(path, name, frame_num, points, translation, rotation):
    """Adds one lidar sensor directory inside pointcloud directory
    Args:
        path: path to LCT directory
        name: name of lidar sensor
        frame_num: frame number
        points: [n, 3] list of (x,y,z) tuples representing x,y,z coordinates
        translation: (x,y,z) tuple representing sensor translation
        rotation: (w,x,y,z) quaternion representing sensor rotation
    Returns:
        None
        """

    # see .pcd file format documentation at https://pointclouds.org/documentation/tutorials/pcd_file_format.html
    pcd_lines = ['# .PCD v0.7 - Point Cloud Data file format', 'VERSION 0.7', 'FIELDS x y z',
                'SIZE 4 4 4', 'TYPE F F F', 'COUNT 1 1 1']
    pcd_lines.append('WIDTH ' + str(len(points)))
    pcd_lines.append('HEIGHT 1')
    pcd_lines.append('VIEWPOINT ' + ' '.join([str(i) for i in translation + rotation]))
    pcd_lines.append('POINTS ' + str(len(points)))
    pcd_lines.append('DATA ascii')
    for point in points:
        pcd_lines.append(' '.join([str(i) for i in point]))
    
    pcd_str = '\n'.join(pcd_lines)

    full_path = os.path.join(path, 'pointcloud', name, str(frame_num) + '.pcd')
    
    with open(full_path, 'w') as f:
        f.write(pcd_str)

def add_lidar_frame_from_pcd(path, name, frame_num, input_path):
    """Copies one .pcd file to pointcloud directory
    Args:
        path: path to LCT directory
        name: name of lidar sensor
        frame_num: frame number
        input_path: path to .pcd file
    Returns:
        None
        """
    
    full_path = os.path.join(path, 'pointcloud', name, str(frame_num) + '.pcd')
    copyfile(input_path, full_path)

def create_frame_bounding_directory(path, frame_num, origins, sizes, rotations, annotation_names, confidences):
    """Adds box data for one frame
    Args:
        path: path to LCT directory
        frame_num: frame index (0-indexed) (will overwrite if duplicate frames are specified)
        origins: [n, 3] list representing x,y,z coordinates of the center of the boxes
        sizes: [n, 3] list representing W,L,H of box
        rotations:[n, 4] list of quaternions representing box rotation with respect to (0,0,0)
        annotation_names: list of length n where every element is a string with the name of the bounding box
        confidence: list of length n of integers where every element is a value from 0-100 representing the confidence percentage
            should be 100 for ground truth
        origins, sizes, rotations, annotation_names, confidences should all be the same size
    Returns:
        None
        """

    #Check that all lists are the same size
    lengths = [len(origins), len(sizes), len(rotations), len(annotation_names), len(confidences)]
    if lengths.count(len(origins)) != len(lengths):
        print("Frame_Bounding_Directory(): Length of lists is not equal!")
        sys.exit(2)

    #Create directory that stores the boxes in one frame
    full_path = os.path.join(path, 'bounding', str(frame_num))
    os.makedirs(full_path, exist_ok=True)
    
    #Create description.json
    description = {}
    description['num_boxes'] = len(origins)
    description_path = os.path.join(full_path, 'description.json')
    with open(description_path, 'w') as f:
        json.dump(description, f)

    #Creates JSON file that stores all the boxes in a frame
    
    json_name = 'boxes' + '.json'
    json_path = os.path.join(full_path, json_name)
    box_data = {}
    box_data['origins'] = origins
    box_data['sizes'] = sizes
    box_data['rotations'] = rotations
    box_data['annotations'] = annotation_names
    box_data['confidences'] = confidences

    with open(json_path, 'w') as f:
        json.dump(box_data, f)

def create_frame_predicted_directory(path, frame_num, origins, sizes, rotations, annotation_names, confidences):
    """Adds box data for one frame
    Args:
        path: path to LCT directory
        frame_num: frame index (0-indexed) (will overwrite if duplicate frames are specified)
        origins: [n, 3] list representing x,y,z coordinates of the center of the boxes
        sizes: [n, 3] list representing W,L,H of box
        rotations:[n, 4] list of quaternions representing box rotation with respect to (0,0,0)
        annotation_names: list of length n where every element is a string with the name of the bounding box
        confidence: list of length n of integers where every element is a value from 0-100 representing the confidence percentage
            should be 100 for ground truth
        origins, sizes, rotations, annotation_names, confidences should all be the same size
    Returns:
        None
        """

    #Check that all lists are the same size
    lengths = [len(origins), len(sizes), len(rotations), len(annotation_names), len(confidences)]
    if lengths.count(len(origins)) != len(lengths):
        print("Frame_Bounding_Directory(): Length of lists is not equal!")
        sys.exit(2)

    #Create directory that stores the boxes in one frame
    full_path = os.path.join(path, 'pred_bounding', str(frame_num))
    os.mkdir(full_path)
    
    #Create description.json
    description = {}
    description['num_boxes'] = len(origins)
    description_path = os.path.join(full_path, 'description.json')
    with open(description_path, 'w') as f:
        json.dump(description, f)

    #Creates JSON file that stores all the boxes in a frame
    
    json_name = 'boxes' + '.json'
    json_path = os.path.join(full_path, json_name)
    box_data = {}
    box_data['origins'] = origins
    box_data['sizes'] = sizes
    box_data['rotations'] = rotations
    box_data['annotations'] = annotation_names
    box_data['confidences'] = confidences
    with open(json_path, 'w') as f:
        json.dump(box_data, f)

def create_ego_directory(path, frame, translation, rotation):
    """Adds ego data for one frame
    Args:
        path: path to LCT dir
        frame: Frame number for ego data
        translation: [x,y,z] that represents translation
        rotation: [w,x,y,z] that represents rotation
    Returns:
        None
        """
    #Join path to ego dir
    full_path = os.path.join(path, 'ego')

    #Add json file for frame number
    json_path = os.path.join(full_path, str(frame) + '.json')
    ego_data = {}
    ego_data['translation'] = translation
    ego_data['rotation'] = rotation

    with open(json_path, 'w') as f:
        json.dump(ego_data, f)



def is_lct_directory(path):
    """Tests to see if specified directory conforms to LCT spec
    Args:
        path: path to LCT directory
    Returns:
        is_verified: True if LCT directory is valid or False if not
    """

    #individual verification bools
    cameras_exist = os.path.exists(os.path.join(path, "cameras"))
    inside_cameras_valid = check_inside_cameras(os.path.join(path, "cameras"))
    pointcloud_exists = os.path.exists(os.path.join(path, "pointcloud"))
    inside_pointcloud_valid = check_inside_pointcloud(os.path.join(path, "pointcloud"))
    bounding_exists = os.path.exists(os.path.join(path, "bounding"))
    inside_bounding_valid = check_inside_bounding(os.path.join(path, "bounding"))
    ego_exists = os.path.exists(os.path.join(path, "ego"))
    inside_ego_valid = check_inside_ego(os.path.join(path, "ego"))
    predicted_exists = os.path.exists(os.path.join(path, "pred_bounding"))


    #overall verification bool
    is_verified = True

    #provides feedback to the user
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
        path: path to LCT directory
    Returns:
        is_verified: false if not valid and true otherwise
    """

    is_verified = True
    
    #cameras
    for dir in os.listdir(path):
        has_in = False
        has_ex = False
        has_jpg = True
        #files in cameras
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
                if not extension == ".jpg":
                    is_verified = False
                    print("There are files in " + dir + "that are not Intrinsics.JSON, Extrinsics.JSON or .jpgs")
    
    return is_verified


def check_inside_pointcloud(path):
    """Checks to make sure that all the subdirectories of pointcloud only have .pcd files
    Prints out reason for invalidity if one exists
    Args:
        path: path to LCT directory
    Returns:
        is_verified: false if not valid and true otherwise
    """

    is_verified = True

    for dir in os.listdir(path):
        for file in os.listdir(os.path.join(path, dir)):
            extension = file[-4:]
            if not extension == ".pcd":
                is_verified = False
                print("There is a file in " + dir + " that is not a .pcd file")
    
    return is_verified

def check_inside_bounding(path):
    is_verified = True

    #loop through frames
    for dir in os.listdir(path):
        has_description = False
        has_boxes = False
        #Loop through files in frame
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

    return True

def check_inside_ego(path):
    is_verified = True

    for file in os.listdir(path):
        if not file[-4:] == ".JSON":
            is_verified = False
            print("There is a file in the ego directory that is not a json file")

    return is_verified

