#Licensing


#Utils for creating LCT Directory
import os
import sys
import json
import PIL
import numpy as np
from shutil import copyfile

#Takes a 1x16 list representing a 4x4 rotation matrix
#Returns a 1x3 translation vector and a 1x4 rotation quaternion
def translation_and_rotation(transform_matrix):

    transform_array = np.array(transform_matrix).reshape(4, 4)

    #Extract the first 3 entries in the last column as the translation
    translation = tuple(transform_array[:-1, -1])

    #Extract the 3x3 rotation matrix from the upper left
    rotation_matrix = transform_array[:-1, :-1]

    #Next, convert the 3x3 matrix into quaternions:
    (eigenvalues, eigenvectors) = np.linalg.eig(rotation_matrix)
    
    #Find the eigenvector with eigenvalue 1, which is guaranteed to exist
    for i in range(3):
        if np.isclose(eigenvalues[i], 1):
            eigenu = eigenvectors[:,i]

    #Check whether sin(theta/2) is positive or negative
    not_parallel = [1, 1, 1] if eigenu[0]*eigenu[1] < 0 else [1, -1, 1]
    orthogonal = np.cross(eigenu, not_parallel)
    sign_check = np.dot(np.cross(orthogonal, np.dot(rotation_matrix, orthogonal)), eigenu)
    sine_sign = 1 if (sign_check > 0) else -1

    #Calculated trig functions for the quaternion
    trace = np.trace(rotation_matrix)
    cos_theta = (trace - 1) / 2
    sine_half_theta = sine_sign*((1 - cos_theta)/ 2)**0.5
    cos_half_theta = ((1 + cos_theta)/ 2)**0.5

    rotation = (sine_half_theta * eigenu[0].real, sine_half_theta * eigenu[1].real, sine_half_theta * eigenu[2].real, cos_half_theta)
    return translation, rotation


#Creates top level lct directory structure at "path"
def create_lct_directory(path, name):
    """Create LCT directory at specified path
    Args:
        path: target path where directory will be stored
    
    Returns:
        None
    """
    sub_directories = ['cameras', 'pointcloud', 'bounding', 'ego']
    try:
        parent_path = os.path.join(path, name)
        os.makedirs(parent_path)
        print("added new folder")
        for directory in sub_directories:
            full_path = os.path.join(parent_path, directory)
            os.mkdir(full_path)
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

    #Create necessary directories: ./Cameras and ./Cameras/[name]. These directories are needed for the
    #file, so the program creates them.
    try:
        os.mkdir(os.path.join(path, "cameras"))
    except FileExistsError:
        pass
    
    work_dir = os.path.join(os.path.join(path, "cameras"), name)
    try:
        os.mkdir(work_dir)
    except FileExistsError:
        pass

    #Creates the Extrinsic.json file in the /Cameras/[name] directory from the
    #translation and rotation parameters.
    extrinsics = {}
    extrinsics['translation'] = translation
    extrinsics['rotation'] = rotation


    extrinsic_file = open(work_dir + "/extrinsics.json", "w")
    extrinsic_file.write(json.dumps(extrinsics))
    extrinsic_file.close()


    #Creates the Extrinsic.json file in the /Cameras/[name] directory from the intrinsic parameter.
    intrinsic_file = open(work_dir + "/intrinsics.json", "w")
    intrinsic_file.write(json.dumps({"matrix" : intrinsic}))
    intrinsic_file.close()


def add_rgb_frame(path, name, image, frame):
    """Adds one jpg from one frame to the structure inside the camera directory for a given sensor
    Args:
        path: path to LCT directory
        name: name of RGB sensor
        images: list of buffers containing JPG images (assumed that length of list is also number of frames)
        frame: the number corresponding to the frame
    Returns:
        None
        """

    #Assumes directory exists
    image.save(os.path.join(os.path.join(os.path.join(path, "cameras"), name),
    str(frame) +".jpg"))

def add_rgb_frame_from_jpg(path, name, frame, input_path):
    """Copies existing jpg file into the cameras directory
    Args:
        path: path to LCT directory
        name: name of RGB sensor
        frame: the number corresponding to the frame
        input_path: Path to source jpg iamge
    Returns:
        None
        """
    full_path = os.path.join(path, 'cameras', name, str(frame) + '.jpg')
    copyfile(input_path, full_path)

def create_lidar_sensor_directory(path, name):
    full_path = os.path.join(path, 'pointcloud', name)

    try:
        os.makedirs(full_path)
    except FileExistsError:
        pass

def add_lidar_frame(path, name, frame, points, translation, rotation):
    """Adds one lidar sensor directory inside pointcloud directory
    Args:
        path: path to LCT directory
        name: name of lidar sensor
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

    full_path = os.path.join(path, 'pointcloud', name, str(frame) + '.pcd')
    
    f = open(full_path, 'w')
    f.write(pcd_str)
    f.close()

def add_lidar_frame_from_pcd(path, name, frame, input_path):
    """Copies one .pcd file to pointcloud directory
    Args:
        path: path to LCT directory
        name: name of lidar sensor
        input_path: path to .pcd file
    Returns:
        None
        """
    
    full_path = os.path.join(path, 'pointcloud', name, str(frame) + '.pcd')

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
    print(full_path)
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
        Boolean: True or False
    """

    #individual verification bools
    cameras_exist = os.path.exists(os.path.join(path, "cameras"))
    inside_cameras_valid = check_inside_cameras(os.path.join(path, "cameras"))
    pointcloud_exists = os.path.exists(os.path.join(path, "pointcloud"))
    inside_pointcloud_valid = check_inside_pointcloud(os.path.join(path, "pointcloud"))
    bounding_exists = os.path.exists(os.path.join(path, "bounding"))
    ego_exists = os.path.exists(os.path.join(path, "ego"))

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
    if not ego_exists:
        print("There is no directory named \"ego\" at the selected path. \n")
        is_verified = False
    
    return is_verified


#Checks to make sure that all the subdirectories of cameras only have Extrinsic.json, Intrinsic.json and .jpg files
#Parameter is the path to the cameras dir
#Returns false if not valid and true otherwise
#Will print out reason for invalidity if one exists
def check_inside_cameras(path):
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


#Checks to make sure that all the subdirectories of cameras only have .pcd files
#Returns false if not valid and true otherwise
#Will print out reason for invalidity if one exists
def check_inside_pointcloud(path):
    is_verified = True

    for dir in os.listdir(path):
        for file in os.listdir(os.path.join(path, dir)):
            extension = file[-4:]
            if not extension == ".pcd":
                is_verified = False
                print("There is a file in " + dir + " that is not a .pcd file")
    
    return is_verified


