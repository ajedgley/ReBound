#Licensing

#Utils for creating LCT Directory
import os
import sys
import json

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


def create_rgb_sensor_directory(path, name, images, translation, rotation, intrinsic):
    """Adds one RGB sensor directory inside camera directory
    Args:
        path: path to LCT directory
        name: name of RGB sensor
        images: list of buffers containing JPG images (assumed that length of list is also number of frames)
        translation: (x,y,z) tuple representing sensor translation
        rotation: (w,x,y,z) quaternion representing sensor rotation
        intrinsic: [3,3] 3x3 2D List representing intrinsic matrix
    Returns:
        None
        """
def create_lidar_sensor_directory(path, name, images, translation, rotation):
    """Adds one lidar sensor directory inside pointcloud directory
    Args:
        path: path to LCT directory
        name: name of lidar sensor
        images: [n, 3] list representing x,y,z coordinates (assumed that length of list is also number of frames)
        translation: (x,y,z) tuple representing sensor translation
        rotation: (w,x,y,z) quaternion representing sensor rotation
    Returns:
        None
        """

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
