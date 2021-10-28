import os
import sys
from shutil import copyfile
#Licensing

#Utils for creating LCT Directory

#Creates top level lct directory structure at "path"
def create_lct_directory(path, name):
    """Create LCT directory at specified path
    Args:
        path: target path where directory will be stored
        name: name of root directory
    
    Returns:
        None
    """

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
def create_lidar(path, name, frame, points, translation, rotation):
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

    full_path = os.path.join(path, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)
    
    full_path = os.path.join(full_path, str(frame) + '.pcd')
    f = open(full_path, 'w')
    f.write(pcd_str)
    f.close()

def create_lidar_from_pcd(path, name, frame, input_path):
    """Copies one .pcd file to pointcloud directory
    Args:
        path: path to LCT directory
        name: name of lidar sensor
        input_path: path to .pcd file
    Returns:
        None
        """
    
    full_path = os.path.join(path, name)

    if not os.path.exists(full_path):
        os.makedirs(full_path)
    
    full_path = os.path.join(full_path, str(frame) + '.pcd')

    copyfile(input_path, full_path)


def create_frame_bounding_directory(path, name, frame_num, origin_list, size_list, rotation):
    """Adds box data for one frame
    Args:
        path: path to LCT directory
        name: name of lidar sensor
        frame_num: frame index (0-indexed) (will overwrite if duplicate frames are specified)
        origin_list: [n, 3] list representing x,y,z coordinates
        size_list: [n, 3] list representing W,L,H of box
        rotation:[n, 4] list of quaternions representing box rotation with respect to (0,0,0)
    Returns:
        None
        """

def is_lct_directory(path):
    """Tests to see if specified directory conforms to LCT spec
    Args:
        path: path to LCT directory
    Returns:
        Boolean: True or False
    """
