#Licensing

#imports
import os.path
import os

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


#TODO implement
#Checks to make sure that all the subdirectories of cameras only have Extrinsic.json, Intrinsic.json and .jpg files
#Parameter is the path to the cameras dir
#Returns false if not valid and true otherwise
#Will print out reason for invalidity if one exists
def check_inside_cameras(path):
    for dir in os.listdir(path):
        print()
    
    return True

#TODO implement
#Checks to make sure that all the subdirectories of cameras only have .pcd files
#Returns false if not valid and true otherwise
#Will print out reason for invalidity if one exists
def check_inside_pointcloud(path):
    print()
    return true


def main():
    answer = is_lct_directory("/home/avetter/Desktop/Dev/lidar/src")

    if answer:
        print("it works")
    else:
        print("doesnt work")

if __name__ == "__main__":
    main()