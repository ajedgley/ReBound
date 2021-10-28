import os
import sys
#Licensing

#Utils for creating LCT Directory

import os
import json
import PIL

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
        os.mkdir(os.path.join(path, "Cameras"))
    except FileExistsError:
        pass
    
    work_dir = os.path.join(os.path.join(path, "Cameras"), name)
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
    image.save(os.path.join(os.path.join(os.path.join(path, "Cameras"), name),
    str(frame) +".jpg"))


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

def import_waymo_debug(filepath):
    import tensorflow as tf
    from waymo_open_dataset import dataset_pb2 as open_dataset
    import io

    dataset = tf.data.TFRecordDataset(os.path.join(filepath,"training0.tfrecord"),'')


    rgb = True
    image_arr = []
    i = 3
    for data in dataset:
            i -= 1
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            image = PIL.Image.open(io.BytesIO(frame.images[0].image))
            image_arr.append(image)
            if(not(i)):
                break

    if rgb:
        #We wrap the raw jpg data using BytesIO to avoid saving a temporary file using BytesIO
        #Then we give the function pointer created by bytesIo to PIL to open it in a standard format
        image = PIL.Image.open(io.BytesIO(frame.images[0].image))
        #convert the  PIL image object to a numpy array using asarray(), since that is what open3d expects
        #image = o3d.geometry.Image(np.asarray(image))
    
    return image_arr

if (__name__ == "__main__"):

    import shutil

    dir_path = "/home/mbetberg/Documents/cmsc435_start/lidar-data"

    img = import_waymo_debug(dir_path)

    try:
        shutil.rmtree(dir_path + "Cameras")
    except OSError as e:
        pass

    create_rgb_sensor_directory(dir_path, "Camera1", (1, 2, 3),
    (1, 2, 3, 4), [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    fram = 0
    for i in img:
        add_rgb_frame(dir_path, "Camera1", i, fram)
        fram += 2

