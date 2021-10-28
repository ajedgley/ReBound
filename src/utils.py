#Licensing

#Utils for creating LCT Directory
import os
import sys
import json
import PIL
from shutil import copyfile

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


