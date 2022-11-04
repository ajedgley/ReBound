import getopt
import json
import os
import sys
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Quaternion
from nuscenes.utils.data_classes import Box

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
    ver_name = ""
    # Read in flags passed in with command line argument
    # Make sure that options which need an argument (namely -f for input file path and -o for output file path) have them
    # User is able to specify -h, -f, -o, -s, and -r options
    # -h brings up help menu
    # -f is used to specify the path to the Waymo file you want to read in and requires one arg. If -r is specified then this arg
    # corresponds to a directory containing all the .tfrecord files you'd like to read in
    # -o is used to specify the path to the directory where the LVT format will go.
    try:
        opts, _ = getopt.getopt(sys.argv[1:], "hf:o:s:p:rv:", "help")
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("REQUIRED: -f to specify directory of nuScenes dataset")
            print("REQUIRED: -o to specify the path where the LVT dataset will go")
            print("OPTIONAL: -p to specify a path to projected data")
            print("REQUIRED: -v to specify the version of this dataset eg: v1.0-mini")
            sys.exit(2)
        elif opt == "-f":
            input_path = arg
            input_string = input("Please enter a comma-delinated list of scene names. Remove any whitespace. To convert all scenes, leave blank\n")
            if len(input_string) != 0:
                list_of_scenes = input_string.split(",")
                scene_names.extend(list_of_scenes)
        elif opt == "-o":
            output_path = arg
        elif opt == "-p":
            pred_path = arg
        elif opt == "-v":
            ver_name = arg

        else:
            print("Invalid set of arguments entered. Please refer to -h flag for more information.")
            sys.exit(2)

    return (input_path, output_path, scene_names, pred_path, ver_name)

def extract_bounding(nusc, sample, frame_num, output_path):
    for i in range(0, len(sample['anns']) - 1):
        token = sample['anns'][i]
        annotation_metadata = nusc.get('sample_annotation', token)
        # Create nuscenes box object so we can easily transform this box to the vehicle frame that our dataset requires
        box = Box(annotation_metadata['translation'], annotation_metadata['size'], Quaternion(annotation_metadata['rotation']))

        # Get ego pose information. The LIDAR sensor has the ego information, so we can use that.
        sensor = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        poserecord = nusc.get('ego_pose', sensor['ego_pose_token'])
        
        #Transform the boxes from global frame to vehicle frame
        # box.translate(-np.array(poserecord['translation']))
        # box.rotate(Quaternion(poserecord['rotation']).inverse)

        # Store data obtained from annotation
        # print(box.center.tolist())
        # print(annotation_metadata['size'])
        # print(box.orientation.q.tolist())
        # print(annotation_metadata['category_name'])
        # print()
        # print(f"Annotation: {annotation_metadata}")
        print(f"Origin: {box.center.tolist()}")
        print(f"Rotations: {box.orientation.q.tolist()}")
        print(f"Translation: {np.array(poserecord['translation'])}")
        break

def convert_dataset(output_path, scene_name):
    # Validate the scene name passed in
    try:
        scene_token = nusc.field2token('scene', 'name', scene_name)[0]
    except Exception:
        print("\n Not a valid scene name for this dataset!")
        exit(2)
    
    scene = nusc.get('scene', scene_token)
    sample = nusc.get('sample', scene['first_sample_token'])
    frame_num = 0
    print("NuScenes Original")
    while sample['next'] != '':
        print(f"Frame: {frame_num}")
        extract_bounding(nusc, sample, frame_num, output_path)
        frame_num += 1
        sample = nusc.get('sample', sample['next'])
        break

if __name__ == "__main__":

    # Read in input database and output directory paths
    (input_path, output_path, scene_names, pred_path,ver_name) = parse_options()
    
    # Validate whether the database path passed in is valid and if the output directory path is valid
    # If the output directory exists, then use that directory. Otherwise, create a new directory at the
    # specified path. 
    print(f"Version name: {ver_name}")
    print(f"Input path: {input_path}")
    nusc = NuScenes(ver_name, input_path, True)

    path = os.getcwd()

    #If this was blank, then convert all scenes
    if len(scene_names) == 0:
        print("Converting All Scenes...")
        for scene in nusc.scene:
            scene_names.append(scene['name'])
        print(scene_names)

    for scene_name in scene_names:
        convert_dataset(output_path + "/" + scene_name, scene_name)

    print()
    print("Need this part to match NuScenes Original")
    print("JSON")
    f = open("/Users/joshualiu/CMSC435/nuScenesv1-output/scene-0061/bounding/0/boxes.json")
    data = json.load(f)
    print(data["boxes"][0])

    sample = data["boxes"][0]
    box = Box(sample["origin"], sample["size"], Quaternion(sample["rotation"]))

    f2 = open("/Users/joshualiu/CMSC435/nuScenesv1-output/scene-0061/ego/0.json")
    data2 = json.load(f2)

    box.rotate(Quaternion(data2["rotation"]))
    box.translate(np.array(data2["translation"]))
    print(f"Origin: {box.center.tolist()}")
    print(f"Rotations: {box.orientation.q.tolist()}")
    print("Translation:", np.array(data2["translation"]))