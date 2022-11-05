import getopt
import json
import os
import sys
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Quaternion
from nuscenes.utils.data_classes import Box

THRESHOLD = 10**-8

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

def extract_bounding(sample, frame_num, output_path):
    for i in range(0, len(sample['anns']) - 1):
        # Reverting bounding box
        with open(output_path + "/bounding/" + str(frame_num) + "/boxes.json") as f1:
            data = json.load(f1)
        bounding_box = data["boxes"][i]
        box = Box(bounding_box["origin"], bounding_box["size"], Quaternion(bounding_box["rotation"]))

        with open(output_path + "/ego/" + str(frame_num) + ".json") as f2:
            ego = json.load(f2)

        box.rotate(Quaternion(ego["rotation"]))
        box.translate(np.array(ego["translation"]))

        # Find corresponding annotation in json file
        # TODO: Need to fix for adding and deleting annotations
        token = sample['anns'][i]
        
        # Update annotations json object
        # TODO: Ask about floating point errors (zero out doesn't seem to work)
        annotations[token]["translation"] = box.center.tolist()
        annotations[token]["size"] = bounding_box["size"]
        annotations[token]["rotation"] = box.orientation.q.tolist()
        # zero out small numbers
        # for k in range(len(annotations[j]["translation"])):
        #     annotations[j]["translation"][k] = 0 if annotations[j]["translation"][k] < THRESHOLD else annotations[j]["translation"][k]
        # for k in range(len(annotations[j]["size"])):
        #     annotations[j]["size"][k] = 0 if annotations[j]["size"][k] < THRESHOLD else annotations[j]["size"][k]
        # for k in range(len(annotations[j]["rotation"])):
        #     annotations[j]["rotation"][k] = 0 if annotations[j]["rotation"][k] < ZERO else annotations[j]["rotation"][k]

        # annotation_metadata = nusc.get('sample_annotation', token)
        # if annotation_metadata['category_name'] != bounding_box["annotation"]:
        category_token = category[bounding_box["annotation"]]
        instance_token = annotations[token]["instance_token"]
        # update instance, will update category for the entire entity
        instance[instance_token]["category_token"] = category_token

        # using name find token in category.json
        # print("Category_token")
        # annotation_metadata = nusc.get("sample_annotation", token)
        # with open(input_path+"/category.json") as f3:
        #     category = json.load(f3)
        # for k in range(len(category)):
        #     if category[k]["name"] == annotation_metadata["category_name"]:
        #         category_token = category[k]["token"]
        #         print(category_token)
        # using category_token find token in instance.json (may need to update nbr_annotations, first_annotation_token, last_annotation_token)\
        # print("Token")
        # with open(input_path+"/instance.json") as f4:
        #     instance = json.load(f4)
        # for k in range(len(instance)):
        #     if instance[k]["category_token"] == category_token:
        #         token = instance[k]["token"]
        #         print(token)
        # update instance_token using token in sample_annotation.json


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

    # Iterate through each frame
    while sample['next'] != '':
        # print(f"Frame: {frame_num}")
        extract_bounding(sample, frame_num, output_path)
        frame_num += 1
        sample = nusc.get('sample', sample['next'])

    # TODO: Write annotations and instances somewhere
    with open("/Users/joshualiu/CMSC435/updated_annotations.json","w") as f1:
        json.dump(list(annotations.values()), f1, indent=0)
    with open("/Users/joshualiu/CMSC435/updated_instance.json","w") as f2:
        json.dump(list(instance.values()), f2, indent=0)

# sanity check
def compare_nescene():
    print("Sanity Check")
    with open("/Users/joshualiu/CMSC435/nuScenesv1/v1.0-mini/v1.0-mini/sample_annotation.json") as f1:
        data1 = json.load(f1)
    with open("/Users/joshualiu/CMSC435/updated_annotations.json") as f2:
        data2 = json.load(f2)
    
    for i in range(len(data1)):
        if data1[i]["token"] != data2[i]["token"]:
            print("Token doesn't match")
        if data1[i]["sample_token"] != data2[i]["sample_token"]:
            print("Sample token doesn't match")
        if data1[i]["instance_token"] != data2[i]["instance_token"]:
            print("Instance token doesn't match")
        if data1[i]["visibility_token"] != data2[i]["visibility_token"]:
            print("Visibility token doesn't match")
        for j in range(len(data1[i]["translation"])):
            if abs(data1[i]["translation"][j] - data2[i]["translation"][j]) > THRESHOLD:
                print(data1[i]["translation"][j], data2[i]["translation"][j])
                print("Translation doesn't match")
        for j in range(len(data1[i]["size"])):
            if abs(data1[i]["size"][j] - data2[i]["size"][j]) > THRESHOLD:
                print("Size doesn't match")
        for j in range(len(data1[i]["rotation"])):
            if abs(data1[i]["rotation"][j] - data2[i]["rotation"][j]) > THRESHOLD:
                print("Rotation doesn't match")
        if data1[i]["prev"] != data2[i]["prev"]:
            print("Prev doesn't match")
        if data1[i]["next"] != data2[i]["next"]:
            print("Next doesn't match")
        if data1[i]["num_lidar_pts"] != data2[i]["num_lidar_pts"]:
            print("Num lidar pts doesn't match")
        if data1[i]["num_radar_pts"] != data2[i]["num_radar_pts"]:
            print("Num radar pts doesn't match")

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

    # Necessary files to update annotations
    with open(input_path + ver_name + "/sample_annotation.json") as f1:
        data = json.load(f1)
        annotations = {}
        for i in range(len(data)):
            annotations[data[i]["token"]] = data[i]
    with open(input_path + ver_name +  "/category.json") as f2:
        data = json.load(f2)
        category = {}
        for i in range(len(data)):
            category[data[i]["name"]] = data[i]["token"]
    with open(input_path + ver_name + "/instance.json") as f3:
        data = json.load(f3)
        instance = {}
        for i in range(len(data)):
            instance[data[i]["token"]] = data[i]

    #If this was blank, then convert all scenes
    if len(scene_names) == 0:
        print("Converting All Scenes...")
        for scene in nusc.scene:
            scene_names.append(scene['name'])
        print(scene_names)

    for scene_name in scene_names:
        convert_dataset(output_path + scene_name, scene_name)

    compare_nescene()