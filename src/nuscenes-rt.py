import getopt
import json
import os
import sys
import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Quaternion
from nuscenes.utils.data_classes import Box
from secrets import token_hex

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
    with open(output_path + "/bounding/" + str(frame_num) + "/boxes.json") as f:
            bounding = json.load(f)
    with open(output_path + "/ego/" + str(frame_num) + ".json") as f:
            ego = json.load(f)

    tokens = []
    # TODO: test edit and add
    for i in range(len(bounding["boxes"])):
        # Reverting bounding box
        bounding_box = bounding["boxes"][i]
        box = Box(bounding_box["origin"], bounding_box["size"], Quaternion(bounding_box["rotation"]))
        box.rotate(Quaternion(ego["rotation"]))
        box.translate(np.array(ego["translation"]))

        if "nuscenes" in bounding_box["data"]:
            # Edit annotation
            token = bounding_box["data"]["nuscenes"]
            tokens.append(token)
        
            # Update annotations json object
            # TODO: Ask about floating point errors (zero out doesn't seem to work)
            sample_annotations[token]["translation"] = box.center.tolist()
            sample_annotations[token]["size"] = bounding_box["size"]
            sample_annotations[token]["rotation"] = box.orientation.q.tolist()

            # Update instance, will update category for the entire entity
            # TODO: Will a new category be added, or will one of the existing ones be used
            category_token = category[bounding_box["annotation"]]
            instance_token = sample_annotations[token]["instance_token"]
            instance[instance_token]["category_token"] = category_token
        else:
            # Add annotation
            # Add to sample_annotation
            ann_token = token_hex(16)
            sample_token = token_hex(16)
            instance_token = token_hex(16)
            print("Adding", ann_token)
            data = {}
            data["token"] = ann_token
            data["sample_token"] = sample_token
            data["instance_token"] = instance_token
            data["attribute_tokens"] = [] # TODO: look into
            data["visibility_token"] = -1 # TODO: calculate
            data["translation"] = box.center.tolist()
            data["size"] = bounding_box["size"]
            data["rotation"] = box.orientation.q.tolist()
            data["num_lidar_pts"] = -1 # TODO: calculate
            data["num_radar_pts"] = -1 # TODO: calculate
            data["prev"] = "" # not able to calculate
            data["next"] = "" # not able to calculate
            sample_annotations[data["token"]] = data
            # Add to instance
            data = {}
            data["token"] = instance_token
            data["category_token"] = category[bounding_box["annotation"]]
            data["nbr_annotations"] = 1
            data["first_annotation_token"] = ann_token
            data["last_annotation_token"] = ann_token
            instance[data["token"]] = data
            # Add to sample
            scene_token = nusc.field2token('scene', 'name', scene_name)[0]
            data = {}
            data["token"] = sample_token
            data["timestamp"] = -1 # TODO: maybe be possible to get
            data["prev"] = -1 # TODO: need timestamp
            data["next"] = -1 # TODO: need timestamp
            data["scene_token"] = scene_token
            samples[data["token"]] = data
            # Update scene
            scenes[scene_token]["nbr_samples"] += 1
            # TODO: need timestamp to update "first_sample_token" and "last_sample_token"

    # TODO: test delete
    for i in range(0, len(sample['anns']) - 1):
        # Find deleted annotations
        ann_token = sample['anns'][i]
        if ann_token in tokens:
            continue
        print("Deleting", ann_token)

        # Delete annotation
        # Delete from sample_annotations and instance
        prev = sample_annotations[ann_token]["prev"]
        next = sample_annotations[ann_token]["next"]
        sample_token = sample_annotations[ann_token]["sample_token"]
        instance_token = sample_annotations[ann_token]["instance_token"]
        sample_annotations.pop(ann_token)
        instance[instance_token]["nbr_annotations"] -= 1
        if prev == "" and next == "":
            instance[instance_token]["first_annotation_token"] = next
            instance[instance_token]["last_annotation_token"] = prev
        elif prev == "":
            instance[instance_token]["prev"] = ""
            instance[instance_token]["first_annotation_token"] = next
        elif next == "":
            instance[instance_token]["next"] = ""
            instance[instance_token]["last_annotation_token"] = prev
        else:
            instance[instance_token]["next"] = next
            instance[instance_token]["prev"] = prev
        # Delete from sample and scene
        prev = samples[sample_token]["prev"]
        next = samples[sample_token]["next"]
        scene_token = samples[sample_token]["scene_token"]
        samples.pop(sample_token)
        scenes[scene_token]["nbr_samples"] -= 1
        if prev == "" and next == "":
            scenes[scene_token]["first_sample_token"] = next
            scenes[scene_token]["last_sample_token"] = prev
        elif prev == "":
            samples[next]["prev"] = ""
            scenes[scene_token]["first_sample_token"] = next
        elif next == "":
            samples[prev]["next"] = ""
            scenes[scene_token]["last_sample_token"] = prev
        else:
            samples[prev]["next"] = next
            samples[next]["prev"] = prev

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

    # TODO: Write updated json files somewhere
    with open("/Users/joshualiu/CMSC435/revert/sample_annotations.json","w") as f:
        json.dump(list(sample_annotations.values()), f, indent=0)
    with open("/Users/joshualiu/CMSC435/revert/sample.json","w") as f:
        json.dump(list(samples.values()), f, indent=0)
    with open("/Users/joshualiu/CMSC435/revert/scene.json","w") as f:
        json.dump(list(scenes.values()), f, indent=0)
    with open("/Users/joshualiu/CMSC435/revert/category.json","w") as f:
        json.dump(list(category.values()), f, indent=0)
    with open("/Users/joshualiu/CMSC435/revert/instance.json","w") as f:
        json.dump(list(instance.values()), f, indent=0)

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
    with open(input_path + ver_name + "/sample_annotation.json") as f:
        data = json.load(f)
        sample_annotations = {}
        for i in range(len(data)):
            sample_annotations[data[i]["token"]] = data[i]
    with open(input_path + ver_name + "/sample.json") as f:
        data = json.load(f)
        samples = {}
        for i in range(len(data)):
            samples[data[i]["token"]] = data[i]
    with open(input_path + ver_name + "/scene.json") as f:
        data = json.load(f)
        scenes = {}
        for i in range(len(data)):
            scenes[data[i]["token"]] = data[i]
    with open(input_path + ver_name +  "/category.json") as f:
        data = json.load(f)
        category = {}
        for i in range(len(data)):
            category[data[i]["name"]] = data[i]
    with open(input_path + ver_name + "/instance.json") as f:
        data = json.load(f)
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
