import getopt
import json
import re
import open3d as o3d
import os
import sys
import numpy as np
from nuscenes.utils.data_classes import Quaternion
from nuscenes.utils.data_classes import Box
from secrets import token_hex
from utils import dataformat_utils
from utils import geometry_utils

def parse_options():
    ''' Read in user command line input to get directory paths which will be used for input and output.
    Args:
        None
    Returns:
        input_path: Path to generic dataset being read into LVT
        output_path: Path where user wants LVT to export to, and has the original dataset
        scene_name: Name of the scene in NuScenes
        pred_path: Path to data based on a model's predictions
    '''
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

def extract_frame(frame_num, input_path):
    ''' Extracts data from a generic frame that is needed for an existing nuScenes data
    Args:
        frame_num: frame number
        input_path: path to generic data
    Returns:
        None
    '''

    # Necessary files from generic data format
    with open(input_path + "/bounding/" + str(frame_num) + "/boxes.json") as f:
            bounding = json.load(f)
    with open(input_path + "/ego/" + str(frame_num) + ".json") as f:
            ego = json.load(f)
    pcd = o3d.io.read_point_cloud(input_path + "pointcloud/LIDAR_TOP/" + str(frame_num) + ".pcd").points

    for i in range(len(bounding["boxes"])):
        # Reverting bounding box
        bounding_box = bounding["boxes"][i]
        box = Box(bounding_box["origin"], bounding_box["size"], Quaternion(bounding_box["rotation"]))
        box.rotate(Quaternion(ego["rotation"]))
        box.translate(np.array(ego["translation"]))

        # Update annotation
        # use ann_token if exists, else create new token
        ann_token = (bounding_box["data"]["ann_token"] if "ann_token" in bounding_box["data"] else token_hex(16))
        sample_token = timestamps[frame_num]
        # use instance_token if exists, else create new token
        instance_token = (bounding_box["id"] if bounding_box["id"] != "" else token_hex(16))
        data = {}
        data["token"] = ann_token
        data["sample_token"] = sample_token
        data["instance_token"] = instance_token
        # Default to none
        data["attribute_tokens"] = (sample_annotations[ann_token]["attribute_tokens"] if ann_token in sample_annotations else [])
        # Default to 4
        data["visibility_token"] = (sample_annotations[ann_token]["visibility_token"] if ann_token in sample_annotations else 4)
        # Use original translation, size, and rotation if not edited/added
        if (ann_token in sample_annotations) and (bounding_box["data"]):
            data["translation"] = sample_annotations[ann_token]["translation"]
            data["size"] = sample_annotations[ann_token]["size"]
            data["rotation"] = sample_annotations[ann_token]["rotation"]
            data["num_lidar_pts"] = sample_annotations[ann_token]["num_lidar_pts"]
            data["num_radar_pts"] = sample_annotations[ann_token]["num_radar_pts"]
        else:
            data["translation"] = box.center.tolist()
            data["size"] = bounding_box["size"]
            data["rotation"] = box.orientation.q.tolist()
            # Calculating num_lidar_pts and num_radar_pts may be expensive
            # data["num_lidar_pts"] = geometry_utils.compute_interior_points(bounding_box, pcd)
            data["num_lidar_pts"] = bounding_box["internal_pts"]
            data["num_radar_pts"] = 0
        # Update this later
        data["prev"] = "" 
        data["next"] = ""
        new_annotation[data["token"]] = data

        # Update instance
        data = {}
        data["token"] = instance_token
        if bounding_box["annotation"] in category:
            # Category exists
            data["category_token"] = category[bounding_box["annotation"]]["token"]
        else:
            # Add new category
            category_token = token_hex(16)
            data2 = {}
            data2["category_token"] = category_token
            data2["name"] = bounding_box["annotation"]
            data2["description"] = "new category"
            category[data2["name"]] = data2

            data["category_token"] = category_token
        if instance_token not in new_instance:
            # Create new instance
            data["nbr_annotations"] = 1
            data["first_annotation_token"] = ann_token
            data["last_annotation_token"] = ann_token

            # Update previous ann_token
            prev_ann_token[instance_token] = ann_token
            new_instance[data["token"]] = data
        else:
            # Update existing instance
            # Each instance will have one category (based on the latest annotation)
            new_instance[instance_token]["category_token"] = category[bounding_box["annotation"]]["token"]
            new_instance[instance_token]["nbr_annotations"] += 1
            new_instance[instance_token]["last_annotation_token"] = ann_token 

            # Add to end of linked list and update previous ann_token
            new_annotation[prev_ann_token[instance_token]]["next"] = ann_token
            new_annotation[ann_token]["prev"] = prev_ann_token[instance_token]
            prev_ann_token[instance_token] = ann_token
    
    try:
        with open(input_path + "/pred_bounding/" + str(frame_num) + "/boxes.json") as f:
            pred_bounding = json.load(f)
    except FileNotFoundError:
        return
    except:
        print("Error in reading pred_bounding")

    for i in range(len(pred_bounding["boxes"])):
        pred_data = {}
        # Reverting bounding box
        bounding_box = pred_bounding["boxes"][i]
        box = Box(bounding_box["origin"], bounding_box["size"], Quaternion(bounding_box["rotation"]))
        box.rotate(Quaternion(ego["rotation"]))
        box.translate(np.array(ego["translation"]))

        sample_token = timestamps[frame_num]
        pred_data["sample_token"] = sample_token
        
        pred_data["translation"] = box.center.tolist()
        pred_data["size"] = bounding_box["size"]
        pred_data["rotation"] = box.orientation.q.tolist()

        pred_data["detection_name"] = bounding_box["annotation"]
        pred_data["detection_score"] = bounding_box["confidence"] / 100
        pred_data["attribute_name"] = bounding_box["data"]["attribute"]
        pred_data["velocity"] = bounding_box["data"]["pred_velocity"]

        if "results" in new_predictions:
            if sample_token in new_predictions["results"]:
                new_predictions["results"][sample_token].append(pred_data)
            else:
                new_predictions["results"][sample_token] = [pred_data]
        else:
            new_predictions["results"] = {sample_token: [pred_data]}


def revert_dataset(input_path, output_path):
    # Setup progress bar
    frame_num = 0
    frame_count = 0
    for d in os.scandir(input_path + "bounding/"):
        res = re.match(r"\d+",d.name)
        if d.is_dir() and res:
            frame_count += 1
    dataformat_utils.print_progress_bar(0, frame_count)

    # Iterate through each frame
    while frame_num < frame_count:
        extract_frame(frame_num, input_path)
        frame_num += 1
        dataformat_utils.print_progress_bar(frame_num, frame_count)

    # TODO: Write updated json files to input
    with open(output_path + "/sample_annotation.json","w") as f:
        json.dump(list(new_annotation.values()), f, indent=0)
    with open(output_path + "/sample.json","w") as f:
        json.dump(list(samples.values()), f, indent=0)
    with open(output_path + "/category.json","w") as f:
        json.dump(list(category.values()), f, indent=0)
    with open(output_path + "/instance.json","w") as f:
        json.dump(list(new_instance.values()), f, indent=0)
    if pred_path is not None and pred_path != "":
        with open(pred_path + "/predictions.json","w") as f:
            json.dump(new_predictions, f, indent=0)

if __name__ == "__main__":
    # Read in input database and output directory paths
    (input_path, output_path, scene_names, pred_path, ver_name) = parse_options()
    
    print(f"Version name: {ver_name}")
    print(f"Input path: {input_path}")

    # Verify input path exists
    input_path += ("" if input_path[-1] == "/" else "/")
    output_path += ("" if output_path[-1] == "/" else "/")
    if not os.path.exists(input_path):
        sys.exit("Invalid input path. Please check paths entered and try again.")
    if not os.path.exists(output_path + ver_name):
        sys.exit("Invalid output path. Please check paths entered and try again.")

    # Necessary files from original nuscenes dataset to update annotations
    with open(output_path + ver_name + "/sample_annotation.json") as f:
        data = json.load(f)
        sample_annotations = {}
        for i in range(len(data)):
            sample_annotations[data[i]["token"]] = data[i]
    with open(output_path + ver_name + "/sample.json") as f:
        data = json.load(f)
        samples = {}
        for i in range(len(data)):
            samples[data[i]["token"]] = data[i]
    with open(output_path + ver_name +  "/category.json") as f:
        data = json.load(f)
        category = {}
        for i in range(len(data)):
            category[data[i]["name"]] = data[i]
    with open(output_path + ver_name + "/instance.json") as f:
        data = json.load(f)
        instance = {}
        for i in range(len(data)):
            instance[data[i]["token"]] = data[i]
    # Recreating annotation and instance
    new_annotation = {}
    new_instance = {}
    # Keeping track of previous ann_token for each instance
    prev_ann_token = {}
    new_predictions = {}

    # If this was blank, then revert all scenes
    if len(scene_names) == 0:
        print("Converting All Scenes...")
        for scene in os.scandir(input_path):
            res = re.match(r"scene-\d{4}",scene.name)
            if not res:
                continue
            scene_names.append(scene.name)
        print(scene_names)

    # Revert all the scenes
    for scene_name in scene_names:
        # Verify correct original format
        with open(input_path + scene_name +  "/metadata.json") as f:
            metadata = json.load(f)
            if metadata["source-format"] != "nuScenes":
                input_string = input("The original and exporting format do not match, do you wish to continue (y/n)?\n").lower()
                # exporting between formats not supported
                if input_string == 'y' or input_string == 'yes':
                    sys.exit("Exporting to different format not yet supported")
                # skip scene
                continue

        with open(input_path + scene_name + "/timestamps.json") as f:
            data = json.load(f)
            timestamps = data["timestamps"]
        print("Reverting scene: " + scene_name)
        revert_dataset(input_path + scene_name + "/", output_path + ver_name)