"""
argoverse-rt.py

Conversion tool for exporting edited annotations back into an argoverse dataset
"""
import getopt
import json
import os
import pandas as pd
import re
import sys
from utils import dataformat_utils
from utils import geometry_utils
import open3d as o3d
import uuid

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
    scene_names = []

    try:
        opts, _ = getopt.getopt(sys.argv[1:], "hf:o:", "help")
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("REQUIRED: -f to specify the path to the LVT file.")
            print("REQUIRED: -o to specify the path to the argoverse file.")
            sys.exit(2)
        elif opt == "-f":
            input_path = arg
            input_string = input("Please enter a comma-delinated list of scene names. Remove any whitespace. To convert all scenes, leave blank.\n")
            if len(input_string) != 0:
                list_of_scenes = input_string.split(",")
                scene_names.extend(list_of_scenes)
        elif opt == "-o":
            output_path = arg
        else:
            sys.exit(2)

    return (input_path, output_path, scene_names)

def export_annotations(input_path, output_path):
    ''' Extracts data from a generic frame that is needed for an existing argoverse dataset
    Args:
        frame_num: frame number
        input_path: path to generic data
    Returns:
        None
    '''

    # Setup progress bar
    frame_count = 0
    for d in os.scandir(input_path + "bounding/"):
        res = re.match(r"\d+",d.name)
        if d.is_dir() and res:
            frame_count += 1
    dataformat_utils.print_progress_bar(0, frame_count)

    # read timestamps
    timestamps = pd.read_csv(input_path + 'timestamps.csv')

    bounding_path = input_path + 'bounding'

    row_list = []

    frame_num = 0
    while os.path.exists(os.path.join(bounding_path, str(frame_num))):
        annotations = json.load(open(os.path.join(bounding_path, str(frame_num), "boxes.json"), "r"))
        # load pointcloud points
        pcd = o3d.io.read_point_cloud(input_path + "pointcloud/lidar/" + str(frame_num) + ".pcd").points
        for box in annotations["boxes"]:
            # assign a new random tracking id if one is not stored
            if box["id"] == "":
                box["id"] = str(uuid.uuid4)
                # calulates internal points if it is not stored
                box["internal_pts"] = geometry_utils.compute_interior_points(box, pcd)
            row_list.append({
                "timestamp_ns": timestamps.loc[frame_num]["timestamps"],
                "track_uuid": box["id"],
                "category": box["annotation"],
                "length_m": box["size"][1],
                "width_m": box["size"][0],
                "height_m": box["size"][2],
                "qw": box["rotation"][0],
                "qx": box["rotation"][1],
                "qy": box["rotation"][2],
                "qz": box["rotation"][3],
                "tx_m": box["origin"][0],
                "ty_m": box["origin"][1],
                "tz_m": box["origin"][2],
                "num_interior_pts": box["internal_pts"],
            })
        
        frame_num += 1
        dataformat_utils.print_progress_bar(frame_num, frame_count)
    
    df = pd.DataFrame(row_list)
    df.to_feather(os.path.join(output_path, "annotations.feather"))

if __name__ == "__main__":
    (input_path, output_path, scene_names) = parse_options()
    print(f"Input path: {input_path}")
    print(f"Output path: {output_path}")

    # Verify input path exists
    if not os.path.exists(input_path):
        sys.exit("Invalid input path. Please check paths entered and try again.")
    # Verify output path exists
    if not os.path.exists(input_path):
        sys.exit("Invalid output path. Please check paths entered and try again.")
    input_path += ("" if input_path[-1] == "/" else "/")
    output_path += ("" if output_path[-1] == "/" else "/")

    # Verify scenes exists
    if len(scene_names) == 0:
        print("Converting All Scenes...")
        for scene in os.scandir(input_path):
            if scene.is_dir():
                scene_names.append(scene.name)
        print(scene_names)
    
    # Export all the scenes
    for scene_name in scene_names:
        # Verify correct original format
        with open(input_path + scene_name +  "/metadata.json") as f:
            metadata = json.load(f)
            if metadata["source-format"] != "Argoverse":
                input_string = input("The original and exporting format do not match, do you wish to continue (y/n)?\n").lower()
                # exporting between formats not supported
                if input_string == 'y' or input_string == 'yes':
                    sys.exit("Exporting to different format not yet supported")
                # skip scene
                continue

        export_annotations(input_path+scene_name+"/", output_path+scene_name+"/")
