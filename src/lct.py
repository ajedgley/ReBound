"""
lct.py

Conversion tool to bring nuscenes dataset into LVT. 
"""

import getopt
import matplotlib.colors
import matplotlib.pyplot as plt
from PIL import Image
import sys
import numpy as np
import open3d.visualization.gui as gui
from PIL import Image
import open3d as o3d
import open3d.visualization.rendering as rendering
import json
import os
import cv2
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

from utils import geometry_utils
from operator import itemgetter
import platform

OS_STRING = platform.system()
ORIGIN = 0
SIZE = 1
ROTATION = 2
ANNOTATION = 3
CONFIDENCE = 4
COLOR = 5

#Taken from http://phrogz.net/tmp/24colors.html
colorlist = [(255,0,0), (255,255,0), (0,234,255), (170,0,255), (255,127,0), (191,255,0), (0,149,255), (255,0,170), (255,212,0), (106,255,0), (0,64,255), (185,237,224), (143,35,35), (35,98,143), (107,35,143), (79,143,35)]
# Parse CLI args and validate input
def parse_options():
    """ This parses the CLI arguments and validates the arguments
        Returns:
            the path where the LCT directory is located
    """
    input_path = ""
    
    # read in flags passed in with command line argument
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:", "help")
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    # make sure that options which need an argument (namely -f for the input file path) have them
    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("use -f to specify the directory of LVT dataset")
            sys.exit(2)
        elif opt == "-f": # and len(opts) == 2:
            input_path = arg
        else:
            # only reach here if the the arguments were incorrect
            print("Invalid set of arguments entered. Please refer to -h flag for more information.")
            sys.exit(2)

    return input_path  

class Window:
    MENU_IMPORT = 1
    def __init__(self, lct_dir):
        
        np.set_printoptions(precision=15)

        # Create the objects for the 3 windows that appear when running the application
        self.controls = gui.Application.instance.create_window("LCT", 400, 800)
        self.pointcloud_window = gui.Application.instance.create_window("PointCloud", 640, 480)
        self.image_window = gui.Application.instance.create_window("Image", 640, 480)

        # Set starting values for variables related to data paths
        # In the future, these values should not be set hard coded directly. Sensible default values should
        # be extracted from the LVT Directory
        self.lct_path = lct_dir
        self.camera_sensors, self.lidar_sensors = self.get_cams_and_pointclouds(self.lct_path)
        self.box_data_name = ["bounding"]
        self.min_confidence = 50
        self.highlight_faults = False
        self.show_false_positive = False
        self.show_incorrect_annotations = False
        self.boxes_to_render = []
        self.show_score = False
        self.label_list = []
        # These three values represent the current LiDAR sensors, RGB sensors, and annotations being displayed
        self.rgb_sensor_name = self.camera_sensors[0]
        self.lidar_sensor_name = self.lidar_sensors[0]
        self.filter_arr = []
        self.pred_filter_arr = []
        self.compare_bounding = False
        self.color_map = {}
        self.pred_color_map = {}
        self.pcd_path = os.path.join(self.lct_path, "pointcloud", self.lidar_sensor_name, "0.pcd")
        self.pcd_paths = []
        self.image_path = os.path.join(self.lct_path, "cameras", self.rgb_sensor_name, "0.jpg")
        # We extract the first image from LCT in order to get the needed data to create our plt figure
        self.image = Image.open(self.image_path)
        self.image_w = self.image.width
        self.image_h = self.image.height
        self.image = np.asarray(self.image)
        self.fig, self.ax = plt.subplots(figsize=plt.figaspect(self.image))
        self.fig.subplots_adjust(0,0,1,1)

        # image widget used to draw an image onto our image window
        self.image_widget = gui.ImageWidget()

        self.frame_num = 0
        # dictionary that stores the imported JSON file that respresents the annotations in the current frame
        self.boxes = json.load(open(os.path.join(self.lct_path ,"bounding", str(self.frame_num), "boxes.json")))
        # num of frames available to display
        frames_available = [entry for entry in os.scandir(os.path.join(self.lct_path, "bounding"))]
        self.num_frames = len(frames_available)
        # List to store bounding boxes as NuScenes Box Objects
        self.n_boxes = []


        #Import the annotation map if it exists
        self.annotation_map = {}
        if os.path.exists(os.path.join(self.lct_path, "pred_bounding", "annotation_map.json")):
            self.annotation_map = json.load(open(os.path.join(self.lct_path, "pred_bounding", "annotation_map.json")))

        # Aliases for easier referencing
        cw = self.controls
        pw = self.pointcloud_window
        iw = self.image_window

        # Set up a vertical widget "layout" that will hold all of our horizontal widgets
        em = cw.theme.font_size
        margin = gui.Margins(0.50 * em, 0.25 * em, 0.50 * em, 0.25 * em)
        layout = gui.Vert(0, margin)

        # Create SceneWidget() object to render pointclouds and bounding boxes
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(pw.renderer)
        self.widget3d.scene.set_background([0,0,0,255])
        self.mat = rendering.MaterialRecord()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = 2
        #self.mat.base_color = [255,255,255,255]

        margin = gui.Margins(0.5 * em, 0.25 * em, 0.25 * em, 0.25 * em)
        self.view = gui.CollapsableVert("View", .25 * em, margin)
        self.scene_nav = gui.CollapsableVert("Scene Navigation", .25 * em, margin)
        self.anno_control = gui.CollapsableVert("Annotation Control", .25 * em, margin)

        # Set up drop down menu for switching between RGB sensors
        sensor_select = gui.Combobox()
        for cam in self.camera_sensors:
            sensor_select.add_item(cam)
        sensor_select.set_on_selection_changed(self.on_sensor_select)

        # Set up checkboxes for selecting ground truth annotations
        # Have to go through each frame to have all possible annotations available
        self.check_horiz = []
        color_counter = 0
        for i in range(0, self.num_frames):
            boxes = json.load(open(os.path.join(self.lct_path ,"bounding", str(i), "boxes.json")))
            for box in boxes['boxes']:
                if box['annotation'] not in self.color_map:
                    horiz = gui.Horiz()
                    check = gui.Checkbox("")
                    check.set_on_checked(self.make_on_check(box['annotation'], self.on_filter_check))
                    self.color_map[box['annotation']] = colorlist[color_counter % len(colorlist)]
                    color_counter += 1
                    # Color Picker
                    color = gui.ColorEdit()
                    (r,g,b) = self.color_map[box['annotation']]
                    color.color_value = gui.Color(r/255,g/255,b/255)
                    color.set_on_value_changed(self.on_color_toggle)
                    horiz.add_child(check)
                    horiz.add_child(gui.Label(box['annotation']))
                    horiz.add_child(color)
                    horiz.add_child(gui.Label("Count: 0"))
                    self.check_horiz.append(horiz)


        # Set up checkboxes for selecting predicted annotations
        frames_available = [entry for entry in os.scandir(os.path.join(self.lct_path, "pred_bounding"))]
        self.pred_frames = len(frames_available) - 1
        self.pred_check_horiz = []
        self.all_pred_annotations = []
        for i in range(0, self.pred_frames):
            boxes = json.load(open(os.path.join(self.lct_path ,"pred_bounding", str(i), "boxes.json")))
            for box in boxes['boxes']:
                if box['annotation'] not in self.pred_color_map:
                    self.all_pred_annotations.append(box['annotation'])
                    horiz = gui.Horiz()
                    check = gui.Checkbox("")
                    check.set_on_checked(self.make_on_check(box['annotation'], self.on_pred_filter_check))
                    self.pred_color_map[box['annotation']] = colorlist[color_counter % len(colorlist)]
                    color_counter += 1
                    color = gui.ColorEdit()
                    (r,g,b) = self.pred_color_map[box['annotation']]
                    color.color_value = gui.Color(r/255,g/255,b/255)
                    color.set_on_value_changed(self.on_color_toggle)
                    horiz.add_child(check)
                    horiz.add_child(gui.Label(box['annotation']))
                    horiz.add_child(color)
                    horiz.add_child(gui.Label("Count: 0"))
                    self.pred_check_horiz.append(horiz)
        if self.pred_frames > 0:
            self.pred_boxes = json.load(open(os.path.join(self.lct_path ,"pred_bounding", str(self.frame_num), "boxes.json")))

        # Horizontal widget where we will insert our drop down menu
        sensor_switch_layout = gui.Horiz()
        sensor_switch_layout.add_child(gui.Label("Switch RGB Sensor"))
        sensor_switch_layout.add_child(sensor_select)
        
        # Vertical widget for inserting checkboxes
        checkbox_layout = gui.CollapsableVert("Ground Truth Filters", .25 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em,
                                         0.5 * em) )
        for horiz_widget in self.check_horiz:
            checkbox_layout.add_child(horiz_widget)

        # Vertical widget for inserting predicted checkboxes
        pred_checkbox_layout = gui.CollapsableVert("Predicted Filters", .25 * em, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em,
                                         0.5 * em))
        for horiz_widget in self.pred_check_horiz:
            pred_checkbox_layout.add_child(horiz_widget)

        # Set up a widget to switch between frames
        self.frame_select = gui.NumberEdit(gui.NumberEdit.INT)
        self.frame_select.set_limits(0, self.num_frames)
        self.frame_select.set_on_value_changed(self.on_frame_switch)

        # Add a frame switching widget to another horizontal widget
        frame_switch_layout = gui.Horiz()
        frame_switch_layout.add_child(gui.Label("Switch Frame"))
        frame_switch_layout.add_child(self.frame_select)

        # Set up a widget to specify a minimum annotation confidence
        confidence_select = gui.NumberEdit(gui.NumberEdit.INT)
        confidence_select.set_limits(0,100)
        confidence_select.set_value(50)
        confidence_select.set_on_value_changed(self.on_confidence_switch)

        # Add confidence select widget to horizontal
        confidence_select_layout = gui.Horiz()
        confidence_select_layout.add_child(gui.Label("Specify Confidence Threshold"))
        confidence_select_layout.add_child(confidence_select)

        # Add combobox to switch between predicted and ground truth
        self.bounding_toggle = gui.Combobox()
        self.bounding_toggle.add_item("Ground Truth")
        self.bounding_toggle.add_item("Predicted")
        self.bounding_toggle.set_on_selection_changed(self.toggle_bounding)

        bounding_toggle_layout = gui.Horiz()
        bounding_toggle_layout.add_child(gui.Label("Toggle Predicted or GT"))
        bounding_toggle_layout.add_child(self.bounding_toggle)

        #Button to jump to Birds eye view of vehicle

        center_horiz = gui.Horiz()
        center_view_button = gui.Button("Center Pointcloud View on Vehicle")
        center_view_button.set_on_clicked(self.jump_to_vehicle)
        #center_horiz.add_child(gui.Label("Center Pointcloud View on Vehicle"))
        center_horiz.add_child(center_view_button)

        #Collapsable vertical widget that will hold comparison controls
        comparison_controls = gui.CollapsableVert("Compare Predicted Data")
        toggle_comparison = gui.Checkbox("Display Predicted and GT")
        toggle_highlight = gui.Checkbox("Show Unmatched GT Annotations")
        toggle_highlight.set_on_checked(self.toggle_highlights)
        toggle_comparison.set_on_checked(self.toggle_box_comparison)

        toggle_false_positive = gui.Checkbox("Show False Positives")
        toggle_false_positive.set_on_checked(self.toggle_false_positive)


        toggle_incorrect_annotations = gui.Checkbox("Show Incorrect Annotations")
        toggle_incorrect_annotations.set_on_checked(self.toggle_incorrect_annotations)

        toggle_score = gui.Checkbox("Show Confidence Score")
        toggle_score.set_on_checked(self.toggle_score)

        comparison_controls.add_child(toggle_comparison)
        comparison_controls.add_child(toggle_highlight)
        

        jump_frame_horiz = gui.Horiz()
        prev_button = gui.Button("Previous")
        prev_button.set_on_clicked(self.jump_prev_frame)
        next_button = gui.Button("Next")
        next_button.set_on_clicked(self.jump_next_frame)
        jump_frame_horiz.add_child(gui.Label("Search Frames for Selected GT Boxes"))
        jump_frame_horiz.add_child(prev_button)
        jump_frame_horiz.add_child(next_button)


        #comparison_controls.add_child(jump_frame_horiz)

        file_menu = gui.Menu()
        file_menu.add_item("Export Current RGB Image...", 0)
        file_menu.add_item("Export Current PointCloud...", 1)
        file_menu.add_separator()
        file_menu.add_item("Quit", 2)

        tools_menu = gui.Menu()
        tools_menu.add_item("Scan For Errors",3)
    

        menu = gui.Menu()
        menu.add_menu("File", file_menu)
        menu.add_menu("Tools", tools_menu)
        gui.Application.instance.menubar = menu



        # Add our widgets to the vertical widget

        self.view.add_child(frame_switch_layout)
        self.view.add_child(center_horiz)

        self.scene_nav.add_child(sensor_switch_layout)
        self.scene_nav.add_child(jump_frame_horiz)

        #self.anno_control.add_child(bounding_toggle_layout)
        self.anno_control.add_child(confidence_select_layout)
        self.anno_control.add_child(toggle_highlight)
        self.anno_control.add_child(toggle_false_positive)
        self.anno_control.add_child(toggle_incorrect_annotations)
        self.anno_control.add_child(toggle_score)
        self.anno_control.add_child(checkbox_layout)
        self.anno_control.add_child(pred_checkbox_layout)

        #layout.add_child(sensor_switch_layout)
        #layout.add_child(frame_switch_layout)
        #layout.add_child(bounding_toggle_layout)
        #layout.add_child(confidence_select_layout)
        #layout.add_child(center_horiz)
        #layout.add_child(comparison_controls)
        #layout.add_child(checkbox_layout)
        #layout.add_child(pred_checkbox_layout)

        layout.add_child(self.view)
        layout.add_child(self.scene_nav)
        layout.add_child(self.anno_control)

        # Add the master widgets to our three windows
        cw.add_child(layout)
        pw.add_child(self.widget3d)        
        iw.add_child(self.image_widget)
        
        cw.set_on_menu_item_activated(0, self.on_menu_export_rgb)
        cw.set_on_menu_item_activated(1, self.on_menu_export_lidar)
        cw.set_on_menu_item_activated(2, self.on_menu_quit)
        cw.set_on_menu_item_activated(3, self.on_error_scan)


        iw.set_on_menu_item_activated(0, self.on_menu_export_rgb)
        iw.set_on_menu_item_activated(1, self.on_menu_export_lidar)
        iw.set_on_menu_item_activated(2, self.on_menu_quit)
        cw.set_on_menu_item_activated(3, self.on_error_scan)


        pw.set_on_menu_item_activated(0, self.on_menu_export_rgb)
        pw.set_on_menu_item_activated(1, self.on_menu_export_lidar)
        pw.set_on_menu_item_activated(2, self.on_menu_quit)
        cw.set_on_menu_item_activated(3, self.on_error_scan)
    

        # Call update function to draw all initial data
        self.update()

        # This 'bounds' variable has nothing to do with the bounding boxes, it represents the box surrounding
        # all of our lidar points and is used to set up the camera for the scene
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(10, bounds, self.frame_extrinsic['translation'])
        eye = [0,0,0]
        eye[0] = self.frame_extrinsic['translation'][0]
        eye[1] = self.frame_extrinsic['translation'][1]
        eye[2] = 150.0
        self.widget3d.scene.camera.look_at(self.frame_extrinsic['translation'], eye, [1, 0, 0])

    def update_image(self):
        """Fetches new image from LVT Directory, and draws it onto a plt figure
           Uses nuScenes API to project 3D bounding boxes onto that plt figure
           Finally, extracts raw image data from plt figure and updates our image widget
            Args:
                self: window object
            Returns:
                None
                """
        # Extract new image from file
        self.image = np.asarray(Image.open(self.image_path))
        for b in self.boxes_to_render:
            box = Box(b[ORIGIN], b[SIZE], Quaternion(b[ROTATION]), name=b[ANNOTATION], score=b[CONFIDENCE], velocity=(0,0,0))
            color = b[COLOR]

            # Box is stored in vehicle frame, so transform it to RGB sensor frame
            box.translate(-np.array(self.image_extrinsic['translation']))
            box.rotate(Quaternion(self.image_extrinsic['rotation']).inverse)
            
            #Thank you to Oscar Beijbom for providing this box rendering algorithm at https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py
            if box_in_image(box, np.asarray(self.image_intrinsic['matrix']), (self.image_w, self.image_h), BoxVisibility.ANY):
                # If the box is in view, then render it onto the PLT frame
                corners = view_points(box.corners(), np.asarray(self.image_intrinsic['matrix']), normalize=True)[:2, :]

                def draw_rect(selected_corners, c):
                    prev = selected_corners[-1]
                    for corner in selected_corners:
                        cv2.line(self.image,
                                (int(prev[0]), int(prev[1])),
                                (int(corner[0]), int(corner[1])),
                                c, 2)
                        prev = corner

                # Draw the sides
                for i in range(4):
                    cv2.line(self.image,
                            (int(corners.T[i][0]), int(corners.T[i][1])),
                            (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                            color, 2)

                # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
                draw_rect(corners.T[:4], color)
                draw_rect(corners.T[4:], color)

                # Draw line indicating the front
                center_bottom_forward = np.mean(corners.T[2:4], axis=0)
                center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
                cv2.line(self.image,
                        (int(center_bottom[0]), int(center_bottom[1])),
                        (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
                        color, 2)

                #Only render confidence if this isnt at GT box        
                if b[CONFIDENCE] < 100 and self.show_score:
                    cv2.putText(self.image, str(b[CONFIDENCE]), (int(corners.T[0][0]), int(corners.T[1][1])), cv2.FONT_HERSHEY_SIMPLEX ,1, (255,0,0), 2)

        new_image = o3d.geometry.Image(self.image)
        self.image_widget.update_image(new_image)

        # Force image widget to redraw
        #Post Redraw calls seem to crash the app on windows. Temporary workaround
        if OS_STRING != "Windows":
            self.image_window.post_redraw()

    def update_bounding(self):
        """Updates bounding box information when switching frames
            Args:
                self: window object
            Returns:
                None
                """

        #Array that will hold list of boxes that will eventually be rendered
        self.boxes_to_render = []
        #
        self.boxes = json.load(open(os.path.join(self.lct_path , "bounding", str(self.frame_num), "boxes.json")))
        #Update the counters for the gt boxes
        for horiz_widget in self.check_horiz:
            children = horiz_widget.get_children()
            label_widget = children[1]
            color_widget = children[2]
            count_widget = children[3]
            self.color_map[label_widget.text] = (int(color_widget.color_value.red * 255), int(color_widget.color_value.green * 255), int(color_widget.color_value.blue * 255))
            count_num = 0
            for box in self.boxes['boxes']:
                if box['annotation'] == label_widget.text:
                    count_num += 1
            count_widget.text = "Count: " + str(count_num)
        
        if self.pred_frames > 0:
            self.pred_boxes = json.load(open(os.path.join(self.lct_path ,"pred_bounding", str(self.frame_num), "boxes.json")))
            #Update the counters for predicted boxes
            for horiz_widget in self.pred_check_horiz:
                children = horiz_widget.get_children()
                label_widget = children[1]
                color_widget = children[2]
                count_widget = children[3]
                self.pred_color_map[label_widget.text] = (int(color_widget.color_value.red * 255), int(color_widget.color_value.green * 255), int(color_widget.color_value.blue * 255))
                count_num = 0
                for box in self.pred_boxes['boxes']:
                    if box['annotation'] == label_widget.text and box['confidence'] >= self.min_confidence:
                        count_num += 1
                count_widget.text = "Count: " + str(count_num)


        #If highlight_faults is False, then we just filter boxes
        if self.highlight_faults is False and self.show_false_positive is False and self.show_incorrect_annotations is False:
            #Add GT Boxes we should render
            for box in self.boxes['boxes']:
                if ((len(self.filter_arr) == 0 and len(self.pred_filter_arr) == 0) or box['annotation'] in self.filter_arr) and box['confidence'] >= self.min_confidence:
                    bounding_box = [box['origin'], box['size'], box['rotation'], box['annotation'], box['confidence'], self.color_map[box['annotation']]]
                    if len(self.filter_arr) == 0 or bounding_box[ANNOTATION] in self.filter_arr:
                        self.boxes_to_render.append(bounding_box)
            #Add Pred Boxes we should render
            if self.pred_frames > 0:
                for box in self.pred_boxes['boxes']:
                    if ((len(self.pred_filter_arr) == 0 and len(self.filter_arr) == 0) or box['annotation'] in self.pred_filter_arr) and box['confidence'] >= self.min_confidence:
                        bounding_box = [box['origin'], box['size'], box['rotation'], box['annotation'], box['confidence'], self.pred_color_map[box['annotation']]]
                        if len(self.pred_filter_arr) == 0 or bounding_box[ANNOTATION] in self.pred_filter_arr:
                            self.boxes_to_render.append(bounding_box)
        
        #Otherwise, the user is trying to highlight faults, so the selected annotations define an equivalancy between annotations
        if self.show_false_positive or self.highlight_faults:
            #For each gt box, only render it if it overlaps with a predicted box
            min_dist = .5 #minimum distance should be half a meter
            #Reverse Sort predicted boxes based on confidence
            sorted_list = sorted(self.pred_boxes['boxes'], key=itemgetter('confidence'), reverse=True)
            sorted_list = [box for box in sorted_list if box['annotation'] in self.pred_filter_arr and box['confidence'] >= self.min_confidence]
            pred_matched = [False] * len(sorted_list)
            
            
            #Get rid of GT boxes that have do not match the annotation
            gt_list = [box for box in self.boxes['boxes'] if box['annotation'] in self.filter_arr]
            gt_matched = [False] * len(gt_list)

            #match each predicted box to a gt box
            for (pred_idx,pred_box) in enumerate(sorted_list):
                dist = float('inf')
                for (i, gt_box) in enumerate(gt_list):
                    temp_dist = geometry_utils.box_dist(pred_box, gt_box)
                    if not gt_matched[i]:
                        if temp_dist < dist:
                            dist = temp_dist
                            match_index = i
                #We found a valid match
                if dist <= min_dist:
                    #TODO Assume that the closet annotation is the correct one? And compare the name of the annotation
                    gt_matched[match_index] = True
                    pred_matched[pred_idx] = True

            #Add false positive predicted boxes to render list
            if self.show_false_positive:
                for (i, box) in enumerate(sorted_list):
                    if pred_matched[i] == False:
                        self.boxes_to_render.append([box['origin'], box['size'], box['rotation'], box['annotation'], box['confidence'], self.pred_color_map[box['annotation']]])

            #Add unmatched gt boxes to render list
            if self.highlight_faults:
                for (i, box) in enumerate(gt_list):
                    if gt_matched[i] == False:
                        self.boxes_to_render.append([box['origin'], box['size'], box['rotation'], box['annotation'], box['confidence'], self.color_map[box['annotation']]])
        
        #The user is trying to see if any GT boxes were categorized by mistake
        if self.show_incorrect_annotations:
            gt_list = [box for box in self.boxes['boxes'] if box['annotation'] in self.filter_arr]
            pred_list = [box for box in self.pred_boxes['boxes'] if box['confidence'] >= self.min_confidence]
            min_dist = .5
            for pred_box in pred_list:
                for gt_box in gt_list:
                    dist = geometry_utils.box_dist(pred_box, gt_box)
                    #if the box is within the distance cuttoff, but has the wrong annotation, we render both the gt box and predicted box
                    if dist <= min_dist and pred_box['annotation'] not in self.pred_filter_arr:
                        self.boxes_to_render.append([pred_box['origin'], pred_box['size'], pred_box['rotation'], pred_box['annotation'], pred_box['confidence'], self.pred_color_map[pred_box['annotation']]])
                        self.boxes_to_render.append([gt_box['origin'], gt_box['size'], gt_box['rotation'], gt_box['annotation'], gt_box['confidence'], self.color_map[gt_box['annotation']]])
        #Post Redraw calls seem to crash the app on windows. Temporary workaround
        if OS_STRING != "Windows":
            self.controls.post_redraw()

    def update_pointcloud(self):
        """Takes new pointcloud data and converts it to global frame, 
           then renders the bounding boxes (Assuming the boxes are already in global frame)
            Args:
                self: window object
            Returns:
                None
                """
        self.widget3d.scene.clear_geometry()
        # Add Pointcloud
        temp_points = np.empty((0,3))
        for label in self.label_list:
            self.widget3d.remove_3d_label(label)

        self.label_list = []

        for i, pcd_path in enumerate(self.pcd_paths):
            temp_cloud = o3d.io.read_point_cloud(pcd_path)
            # sensor_rotation_matrix = R.from_quat(self.pcd_extrinsic[sensor]['rotation']).as_matrix()
            ego_rotation_matrix = Quaternion(self.frame_extrinsic['rotation']).rotation_matrix

            # Transform lidar points into global frame
            temp_cloud.rotate(ego_rotation_matrix, [0,0,0])
            temp_cloud.translate(self.frame_extrinsic['translation'])
            temp_points = np.concatenate((temp_points, np.asarray(temp_cloud.points)))
 
        self.pointcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(temp_points)))
        # Add new global frame pointcloud to our 3D widget
        self.widget3d.scene.add_geometry("Point Cloud", self.pointcloud, self.mat)
        
        i = 0
        mat = rendering.MaterialRecord()
        mat.shader = "unlitLine"
        mat.line_width = .25
        for box in self.boxes_to_render:
            size = [0,0,0]
            # We have to do this because open3D mixes up the length and the width of the boxes, however the height is still the third element
            # in other words nuscenes stores box data in [L,W,H] but open3d expects [W,L,H]
            size[0] = box[SIZE][1]
            size[1] = box[SIZE][0]
            size[2] = box[SIZE][2]
            color = box[COLOR]
            bounding_box = o3d.geometry.OrientedBoundingBox(box[ORIGIN], Quaternion(box[ROTATION]).rotation_matrix, size)
            bounding_box.rotate(Quaternion(self.frame_extrinsic['rotation']).rotation_matrix, [0,0,0])
            bounding_box.translate(self.frame_extrinsic['translation'])
            hex = '#%02x%02x%02x' % color # bounding_box.color needs to be a tuple of floats (color is a tuple of ints)
            bounding_box.color = matplotlib.colors.to_rgb(hex)

            if box[CONFIDENCE] < 100 and self.show_score:
                label = self.widget3d.add_3d_label(bounding_box.center, str(box[CONFIDENCE]))
                label.color = gui.Color(1.0,0.0,0.0)
                self.label_list.append(label)

            self.widget3d.scene.add_geometry(box[ANNOTATION] + str(i), bounding_box, mat)
            i += 1
        


        #Add Line that indicates current RGB Camera View
        line = o3d.geometry.LineSet()
        line.points = o3d.utility.Vector3dVector([[0,0,0], [0,0,2]])
        line.lines =  o3d.utility.Vector2iVector([[0,1]])
        line.colors = o3d.utility.Vector3dVector([[1.0,0,0]])

        
        line.rotate(Quaternion(self.image_extrinsic['rotation']).rotation_matrix, [0,0,0])
        line.translate(self.image_extrinsic['translation'])
        
        
        line.rotate(Quaternion(self.frame_extrinsic['rotation']).rotation_matrix, [0,0,0])
        line.translate(self.frame_extrinsic['translation'])
        

        self.widget3d.scene.add_geometry("RGB Line",line, mat)

        
        # Force our widgets to update
        self.widget3d.force_redraw()
        #Post Redraw calls seem to crash the app on windows. Temporary workaround
        if OS_STRING != "Windows":
            self.pointcloud_window.post_redraw()
    
    def update_poses(self):
        """Extracts all the pose data when switching sensors, and or frames
            Args:
                self: window object
            Returns:
                None
                """
        # Pulling intrinsic and extrinsic data from LVT directory based on current selected frame and sensor       
        self.image_intrinsic = json.load(open(os.path.join(self.lct_path, "cameras", self.rgb_sensor_name, "intrinsics.json")))
        self.image_extrinsic = json.load(open(os.path.join(self.lct_path, "cameras" , self.rgb_sensor_name, "extrinsics.json")))
        self.frame_extrinsic = json.load(open(os.path.join(self.lct_path, "ego", str(self.frame_num) + ".json")))
        self.pcd_extrinsic = {}
        # iterates over pointcloud paths that are currently stored
        for sensor_idx, path in enumerate(self.pcd_paths):
            fp = open(os.path.join(path))
            for i, line in enumerate(fp):
                if i == 8:
                    # setting translation and rotation arrays based on new pointcloud
                    vals = line.split()
                    self.pcd_extrinsic[self.lidar_sensors[sensor_idx]] = {}
                    self.pcd_extrinsic[self.lidar_sensors[sensor_idx]]['translation'] = [float(vals[1]), float(vals[2]), float(vals[3])]
                    self.pcd_extrinsic[self.lidar_sensors[sensor_idx]]['rotation'] = [float(vals[4]), float(vals[5]), float(vals[6]), float(vals[7])]
            fp.close()

    def on_sensor_select(self, new_val, new_idx):
        """This updates the name of the selected rgb sensor after user input
           Updates the window with the new information 
            Args:
                self: window object
                new_val: new name of the rgb sensor
                new_inx:
            Returns:
                None
                """
        self.rgb_sensor_name = new_val
        self.update()

    # This creates a new function for every annotation value, so the annotation name can be 
    # passed through
    def make_on_check(self, annotation, func):
        """This creates a function for every annotation value
           This way the annotation name can be passed through 
            Args:
                self: window object
                annotation: name of annotation
                func: name of function
            Returns:
                a fucntion calling the func argument
                """
        def on_checked(checked):
            func(annotation, checked)
        return on_checked

    
    def on_filter_check(self, annotation, checked):
        """This updates the filter_arr (array of annotations to display) based on new user input
           If the user checked it, it will add the annotation type
           If the user unchecked it, it will remove the annotation type
           Updates the window object after changes are made 
            Args:
                self: window object
                annotation: type of annotation that is being modified
                checked: value of checkbox, true if checked and false if unchecked
            Returns:
                None
                """
        if checked:
            self.filter_arr.append(annotation)
        else:
            self.filter_arr.remove(annotation)
        self.update()

    def on_pred_filter_check(self, annotation, checked):
        """This updates the pred_filter (array of predicted annotations to display) based on new user input
           If the user checked it, it will add the annotation type
           If the user unchecked it, it will remove the annotation type
           Updates the window object after changes are made 
            Args:
                self: window object
                annotation: type of annotation that is being modified
                checked: value of checkbox, true if checked and false if unchecked
            Returns:
                None
                """
        if checked:
            self.pred_filter_arr.append(annotation)
        else:
            self.pred_filter_arr.remove(annotation)
        self.update()

    def update_image_path(self):
        """This updates the image path based on current rgb sensor name and frame number
            Args:
                self: window object
            Returns:
                None
                """
        self.image_path = os.path.join(self.lct_path, "cameras", self.rgb_sensor_name, str(self.frame_num) +".jpg")
    
    def update_pcd_path(self):
        """This clears the current pcd_paths stored and updates it with the sensors currently stored in lidar_sensors
            Args:
                self: window object
            Returns:
                None
                """
        self.pcd_paths.clear()
        for sensor in self.lidar_sensors:
            self.pcd_paths.append(os.path.join(self.lct_path, "pointcloud", sensor, str(self.frame_num) + ".pcd"))

    
    def on_frame_switch(self, new_val):
        """This updates the frame number of the window based on user input and then updates the window
           Validates that the new frame number is valid (within range of frame nums) 
            Args:
                self: window object
                new_val: new fram number
            Returns:
                None
                """
        if int(new_val) >= 0 and int(new_val) < self.num_frames:
            # Set new frame value
            self.frame_num = int(new_val)
            # Update Bounding Box List
            self.update()

    def on_confidence_switch(self, new_val):
        """This updates the minimum confidence after the user changed it.
           New value must be between 0 and 100 inclusive 
           Updates the window afterwards
            Args:
                self: window object
                new_val: new value of min confidence
            Returns:
                None
                """
        if int(new_val) >= 0 and int(new_val) <= 100:
            self.min_confidence = int(new_val)
            self.update()

    def on_menu_quit(self):
        gui.Application.instance.quit()

    def on_color_toggle(self, new_color):
        self.update()

    def toggle_bounding(self, new_val, new_idx):
        """This updates the bounding box on the window to reflect either bounding or predicted bounding
           Then updates the window to reflect changes 
            Args:
                self: window object
                new_val: the new value of the box
                new_idx: 
            Returns:
                None
                """
        # switch to predicted boxes
        if self.box_data_name != ["pred_bounding", "bounding"]:
            if new_val == "Predicted" and self.pred_frames > 0:
                self.box_data_name = ["pred_bounding"]
                self.update()
            else: # switched to ground truth boxes
                self.box_data_name = ["bounding"]
                self.update()
    
    def toggle_box_comparison(self, checked):
        if self.pred_frames > 0:
            if checked:
                self.box_data_name = ["pred_bounding", "bounding"]
                self.compare_bounding = True
            else:
                if self.bounding_toggle.selected_text == "Ground Truth":
                    self.box_data_name = ["bounding"]
                else:
                    self.box_data_name = ["pred_bounding"]
                self.compare_bounding = False
            self.update()

    def toggle_highlights(self, checked):
        if self.pred_frames > 0:
            if checked:
                self.highlight_faults = True
            else:
                self.highlight_faults = False
            self.update()
    
    def toggle_false_positive(self, checked):
        if self.pred_frames > 0:
            if checked:
                self.show_false_positive = True
            else:
                self.show_false_positive = False
            self.update()
    def toggle_incorrect_annotations(self, checked):
        if self.pred_frames > 0:
            if checked:
                self.show_incorrect_annotations = True
            else:
                self.show_incorrect_annotations = False
            self.update()

    def toggle_score(self, checked):
        if checked:
            self.show_score = True
        else:
            self.show_score = False
        self.update()
    def jump_next_frame(self):
        found = False
        current_frame = self.frame_num
        while not found:
            #If the user has not selected any ground truth boxes, then dont try to search anything
            if len(self.filter_arr) == 0:
                return
            current_frame = (current_frame + 1) % self.num_frames
            current_box_list = json.load(open(os.path.join(self.lct_path , "bounding", str(current_frame), "boxes.json")))
            for box in current_box_list['boxes']:
                if box['annotation'] in self.filter_arr:
                    found = True
                    self.frame_num = current_frame
                    break
        self.frame_select.set_value(current_frame)
        self.update()


    def jump_prev_frame(self):
        found = False
        current_frame = self.frame_num
        
        while not found:
            #If the user has not selected any ground truth boxes, then dont try to search anything
            if len(self.filter_arr) == 0:
                return
            current_frame = (current_frame - 1) % self.num_frames
            current_box_list = json.load(open(os.path.join(self.lct_path , "bounding", str(current_frame), "boxes.json")))
            for box in current_box_list['boxes']:
                if box['annotation'] in self.filter_arr:
                    found = True
                    self.frame_num = current_frame
                    break
        self.frame_select.set_value(current_frame)
        self.update()
    
    def jump_to_vehicle(self):
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(10, bounds, self.frame_extrinsic['translation'])
        eye = [0,0,0]
        eye[0] = self.frame_extrinsic['translation'][0]
        eye[1] = self.frame_extrinsic['translation'][1]
        eye[2] = 150.0
        self.widget3d.scene.camera.look_at(self.frame_extrinsic['translation'], eye, [1, 0, 0])
        self.update()
    
    def on_menu_export_rgb(self):
        file_dialog = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save", self.controls.theme)
        file_dialog.add_filter(".png", "PNG files (.png)")
        file_dialog.set_on_cancel(self.on_file_dialog_cancel)
        file_dialog.set_on_done(self.on_export_rgb_dialog_done)
        self.controls.show_dialog(file_dialog)
    
    def on_menu_export_lidar(self):
        file_dialog = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save", self.controls.theme)
        file_dialog.add_filter(".png", "PNG files (.png)")
        file_dialog.set_on_cancel(self.on_file_dialog_cancel)
        file_dialog.set_on_done(self.on_export_lidar_dialog_done)
        self.controls.show_dialog(file_dialog)

    def on_file_dialog_cancel(self):
        self.controls.close_dialog()
    
    def on_export_rgb_dialog_done(self, filename):
        self.controls.close_dialog()
        image = Image.fromarray(self.image)
        image.save(filename)
        self.update()

    def on_export_lidar_dialog_done(self, filename):
        self.controls.close_dialog()
        def on_image(image):
            img = image
            o3d.io.write_image(filename, img, 9)
        self.widget3d.scene.scene.render_to_image(on_image)
        self.update()

    

    def on_error_scan(self):

        window = gui.Application.instance.create_window("Errors", 400, 800)

        

        em = self.controls.theme.font_size
        margin = gui.Margins(0.50 * em, 0.25 * em, 0.50 * em, 0.25 * em)
        layout = gui.Vert(0, margin)
        
        error_count = 0

        for j in range(0, self.num_frames):
            boxes = json.load(open(os.path.join(self.lct_path , "bounding", str(j), "boxes.json")))
            pred_boxes = json.load(open(os.path.join(self.lct_path , "pred_bounding", str(j), "boxes.json")))

            unmatched_map = {}
            false_positive_map = {}
            incorrect_annotation_map = {}
            
            ####################

            if self.show_false_positive or self.highlight_faults:
                #For each gt box, only render it if it overlaps with a predicted box
                min_dist = .5 #minimum distance should be half a meter
                #Reverse Sort predicted boxes based on confidence
                sorted_list = sorted(pred_boxes['boxes'], key=itemgetter('confidence'), reverse=True)
                sorted_list = [box for box in sorted_list if box['annotation'] in self.pred_filter_arr and box['confidence'] >= self.min_confidence]
                pred_matched = [False] * len(sorted_list)
                
                
                #Get rid of GT boxes that have do not match the annotation
                gt_list = [box for box in boxes['boxes'] if box['annotation'] in self.filter_arr]
                gt_matched = [False] * len(gt_list)

                #match each predicted box to a gt box
                for (pred_idx,pred_box) in enumerate(sorted_list):
                    dist = float('inf')
                    for (i, gt_box) in enumerate(gt_list):
                        temp_dist = geometry_utils.box_dist(pred_box, gt_box)
                        if not gt_matched[i]:
                            if temp_dist < dist:
                                dist = temp_dist
                                match_index = i
                    #We found a valid match
                    if dist <= min_dist:
                        gt_matched[match_index] = True
                        pred_matched[pred_idx] = True

                #Add false positive predicted boxes to render list
                if self.show_false_positive:
                    for (i, box) in enumerate(sorted_list):
                        if pred_matched[i] == False:
                            false_positive_map[box['annotation']] = false_positive_map.get(box['annotation'], 0) + 1
                #Add unmatched gt boxes to render list
                if self.highlight_faults:
                    for (i, box) in enumerate(gt_list):
                        if gt_matched[i] == False:
                            unmatched_map[box['annotation']] = unmatched_map.get(box['annotation'], 0) + 1
        
            #The user is trying to see if any GT boxes were categorized by mistake
            if self.show_incorrect_annotations:
                gt_list = [box for box in boxes['boxes'] if box['annotation'] in self.filter_arr]
                pred_list = [box for box in pred_boxes['boxes'] if box['confidence'] >= self.min_confidence]
                min_dist = .5
                for pred_box in pred_list:
                    for gt_box in gt_list:
                        dist = geometry_utils.box_dist(pred_box, gt_box)
                        #if the box is within the distance cuttoff, but has the wrong annotation, we render both the gt box and predicted box
                        if dist <= min_dist and pred_box['annotation'] not in self.pred_filter_arr:
                            incorrect_annotation_map[gt_box['annotation']] = incorrect_annotation_map.get(gt_box['annotation'], 0) + 1

            ####################




            #If we found any specified incorrect annotations then we create a collapsable widget for this frame
            if unmatched_map or false_positive_map or incorrect_annotation_map:
                frame_vert = gui.CollapsableVert("Frame " + str(j), .25 * em, margin)
                frame_vert.set_is_open(False)
                for key in unmatched_map.keys():
                    message = gui.Label(str(unmatched_map[key]) + " Unmatched Boxes for GT Label: " + str(key))
                    frame_vert.add_child(message)

                for key in false_positive_map.keys():
                    message = gui.Label(str(false_positive_map[key]) + " False Positives for Pred Label: " + str(key))
                    frame_vert.add_child(message)

                for key in incorrect_annotation_map.keys():
                    message = gui.Label(str(incorrect_annotation_map[key]) + " Incorrect Annotations for GT Label: " + str(key))
                    frame_vert.add_child(message)
                error_count += 1
                layout.add_child(frame_vert)
            
                

        if error_count == 0:
            message = gui.Label("No Errors Found")
            layout.add_child(message)
        window.add_child(layout)
      


    def close_dialog(self):
        self.controls.close_dialog()
    
    def update(self):
        """ This updates the window object to reflect the current state
        Args:
            self: window object
        Returns:
            None
            """
        self.update_image_path()
        self.update_pcd_path()
        self.update_poses()
        self.update_bounding()
        self.update_image()
        self.update_pointcloud()

    def get_cams_and_pointclouds(self, path):
        """This gets the names of the cameras and lidar sensors
            Args:
                self: window object
                path: path to the LVT directory
            Returns:
                a tuple of lists [camera sensors (RGB), lidar sensors (Pointcloud)]
                """
        
        camera_sensors = []
        lidar_sensors = []

        # Adds cameras to the GUI
        for camera_name in os.listdir(os.path.join(path, "cameras")):
            camera_sensors.append(camera_name)

        # Adds lidar to the GUI
        for lidar_name in os.listdir(os.path.join(path, "pointcloud")):
            lidar_sensors.append(lidar_name)

        return (camera_sensors, lidar_sensors)


if __name__ == "__main__":
    lct_dir = parse_options()
    gui.Application.instance.initialize()
    w = Window(lct_dir)
    o3d.visualization.gui.Application.instance.run()
    
    