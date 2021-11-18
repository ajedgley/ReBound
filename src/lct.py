"""
lct.py

Conversion tool to bring nuscenes dataset into LVT. 
"""

import getopt
import matplotlib.colors
import matplotlib.pyplot as plt
import time
import queue
from enum import Enum
from PIL import Image
from random import randint
import sys
import io
import numpy as np
from numpy import core
from numpy.core.numeric import normalize_axis_tuple
import open3d.visualization.gui as gui
from PIL import Image
import open3d as o3d
import open3d.visualization.rendering as rendering
import json
import os
import cv2
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility, transform_matrix
from scipy.spatial.transform import Rotation as R

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
        self.controls = gui.Application.instance.create_window("LCT", 400, 400)
        self.pointcloud_window = gui.Application.instance.create_window("PointCloud", 640, 480)
        self.image_window = gui.Application.instance.create_window("Image", 640, 480)

        # Set starting values for variables related to data paths
        # In the future, these values should not be set hard coded directly. Sensible default values should
        # be extracted from the LVT Directory
        self.lct_path = lct_dir
        self.camera_sensors, self.lidar_sensors = self.get_cams_and_pointclouds(self.lct_path)
        self.box_data_name = "bounding"
        self.min_confidence = 80

        # These three values represent the current LiDAR sensors, RGB sensors, and annotations being displayed
        self.rgb_sensor_name = self.camera_sensors[0]
        self.lidar_sensor_name = self.lidar_sensors[0]
        self.filter_arr = []
        
        self.color_map = {}
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

        # Aliases for easier referencing
        cw = self.controls
        pw = self.pointcloud_window
        iw = self.image_window

        # Set up a vertical widget "layout" that will hold all of our horizontal widgets
        em = cw.theme.font_size
        layout = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em,
                                         0.5 * em))

        # Create SceneWidget() object to render pointclouds and bounding boxes
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(pw.renderer)
        self.mat = rendering.Material()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = 3 * pw.scaling

        # Set up drop down menu for switching between RGB sensors
        sensor_select = gui.Combobox()
        for cam in self.camera_sensors:
            sensor_select.add_item(cam)
        sensor_select.set_on_selection_changed(self.on_sensor_select)

        # Set up checkboxes for selecting ground truth annotations
        # Have to go through each frame to have all possible annotations available
        check_boxes = []
        for i in range(0, self.num_frames):
            boxes = json.load(open(os.path.join(self.lct_path ,"bounding", str(i), "boxes.json")))
            for annotation in boxes['annotations']:
                if annotation not in self.color_map:
                    horiz = gui.Horiz()
                    check = gui.Checkbox(annotation)
                    check.set_on_checked(self.make_on_check(annotation, self.on_filter_check))
                    self.color_map[annotation] = (randint(0, 255), randint(0, 255), randint(0, 255))

                    # Color Picker
                    color = gui.ColorEdit()
                    (r,g,b) = self.color_map[annotation]
                    color.color_value = gui.Color(r/255,g/255,b/255)
                    horiz.add_child(check)
                    horiz.add_child(color)
                    check_boxes.append(horiz)


        # Set up checkboxes for selecting predicted annotations
        frames_available = [entry for entry in os.scandir(os.path.join(self.lct_path, "pred_bounding"))]
        self.pred_frames = len(frames_available)
        pred_check_boxes = []
        for i in range(0, self.pred_frames):
            boxes = json.load(open(os.path.join(self.lct_path ,"pred_bounding", str(i), "boxes.json")))
            for annotation in boxes['annotations']:
                if annotation not in self.color_map:
                    horiz = gui.Horiz()
                    check = gui.Checkbox(annotation)
                    check.set_on_checked(self.make_on_check(annotation, self.on_filter_check))
                    self.color_map[annotation] = (randint(0, 255), randint(0, 255), randint(0, 255))
                    color = gui.ColorEdit()
                    (r,g,b) = self.color_map[annotation]
                    color.color_value = gui.Color(r/255,g/255,b/255)
                    horiz.add_child(check)
                    horiz.add_child(color)
                    pred_check_boxes.append(horiz)
        
        # Horizontal widget where we will insert our drop down menu
        sensor_switch_layout = gui.Horiz()
        sensor_switch_layout.add_child(gui.Label("Switch RGB Sensor"))
        sensor_switch_layout.add_child(sensor_select)

        # Vertical widget for inserting checkboxes
        checkbox_layout = gui.Vert()
        checkbox_layout.add_child(gui.Label("Filter GT annotations"))
        for box in check_boxes:
            checkbox_layout.add_child(box)

        # Vertical widget for inserting predicted checkboxes
        pred_checkbox_layout = gui.Vert()
        pred_checkbox_layout.add_child(gui.Label("Filter predicted annotations"))
        for box in pred_check_boxes:
            pred_checkbox_layout.add_child(box)
       
        # Set up a widget to switch between frames
        frame_select = gui.NumberEdit(gui.NumberEdit.INT)
        frame_select.set_limits(0, self.num_frames)
        frame_select.set_on_value_changed(self.on_frame_switch)

        # Add a frame switching widget to another horizontal widget
        frame_switch_layout = gui.Horiz()
        frame_switch_layout.add_child(gui.Label("Switch Frame"))
        frame_switch_layout.add_child(frame_select)

        # Set up a widget to specify a minimum annotation confidence
        confidence_select = gui.NumberEdit(gui.NumberEdit.INT)
        confidence_select.set_limits(0,100)
        confidence_select.set_value(80)
        confidence_select.set_on_value_changed(self.on_confidence_switch)

        # Add confidence select widget to horizontal
        confidence_select_layout = gui.Horiz()
        confidence_select_layout.add_child(gui.Label("Specify Confidence Threshold"))
        confidence_select_layout.add_child(confidence_select)

        # Add combobox to switch between predicted and ground truth
        bounding_toggle = gui.Combobox()
        bounding_toggle.add_item("Ground Truth")
        bounding_toggle.add_item("Predicted")
        bounding_toggle.set_on_selection_changed(self.toggle_bounding)

        bounding_toggle_layout = gui.Horiz()
        bounding_toggle_layout.add_child(gui.Label("Toggle Predicted or GT"))
        bounding_toggle_layout.add_child(bounding_toggle)

        # Add our widgets to the vertical widget
        layout.add_child(sensor_switch_layout)
        layout.add_child(frame_switch_layout)
        layout.add_child(bounding_toggle_layout)
        layout.add_child(confidence_select_layout)
        layout.add_child(checkbox_layout)
        layout.add_child(pred_checkbox_layout)

        # Add the master widgets to our three windows
        cw.add_child(layout)
        pw.add_child(self.widget3d)        
        iw.add_child(self.image_widget)

        # Call update function to draw all initial data
        self.update()

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

        # Set the image width and height   
        # Figure out which bounding boxes are in our frame
        for i in range(0, len(self.n_boxes)):
            # Make sure annotation matches the filter
            if (len(self.filter_arr) == 0 or self.boxes['annotations'][i] in self.filter_arr) and self.boxes['confidences'][i] >= self.min_confidence:
                box = self.n_boxes[i]
                color = self.color_map[self.boxes['annotations'][i]]

                # Box is stored in vehicle frame, so transform it to RGB sensor frame
                box.translate(-np.array(self.image_extrinsic['translation']))
                box.rotate(Quaternion(self.image_extrinsic['rotation']).inverse)
                
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

        new_image = o3d.geometry.Image(self.image)
        self.image_widget.update_image(new_image)

        # Force image widget to redraw
        self.image_window.post_redraw()

    def update_bounding(self):
        """Updates bounding box information when switching frames
            Args:
                self: window object
            Returns:
                None
                """
        self.boxes = json.load(open(os.path.join(self.lct_path ,self.box_data_name, str(self.frame_num), "boxes.json")))

        # If there are no bounding data, nothing is displayed
        if self.boxes['origins']:
            print(self.boxes['origins'][0])
            self.n_boxes = []
            for i in range(0, len(self.boxes['origins'])):
                self.n_boxes.append(Box(self.boxes['origins'][i], self.boxes['sizes'][i], Quaternion(self.boxes['rotations'][i]), name=self.boxes['annotations'][i], score=self.boxes['confidences'][i], velocity=(0,0,0)))

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
        
        # This 'bounds' variable has nothing to do with the bounding boxes, it represents the box surrounding
        # all of our lidar points and is used to set up the camera for the scene
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(10, bounds, self.frame_extrinsic['translation'])
        eye = [0,0,0]
        eye[0] = self.frame_extrinsic['translation'][0]
        eye[1] = self.frame_extrinsic['translation'][1]
        eye[2] = 150.0
        self.widget3d.scene.camera.look_at(self.frame_extrinsic['translation'], eye, [1, 0, 0])
        
        
        # Go through each box and render it onto our 3D Widget
        # Only render box if it matches the filtering criteria
        for i in range(0, len(self.boxes['origins'])):
            if (len(self.filter_arr) == 0 or self.boxes['annotations'][i] in self.filter_arr) and self.boxes['confidences'][i] >= self.min_confidence:
                size = [0,0,0]
                # We have to do this because open3D mixes up the length and the width of the boxes, however the height is still the third element
                # in other words nuscenes stores box data in [L,W,H] but open3d expects [W,L,H]
                size[0] = self.boxes['sizes'][i][1]
                size[1] = self.boxes['sizes'][i][0]
                size[2] = self.boxes['sizes'][i][2]

                color = self.color_map[self.boxes['annotations'][i]]
                bounding_box = o3d.geometry.OrientedBoundingBox(self.boxes['origins'][i], Quaternion(self.boxes['rotations'][i]).rotation_matrix, size)
                bounding_box.rotate(Quaternion(self.frame_extrinsic['rotation']).rotation_matrix, [0,0,0])
                bounding_box.translate(self.frame_extrinsic['translation'])
                hex = '#%02x%02x%02x' % color # bounding_box.color needs to be a tuple of floats (color is a tuple of ints)
                bounding_box.color = matplotlib.colors.to_rgb(hex)
                self.widget3d.scene.add_geometry(self.boxes['annotations'][i] + str(i), bounding_box, self.mat)
        
        # Force our widgets to update
        self.widget3d.force_redraw()
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
        if new_val == "Predicted" and self.pred_frames > 0:
            self.box_data_name = "pred_bounding"
            self.update()
        else: # switched to ground truth boxes
            self.box_data_name = "bounding"
            self.update()
    
    def update(self):
        """ This updates the window object to reflect new information such as user input
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
    
    