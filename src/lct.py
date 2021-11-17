"""
lct.py

Conversion tool to bring nuscenes dataset into LVT. 
"""

import getopt
import matplotlib.pyplot as plt
import time
import queue
from enum import Enum
from PIL import Image
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

    input_path = ""
    
    # Read in flags passed in with command line argument
    # Make sure that options which need an argument (namely -f for input file path and -o for output file path) have them
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:", "help")
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("use -f to specify directory of LVT dataset")

            sys.exit(2)
        elif opt == "-f": #and len(opts) == 2:
            input_path = arg
        else:
            # Only reach here if you were passed in a single option; consider this invalid input since we need both file paths
            print("Invalid set of arguments entered. Please refer to -h flag for more information.")
            sys.exit(2)

    return input_path  

class Window:
    MENU_IMPORT = 1
    def __init__(self, lct_dir):
        
        np.set_printoptions(precision=15)

        #Create the objects for the 3 windows that appear when running the application
        self.controls = gui.Application.instance.create_window("LCT", 400, 768)
        self.pointcloud_window = gui.Application.instance.create_window("PointCloud", 640, 480)
        self.image_window = gui.Application.instance.create_window("Image", 640, 480)

        #Set starting values for variables related to data paths
        #In the future, these values should not be set hard coded directly. Sensible default values should
        #be extracted from the LVT Directory

        self.lct_path = lct_dir
        print(self.lct_path)
        self.camera_sensors, self.lidar_sensors = self.get_cams_and_pointclouds(self.lct_path)
        
        #These two values represent the current Lidar and RGB sensors being displayed
        self.rgb_sensor_name = self.camera_sensors[0]
        self.lidar_sensor_name = self.lidar_sensors[0]
        self.pcd_path = os.path.join(self.lct_path, "pointcloud", self.lidar_sensor_name, "0.pcd")
        self.pcd_paths = []
        self.image_path = os.path.join(self.lct_path, "cameras", self.rgb_sensor_name, "0.jpg")

        #We extract any image from LCT in order to get the needed data to create our plt figure
        self.image = Image.open(self.image_path)
        self.image_w = self.image.width
        self.image_h = self.image.height
        self.image = np.asarray(self.image)
        self.fig, self.ax = plt.subplots(figsize=plt.figaspect(self.image))
        self.fig.subplots_adjust(0,0,1,1)

        #Image widget used to draw an image onto our image window
        self.image_widget = gui.ImageWidget()


        self.frame_num = 0
        #Dict that stores the imported JSON file that respresents the annoations in the current frame
        self.boxes = {}
        #List to store boxes as NuScenes Box Objects
        self.n_boxes = []

        #Alias
        cw = self.controls
        pw = self.pointcloud_window
        iw = self.image_window

        #Set up a vertical widget "layout" that will hold all of our horizontal widgets
        em = cw.theme.font_size
        layout = gui.Vert(0, gui.Margins(0.5 * em, 0.5 * em, 0.5 * em,
                                         0.5 * em))

        #Create SceneWidget() object to render pointclouds and bounding boxes
        self.widget3d = gui.SceneWidget()
        self.widget3d.scene = rendering.Open3DScene(pw.renderer)
        self.mat = rendering.Material()
        self.mat.shader = "defaultUnlit"
        self.mat.point_size = 3 * pw.scaling

        #Set up drop down menu for switching between RGB sensors
        sensor_select = gui.Combobox()
        for cam in self.camera_sensors:
            sensor_select.add_item(cam)
        sensor_select.set_on_selection_changed(self.on_sensor_select)

        #Horizontal widget where we will insert our drop down menu
        sensor_switch_layout = gui.Horiz()
        sensor_switch_layout.add_child(gui.Label("Switch RGB Sensor"))
        sensor_switch_layout.add_child(sensor_select)
       
        #Set up widget to switch between frames
        frame_select = gui.NumberEdit(gui.NumberEdit.INT)
        frame_select.set_on_value_changed(self.on_frame_switch)
        
        #Add frame switching widget to another horizontal widget
        frame_switch_layout = gui.Horiz()
        frame_switch_layout.add_child(gui.Label("Switch Frame"))
        frame_switch_layout.add_child(frame_select)

        #Add our two horizontal widgets to the vertical widget
        layout.add_child(sensor_switch_layout)
        layout.add_child(frame_switch_layout)

        #Add the master widgets to our three windows
        cw.add_child(layout)
        pw.add_child(self.widget3d)        
        iw.add_child(self.image_widget)

        #Call update function to draw all initial data
        self.update()

    #Fetches new image from LVT Directory, and draws it onto a plt figure
    #Uses nuScenes API to project 3D bounding boxes onto that plt figure
    #Finally, extracts raw image data from plt figure and updates our image widget
    def update_image(self):
        #Extract new image from file
        self.image = np.asarray(Image.open(self.image_path))

        #Set image width and height   
        #Figure out which bounding boxes are in our frame
        for box in self.n_boxes:

            #Box is stored in vehicle frame, so transform it to rgb sensor frame
            box.translate(-np.array(self.image_extrinsic['translation']))
            box.rotate(Quaternion(self.image_extrinsic['rotation']).inverse)
            
            if box_in_image(box, np.asarray(self.image_intrinsic['matrix']), (self.image_w, self.image_h), BoxVisibility.ANY):
                #If the box is in view, then render it onto the plt frame
                corners = view_points(box.corners(), np.asarray(self.image_intrinsic['matrix']), normalize=True)[:2, :]
                def draw_rect(selected_corners, color):
                    prev = selected_corners[-1]
                    for corner in selected_corners:
                        cv2.line(self.image,
                                (int(prev[0]), int(prev[1])),
                                (int(corner[0]), int(corner[1])),
                                color, 2)
                        prev = corner

                # Draw the sides
                for i in range(4):
                    cv2.line(self.image,
                            (int(corners.T[i][0]), int(corners.T[i][1])),
                            (int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
                            (255, 0, 0), 2)

                # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
                draw_rect(corners.T[:4], (255, 0, 0))
                draw_rect(corners.T[4:], (255, 0, 0))

                # Draw line indicating the front
                center_bottom_forward = np.mean(corners.T[2:4], axis=0)
                center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
                cv2.line(self.image,
                        (int(center_bottom[0]), int(center_bottom[1])),
                        (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
                        (0, 0, 255)[::-1], 2)

        new_image = o3d.geometry.Image(self.image)
        self.image_widget.update_image(new_image)

        #Force image widget to redraw
        self.image_window.post_redraw()

    #Updates bounding box information when switching frames
    def update_bounding(self):
        self.boxes = json.load(open(os.path.join(self.lct_path ,"bounding", str(self.frame_num), "boxes.json")))
        self.n_boxes = []
        for i in range(0, len(self.boxes['origins'])):
            self.n_boxes.append(Box(self.boxes['origins'][i], self.boxes['sizes'][i], Quaternion(self.boxes['rotations'][i]), name=self.boxes['annotations'][i], score=self.boxes['confidences'][i], velocity=(0,0,0)))

    #Takes new pointcloud data and converts it to global frame
    #Then renders the bounding boxes (Assuming the boxes are already in global frame)
    def update_pointcloud(self):
        self.widget3d.scene.clear_geometry()
        #Add Pointcloud
        temp_points = np.empty((0,3))
  
        for i, pcd_path in enumerate(self.pcd_paths):
            temp_cloud = o3d.io.read_point_cloud(pcd_path)
            sensor = self.lidar_sensors[i]
            #sensor_rotation_matrix = R.from_quat(self.pcd_extrinsic[sensor]['rotation']).as_matrix()
            ego_rotation_matrix = Quaternion(self.frame_extrinsic['rotation']).rotation_matrix

            #Transform lidar points into global frame
            temp_cloud.rotate(ego_rotation_matrix, [0,0,0])
            temp_cloud.translate(self.frame_extrinsic['translation'])
            temp_points = np.concatenate((temp_points, np.asarray(temp_cloud.points)))
 
        self.pointcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(temp_points)))
        #Add new global frame pointcloud to our 3D widget
        self.widget3d.scene.add_geometry("Point Cloud", self.pointcloud, self.mat)
        
        #This 'bounds' variable has nothing to do with the bounding boxes, it represents the box surrounding
        #all of our lidar points and is used to set up the camera for the scene
        bounds = self.widget3d.scene.bounding_box
        self.widget3d.setup_camera(10, bounds, self.frame_extrinsic['translation'])
        eye = [0,0,0]
        eye[0] = self.frame_extrinsic['translation'][0]
        eye[1] = self.frame_extrinsic['translation'][1]
        eye[2] = 150.0
        self.widget3d.scene.camera.look_at(self.frame_extrinsic['translation'], eye, [1, 0, 0])
        
        
        #Go through each box and render it onto our 3D Widget
        for i in range(0, len(self.boxes['origins'])):
            size = [0,0,0]
            #Open3D expects LxWxH but we store data in WxLxH so we do this conversion
            size[0] = self.boxes['sizes'][i][1]
            size[1] = self.boxes['sizes'][i][0]
            size[2] = self.boxes['sizes'][i][2]

            bounding_box = o3d.geometry.OrientedBoundingBox(self.boxes['origins'][i], o3d.geometry.get_rotation_matrix_from_quaternion(self.boxes['rotations'][i]), size)
            bounding_box.rotate(Quaternion(self.frame_extrinsic['rotation']).rotation_matrix, [0,0,0])
            bounding_box.translate(self.frame_extrinsic['translation'])
            self.widget3d.scene.add_geometry(self.boxes['annotations'][i] + str(i), bounding_box, self.mat)
        
        #Force our widgets to update
        self.widget3d.force_redraw()
        self.pointcloud_window.post_redraw()
    
    #Extracts all the pose data when switching sensors, and or frames
    def update_poses(self):
        self.image_intrinsic = json.load(open(os.path.join(self.lct_path, "cameras", self.rgb_sensor_name, "intrinsics.json")))
        self.image_extrinsic = json.load(open(os.path.join(self.lct_path, "cameras" , self.rgb_sensor_name, "extrinsics.json")))
        self.frame_extrinsic = json.load(open(os.path.join(self.lct_path, "ego", str(self.frame_num) + ".json")))
        
        
        self.pcd_extrinsic = {}
        for sensor_idx, path in enumerate(self.pcd_paths):
            fp = open(os.path.join(path))
            for i, line in enumerate(fp):
                if i == 8:
                    vals = line.split()
                    self.pcd_extrinsic[self.lidar_sensors[sensor_idx]] = {}
                    self.pcd_extrinsic[self.lidar_sensors[sensor_idx]]['translation'] = [float(vals[1]), float(vals[2]), float(vals[3])]
                    self.pcd_extrinsic[self.lidar_sensors[sensor_idx]]['rotation'] = [float(vals[4]), float(vals[5]), float(vals[6]), float(vals[7])]
            fp.close()

    #Callback function when a user selects a new RGB sensor from the dropdown menu    
    def on_sensor_select(self, new_val, new_idx):
        self.rgb_sensor_name = new_val
        self.update()

    def update_image_path(self):
        self.image_path = os.path.join(self.lct_path, "cameras", self.rgb_sensor_name, str(self.frame_num) +".jpg")
    
    def update_pcd_path(self):
        self.pcd_paths.clear()
        for sensor in self.lidar_sensors:
            self.pcd_paths.append(os.path.join(self.lct_path, "pointcloud", sensor, str(self.frame_num) + ".pcd"))

    #Callback function when the user selects a new frame
    def on_frame_switch(self, new_val):
        #Set new frame value
        self.frame_num = int(new_val)
        #Update Bounding Box List
        self.update()

    #Function that updates all displayed data based on the current state of the controls window
    def update(self):
        self.update_image_path()
        self.update_pcd_path()
        self.update_poses()
        self.update_bounding()
        self.update_image()
        self.update_pointcloud()

    #returns two lists of the names of the following sensors: (cameras, lidar sensors)
    def get_cams_and_pointclouds(self, path):
        camera_sensors = []
        lidar_sensors = []

        for camera_name in os.listdir(os.path.join(path, "cameras")):
            camera_sensors.append(camera_name)

        for lidar_name in os.listdir(os.path.join(path, "pointcloud")):
            lidar_sensors.append(lidar_name)

        return (camera_sensors, lidar_sensors)


if __name__ == "__main__":
    lct_dir = parse_options()
    gui.Application.instance.initialize()
    w = Window(lct_dir)
    o3d.visualization.gui.Application.instance.run()
    
    