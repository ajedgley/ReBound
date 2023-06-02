"""
	functions for editing
"""
import math

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import open3d as o3d
import functools
from functools import partial
from nuscenes.utils.data_classes import Box
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility
import matplotlib.colors
import numpy as np
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
from PIL import Image
import random
import os
import sys
import json
import cv2
from lct import Window
import platform

from copy import deepcopy

OS_STRING = platform.system()
ORIGIN = 0
SIZE = 1
ROTATION = 2
ANNOTATION = 3
CONFIDENCE = 4
COLOR = 5


class Annotation:
	# returns created window with all its buttons and whatnot
	def __init__(self, scene_widget, point_cloud, frame_extrinsic, boxes, pred_boxes, boxes_to_render,
				 boxes_in_scene, box_indices, annotation_types, path, color_map, pred_color_map,
				 image_window, image_widget, lct_path, frame_num, camera_sensors, lidar_sensors):
		self.cw = gui.Application.instance.create_window("LCT", 400, 800)
		self.scene_widget = scene_widget
		self.point_cloud = point_cloud
		self.image_window = image_window
		self.image_widget = image_widget
		self.frame_extrinsic = frame_extrinsic
		self.all_pred_annotations = annotation_types
		self.all_gt_annotations = list(color_map.keys())
		self.old_boxes = deepcopy(boxes)
		self.old_pred_boxes = deepcopy(pred_boxes)
		self.boxes_to_render = boxes_to_render 		#list of box metadata in scene
		self.box_indices = box_indices 				#name references for bounding boxes in scene
		self.boxes_in_scene = boxes_in_scene 		#current bounding box objects in scene
		self.volume_indices = [] 					#name references for clickable cube volumes in scene
		self.volumes_in_scene = [] 					#current clickable cube volume objects in scene
		self.color_map = color_map
		self.pred_color_map = pred_color_map

		self.frame_num = frame_num
		self.camera_sensors = camera_sensors
		self.rgb_sensor_name = self.camera_sensors[0]
		self.lct_path = lct_path
		self.image_path = os.path.join(self.lct_path, "cameras", self.rgb_sensor_name, str(self.frame_num) + ".jpg")
		self.image = Image.open(self.image_path)
		self.image_w = self.image.width
		self.image_h = self.image.height
		self.image = np.asarray(self.image)

		self.lidar_sensors = lidar_sensors
		self.lidar_sensor_name = lidar_sensors[0]
		self.pcd_path = os.path.join(self.lct_path, "pointcloud", self.lidar_sensor_name, "0.pcd")
		self.pcd_paths = []
		
		self.box_selected = None
		self.box_props_selected = [] #used for determining changes to property fields
		self.curr_box_depth = 0.0
		self.previous_index = -1 #-1 denotes, no box selected
		#used to generate unique ids for boxes and volumes
		self.box_count = 0

		#common materials
		self.transparent_mat = rendering.MaterialRecord() #invisible material for box volumes
		self.transparent_mat.shader = "defaultLitTransparency"
		self.transparent_mat.base_color = (0.0, 0.0, 0.0, 0.0)

		self.line_mat_highlight = rendering.MaterialRecord()
		self.line_mat_highlight.shader = "unlitLine"

		self.line_mat = rendering.MaterialRecord()
		self.line_mat.shader = "unlitLine"
		self.line_mat.line_width = 0.25

		self.coord_frame_mat = rendering.MaterialRecord()
		self.coord_frame_mat.shader = "defaultUnlit"

		self.coord_frame = "coord_frame"

		# mouse and key event modifiers
		self.z_drag = False
		self.drag_operation = True
		self.curr_x = 0.0 #used for initial mouse position in drags
		self.curr_y = 0.0
		self.ctrl_is_down = False
		self.nudge_sensitivity = 1.0
		
		# modify temp boxes in this file, then when it's time to save use them to overwrite existing json
		self.temp_boxes = boxes.copy()
		self.temp_pred_boxes = pred_boxes.copy()

		# used for adding new annotations
		self.new_annotation_types = []

		#initialize the scene with transparent volumes to allow mouse interactions with boxes
		self.create_box_scene(scene_widget, boxes_to_render, frame_extrinsic)
		self.average_depth = self.get_depth_average()

		# calculates margins off of font size
		em = self.cw.theme.font_size
		margin = gui.Margins(0.50 * em, 0.25 * em, 0.50 * em, 0.25 * em)
		layout = gui.Vert(0.50 * em, margin)

		# num of frames available to display
		frames_available = [entry for entry in os.scandir(os.path.join(self.lct_path, "bounding"))]
		self.num_frames = len(frames_available)

		# switch between frames
		self.frame_select = gui.NumberEdit(gui.NumberEdit.INT)
		self.frame_select.set_limits(0, self.num_frames)
		self.frame_select.set_value(self.frame_num)
		self.frame_select.set_on_value_changed(self.on_frame_switch)

		frame_switch_layout = gui.Horiz()
		frame_switch_layout.add_child(gui.Label("Switch Frame"))
		frame_switch_layout.add_child(self.frame_select)

		# button to center pointcloud view on vehicle
		center_horiz = gui.Horiz()
		center_view_button = gui.Button("Center Pointcloud View on Vehicle")
		center_view_button.set_on_clicked(self.jump_to_vehicle)
		#center_horiz.add_child(gui.Label("Center Pointcloud View on Vehicle"))
		center_horiz.add_child(center_view_button)

		self.label_list = []

		# default to showing predicted data while editing
		self.show_gt = False
		self.show_pred = True

		# hardcoding to test
		self.min_confidence = 50
		# Set up a widget to specify a minimum annotation confidence
		confidence_select = gui.NumberEdit(gui.NumberEdit.INT)
		confidence_select.set_limits(0,100)
		confidence_select.set_value(50)
		confidence_select.set_on_value_changed(self.on_confidence_switch)

		# Add confidence select widget to horizontal
		confidence_select_layout = gui.Horiz()
		confidence_select_layout.add_child(gui.Label("Specify (Pred) Confidence Threshold"))
		confidence_select_layout.add_child(confidence_select)

		# Add combobox to switch between predicted and ground truth
		self.bounding_toggle = gui.Combobox()
		self.bounding_toggle.add_item("Predicted")
		self.bounding_toggle.add_item("Ground Truth")
		self.bounding_toggle.set_on_selection_changed(self.toggle_bounding)

		bounding_toggle_layout = gui.Horiz()
		bounding_toggle_layout.add_child(gui.Label("Toggle Predicted or GT"))
		bounding_toggle_layout.add_child(self.bounding_toggle)

		frames_available = [entry for entry in os.scandir(os.path.join(self.lct_path, "bounding"))]
		self.pred_frames = len(frames_available) - 1

		self.propagated_gt_boxes = []
		self.propagated_pred_boxes = []

		# buttons for saving/saving as annotation changes
		save_annotation_vert = gui.CollapsableVert("Save")
		save_annotation_horiz = gui.Horiz(0.50 * em, margin)
		save_annotation_button = gui.Button("Save Changes")
		save_partial = functools.partial(self.save_changes_to_json)
		save_annotation_button.set_on_clicked(save_partial)
		self.save_check = 0
		save_as_button = gui.Button("Save As")
		save_as_button.set_on_clicked(self.save_as)
		save_and_prop_button = gui.Button("Save and Propagate Changes to Next Frame")
		save_and_prop_to_next = functools.partial(self.save_and_propagate)
		save_and_prop_button.set_on_clicked(save_and_prop_to_next)


		save_annotation_horiz.add_child(save_annotation_button)
		save_annotation_horiz.add_child(save_as_button)
		save_annotation_vert.add_child(save_annotation_horiz)
		save_annotation_vert.add_child(save_and_prop_button)

		add_remove_vert = gui.CollapsableVert("Add/Delete")
		add_box_button = gui.Button("Add Bounding Box")
		add_box_button.set_on_clicked(self.place_bounding_box)
		self.delete_annotation_button = gui.Button("Delete Bounding Box")
		self.delete_annotation_button.set_on_clicked(self.delete_annotation)
		add_remove_horiz = gui.Horiz(0.50 * em, margin)
		add_remove_horiz.add_child(add_box_button)
		add_remove_horiz.add_child(self.delete_annotation_button)
		add_remove_vert.add_child(add_remove_horiz)

		tool_vert = gui.CollapsableVert("Tools")
		tool_status_horiz = gui.Horiz(0.50 * em, margin)
		tool_status_horiz.add_child(gui.Label("Current Tool:"))
		self.current_tool = gui.Label("Translation")
		tool_status_horiz.add_child(self.current_tool)


		toggle_operation_horiz = gui.Horiz(0.50 * em, margin)
		toggle_operation_button = gui.Button("Toggle Translate/Rotate")
		toggle_operation_button.set_on_clicked(self.toggle_drag_operation)
		toggle_operation_button.tooltip = "To use the tool, \n select a box using CTRL + Left Click, \n then hold SHIFT + Left Click to drag the box"
		toggle_operation_horiz.add_child(toggle_operation_button)

		# dropdown selector for selecting current drag mode
		self.toggle_horiz = gui.Horiz(0.50 * em, margin)
		toggle_label = gui.Label("Current Drag Mode:")
		self.toggle_axis_selector = gui.Combobox()
		self.toggle_axis_selector.set_on_selection_changed(self.toggle_axis)
		self.toggle_axis_selector.add_item("Horizontal")
		self.toggle_axis_selector.add_item("Vertical")
		self.toggle_horiz.add_child(toggle_label)
		self.toggle_horiz.add_child(self.toggle_axis_selector)

		tool_vert.add_child(tool_status_horiz)
		tool_vert.add_child(toggle_operation_horiz)
		tool_vert.add_child(self.toggle_horiz)

		toggle_camera_vert = gui.CollapsableVert("Camera")
		toggle_camera_horiz = gui.Horiz(0.50 * em, margin)
		toggle_camera_label = gui.Label("Camera:")
		toggle_camera_selector = gui.Combobox()
		toggle_camera_selector.set_on_selection_changed(self.on_sensor_select)
		for cam in self.camera_sensors:
			toggle_camera_selector.add_item(cam)
		toggle_camera_horiz.add_child(toggle_camera_label)
		toggle_camera_horiz.add_child(toggle_camera_selector)
		toggle_camera_vert.add_child(toggle_camera_horiz)

		#The data for a selected box will be displayed in these fields
		#the data fields are accessible to any function to allow easy manipulation during drag operations
		properties_vert = gui.CollapsableVert("Properties", 0.25 * em, margin)
		trans_collapse = gui.CollapsableVert("Position")
		rot_collapse = gui.CollapsableVert("Rotation")
		scale_collapse = gui.CollapsableVert("Scale")

		self.annotation_class = gui.Combobox()
		self.annotation_class.set_on_selection_changed(self.label_change_handler)
		for annotation in self.all_pred_annotations:
			self.annotation_class.add_item(annotation)
		self.trans_x = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
		self.trans_x.set_on_value_changed(partial(self.property_change_handler, prop="trans", axis="x"))
		self.trans_y = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
		self.trans_y.set_on_value_changed(partial(self.property_change_handler, prop="trans", axis="y"))
		self.trans_z = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
		self.trans_z.set_on_value_changed(partial(self.property_change_handler, prop="trans", axis="z"))
		self.rot_x = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
		self.rot_x.set_on_value_changed(partial(self.property_change_handler, prop="rot", axis="x"))
		self.rot_y = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
		self.rot_y.set_on_value_changed(partial(self.property_change_handler, prop="rot", axis="y"))
		self.rot_z = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
		self.rot_z.set_on_value_changed(partial(self.property_change_handler, prop="rot", axis="z"))
		self.scale_x = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
		self.scale_x.set_on_value_changed(partial(self.property_change_handler, prop="scale", axis="x"))
		self.scale_y = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
		self.scale_y.set_on_value_changed(partial(self.property_change_handler, prop="scale", axis="y"))
		self.scale_z = gui.NumberEdit(gui.NumberEdit.Type.DOUBLE)
		self.scale_z.set_on_value_changed(partial(self.property_change_handler, prop="scale", axis="z"))
		
		annot_type = gui.Horiz(0.50 * em, margin)
		annot_type.add_child(gui.Label("Type:"))
		self.annotation_type = gui.Label("                       ")
		annot_type.add_child(self.annotation_type)
		annot_class = gui.Horiz(0.50 * em, margin)
		annot_class.add_child(gui.Label("Class:"))
		annot_class.add_child(self.annotation_class)
		add_custom_horiz = gui.Horiz(0.50 * em, margin)
		add_custom_annotation_button = gui.Button("Add Custom Annotation")
		add_custom_annotation_button.set_on_clicked(self.add_new_annotation_type)
		add_custom_horiz.add_child(add_custom_annotation_button)
		annot_vert = gui.CollapsableVert("Annotation")
		annot_vert.add_child(annot_type)
		annot_vert.add_child(annot_class)
		annot_vert.add_child(add_custom_horiz)
		
		trans_horiz = gui.Horiz(0.5 * em)
		trans_horiz.add_child(gui.Label("X:"))
		trans_horiz.add_child(self.trans_x)
		trans_horiz.add_child(gui.Label("Y:"))
		trans_horiz.add_child(self.trans_y)
		trans_horiz.add_child(gui.Label("Z:"))
		trans_horiz.add_child(self.trans_z)
		trans_collapse.add_child(trans_horiz)

		rot_horiz = gui.Horiz(0.5 * em)
		rot_horiz.add_child(gui.Label("X:"))
		rot_horiz.add_child(self.rot_x)
		rot_horiz.add_child(gui.Label("Y:"))
		rot_horiz.add_child(self.rot_y)
		rot_horiz.add_child(gui.Label("Z:"))
		rot_horiz.add_child(self.rot_z)
		rot_collapse.add_child(rot_horiz)

		scale_horiz = gui.Horiz(0.5 * em)
		scale_horiz.add_child(gui.Label("X:"))
		scale_horiz.add_child(self.scale_x)
		scale_horiz.add_child(gui.Label("Y:"))
		scale_horiz.add_child(self.scale_y)
		scale_horiz.add_child(gui.Label("Z:"))
		scale_horiz.add_child(self.scale_z)
		scale_collapse.add_child(scale_horiz)

		properties_vert.add_child(annot_vert)
		properties_vert.add_child(trans_collapse)
		properties_vert.add_child(rot_collapse)
		properties_vert.add_child(scale_collapse)

		# button for exiting annotation mode, set_on_click in lct.py for a cleaner restart
		exit_annotation_horiz = gui.Horiz(0.50 * em, margin)
		exit_annotation_button = gui.Button("Exit Annotation Mode")
		exit_annotation_button.set_on_clicked(self.exit_annotation_mode)
		exit_annotation_horiz.add_child(exit_annotation_button)

		# adding all of the horiz to the vert, in order
		layout.add_child(save_annotation_vert)
		layout.add_child(frame_switch_layout)
		layout.add_child(center_horiz)
		layout.add_child(confidence_select_layout)
		layout.add_child(bounding_toggle_layout)
		layout.add_child(add_remove_vert)
		layout.add_child(tool_vert)
		layout.add_child(toggle_camera_vert)
		layout.add_child(properties_vert)

		layout.add_child(exit_annotation_horiz)

		self.cw.add_child(layout)
		self.update_props()
		self.update_poses()
		# Event handlers
		
		# sets up onclick box selection and drag interactions
		self.scene_widget.set_on_mouse(self.mouse_event_handler)

		# sets up keyboard event handling
		#key_partial = functools.partial(self.key_event_handler, widget=scene_widget)
		scene_widget.set_on_key(self.key_event_handler)

	#helper function to place new boxes at the direct camera origin at the depth average
	def get_center_of_rotation(self):
		#view_matrix = self.scene_widget.scene.camera.get_view_matrix()
		#inverse = np.linalg.inv(view_matrix)
		#return (inverse[0][3], inverse[1][3], self.average_depth)
		R = Quaternion(scalar=1.0, vector=[0.0, 0.0, 0.0]).rotation_matrix
		box = o3d.geometry.OrientedBoundingBox([0.0, 0.0, 0.0], R, [0.0, 0.0, 0.0])
		box.rotate(Quaternion(self.image_extrinsic['rotation']).rotation_matrix, [0, 0, 0])
		box.translate(self.image_extrinsic['translation'])
		box.rotate(Quaternion(self.frame_extrinsic['rotation']).rotation_matrix, [0, 0, 0])
		box.translate(self.frame_extrinsic['translation'])
		return box.get_center()

	# onclick, places down a bounding box on the cursor, then reenables mouse functionality
	def place_bounding_box(self):
		# Random values are placeholders until we implement the desired values
		qtr = Quaternion(axis=(1.0,0.0,0.0), degrees=0) #Randomized rotation of box
		origin = self.get_center_of_rotation()
		size = [random.randint(1,5),random.randint(1,5),random.randint(1,5)] #Random dimensions of box
		bbox_params = [origin, size, qtr.rotation_matrix] #create_volume uses box meta data to create mesh
		vol_size = [bbox_params[1][1], bbox_params[1][0], bbox_params[1][2]]
		vol_params = [origin, vol_size, qtr.rotation_matrix]
		bounding_box = o3d.geometry.OrientedBoundingBox(origin, qtr.rotation_matrix, size) #Creates bounding box object
		if self.show_gt:
			color = self.color_map[self.all_gt_annotations[0]]
		else:
			color = self.pred_color_map[self.all_pred_annotations[0]]
		hex = '#%02x%02x%02x' % color
		bounding_box.color = matplotlib.colors.to_rgb(hex) #will select color from annotation type list
		bbox_name = "bbox_" + str(self.box_count)
		self.box_indices.append(bbox_name)
		self.boxes_in_scene.append(bounding_box)

		volume_to_add = self.add_volume(vol_params)
		volume_name = "volume_" + str(self.box_count)
		self.volume_indices.append(volume_name)
		self.volumes_in_scene.append(volume_to_add)
		volume_to_add.compute_vertex_normals()

		box_object_data = self.create_box_metadata(origin, size, qtr.elements, self.all_pred_annotations[0], 101, "", 0, {})
		if self.show_gt:
			self.temp_boxes['boxes'].append(box_object_data)
		else:
			self.temp_pred_boxes['boxes'].append(box_object_data)
		self.scene_widget.scene.add_geometry(bbox_name, bounding_box, self.line_mat) #Adds the box to the scene
		self.scene_widget.scene.add_geometry(volume_name, volume_to_add, self.transparent_mat)#Adds the volume
		self.box_selected = bbox_name

		self.point_cloud.post_redraw()
		self.cw.post_redraw()
		self.box_count += 1

	#Takes the frame x and y coordinates and flattens the 3D scene into a 2D depth image
	#The X and Y coordinates select the depth value from the depth image and converts it into a depth value
	#After getting the coordinates, it automatically calls the closest distance function
	def mouse_event_handler(self, event):
		widget = self.scene_widget
		if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.CTRL):

			#uses depth image to calculate a depth from an x and y mouse pointer coordinate
			#automatically finds closest box to resulting world coordinates
			def get_depth(depth_image): #gets world coords from mouse click

				x = event.x - widget.frame.x
				y = event.y - widget.frame.y

				depth = np.asarray(depth_image)[y, x] #flatten image to depth image, get color value from x,y point
				if depth == 1.0:
					self.deselect_box() #if depth is 1.0, that is the far plane, deselect
				else: #select nearest box
					world = widget.scene.camera.unproject(
						event.x, event.y, depth, widget.frame.width, widget.frame.height)
					self.curr_box_depth = depth

					get_nearest(world)


			#simple shortest distance comparison from pointer to box center points
			#searches using x and y coordinates because using z coord makes behavior unpredictable
			def get_nearest(world_coords):
				boxes = self.volumes_in_scene
				if len(boxes) != 0:
					flat_coords_world = np.array([world_coords[0], world_coords[1], self.average_depth])
					curr_coords_box = boxes[0].get_center()
					flat_coords_box = np.array([curr_coords_box[0], curr_coords_box[1], self.average_depth])
					smallest_dist = np.linalg.norm(flat_coords_world - flat_coords_box)
					closest_box = boxes[0]

					for box in boxes:
						curr_coords_box = box.get_center()
						flat_coords_box = np.array([curr_coords_box[0], curr_coords_box[1], self.average_depth])
						curr_dist = np.linalg.norm(flat_coords_world - flat_coords_box)
						if curr_dist < smallest_dist:
							smallest_dist = curr_dist
							closest_box = box

					closest_index = boxes.index(closest_box) #get the array position of closest_box
					self.box_selected = self.box_indices[closest_index]
					self.select_box(closest_index) #select the nearest box

			widget.scene.scene.render_to_depth_image(get_depth)
			return gui.Widget.EventCallbackResult.HANDLED

		#If shift button is down during click event, indicates potential drag operation
		elif event.is_modifier_down(gui.KeyModifier.SHIFT) and self.previous_index != -1:
			current_box = self.previous_index
			scene_camera = self.scene_widget.scene.camera
			volume_to_drag = self.volumes_in_scene[current_box]
			volume_name = self.volume_indices[current_box]
			box_to_drag = self.boxes_in_scene[current_box]
			box_name = self.box_indices[current_box]
			

			#otherwise it's the drag part of the event, continually translate current box by the difference between
			#start position and current position, multiply by scaling factor due to size of grid
			if event.type == gui.MouseEvent.Type.DRAG:
				box_center = box_to_drag.center
				rendering.Open3DScene.remove_geometry(self.scene_widget.scene, box_name)
				rendering.Open3DScene.remove_geometry(self.scene_widget.scene, volume_name)
				rendering.Open3DScene.remove_geometry(self.scene_widget.scene, "coord_frame")
				curr_pos = scene_camera.unproject(event.x, event.y, self.curr_box_depth,
												  self.scene_widget.frame.width, self.scene_widget.frame.height)
				x_diff = curr_pos[0] - box_center[0]
				y_diff = curr_pos[1] - box_center[1]

				if self.drag_operation:
					if self.z_drag:  # if z_drag is on, translate by z axis only
						box_to_drag.translate((0, 0, x_diff/10))
						volume_to_drag.translate((0, 0, x_diff/10))
					else:
						box_to_drag.translate((x_diff, y_diff, 0))
						volume_to_drag.translate((x_diff, y_diff, 0))
				else:
					rotation = Quaternion(axis=[0, 0, -1], degrees=x_diff).rotation_matrix
					box_to_drag.rotate(rotation)
					volume_to_drag.rotate(rotation)


				self.scene_widget.scene.add_geometry(box_name, box_to_drag, self.line_mat_highlight)
				self.scene_widget.scene.add_geometry(volume_name, volume_to_drag, self.transparent_mat)
				coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(2.0, box_to_drag.center)
				self.scene_widget.scene.add_geometry("coord_frame", coord_frame, self.coord_frame_mat)

				self.update_props()
				self.scene_widget.force_redraw()
				self.point_cloud.post_redraw()

			elif event.type == gui.MouseEvent.Type.BUTTON_UP:
				self.update_poses()

			return gui.Widget.EventCallbackResult.CONSUMED



		return gui.Widget.EventCallbackResult.IGNORED

		# Handles key events
	def key_event_handler(self, event):
		# delete button handler
		if event.key == gui.KeyName.LEFT_CONTROL:  # handles events involving ctrl + key
			if event.type == event.Type.DOWN:
				self.ctrl_is_down = True
			else:
				self.ctrl_is_down = False
			return gui.Widget.EventCallbackResult.HANDLED

		elif event.type == event.Type.DOWN:

			if event.key == 127:
				self.delete_annotation()
				return gui.Widget.EventCallbackResult.CONSUMED
			elif event.key == 100 and self.ctrl_is_down:
				self.deselect_box()
				return gui.Widget.EventCallbackResult.CONSUMED
			elif self.previous_index != -1:
				if event.key == 119:
					z_location = self.box_props_selected[2] + self.nudge_sensitivity
					self.property_change_handler(z_location, "trans", "z")
					return gui.Widget.EventCallbackResult.CONSUMED
				elif event.key == 115:
					z_location = self.box_props_selected[2] + (-1 * self.nudge_sensitivity)
					self.property_change_handler(z_location, "trans", "z")
					return gui.Widget.EventCallbackResult.CONSUMED
				elif event.key == 97:
					self.property_change_handler(self.nudge_sensitivity, "rot", "z")
					return gui.Widget.EventCallbackResult.CONSUMED
				elif event.key == 100:
					self.property_change_handler(-1 * self.nudge_sensitivity, "rot", "z")
					return gui.Widget.EventCallbackResult.CONSUMED
				elif event.key == 265:
					y_location = self.box_props_selected[1] + self.nudge_sensitivity
					self.property_change_handler(y_location, "trans", "y")
					return gui.Widget.EventCallbackResult.CONSUMED
				elif event.key == 266:
					y_location = self.box_props_selected[1] + (-1 * self.nudge_sensitivity)
					self.property_change_handler(y_location, "trans", "y")
					return gui.Widget.EventCallbackResult.CONSUMED
				elif event.key == 263:
					x_location = self.box_props_selected[0] + self.nudge_sensitivity
					self.property_change_handler(x_location, "trans", "x")
					return gui.Widget.EventCallbackResult.CONSUMED
				elif event.key == 264:
					x_location = self.box_props_selected[0] + (-1 * self.nudge_sensitivity)
					self.property_change_handler(x_location, "trans", "x")
					return gui.Widget.EventCallbackResult.CONSUMED

		return gui.Widget.EventCallbackResult.IGNORED
	#deselect_box removes current properties, un-highlights box, and sets selected box back to -1
	def deselect_box(self):
		if self.previous_index != -1:
			self.scene_widget.scene.modify_geometry_material(self.box_indices[self.previous_index], self.line_mat)
			self.scene_widget.scene.show_geometry(self.coord_frame, False)

		self.point_cloud.post_redraw()

		self.previous_index = -1
		self.box_selected = None
		self.update_props()
		self.update_poses()

	#select_box takes a box name (string) and checks to see if a previous box has been selected
	#then it modifies the appropriate line widths to select and deselect boxes
	#it also moves the coordinate frame to the selected box
	def select_box(self, box_index):
		if self.previous_index != -1:  # if not first box clicked "deselect" previous box
			self.scene_widget.scene.modify_geometry_material(self.box_indices[self.previous_index], self.line_mat)

		rendering.Open3DScene.remove_geometry(self.scene_widget.scene, self.coord_frame)
		self.previous_index = box_index
		box = self.box_indices[box_index]
		origin = o3d.geometry.TriangleMesh.get_center(self.volumes_in_scene[box_index])
		frame = o3d.geometry.TriangleMesh.create_coordinate_frame(2.0, origin)
		rendering.Open3DScene.modify_geometry_material(self.scene_widget.scene, box, self.line_mat_highlight)
		self.scene_widget.scene.add_geometry("coord_frame", frame, self.coord_frame_mat, True)
		self.scene_widget.force_redraw()
		self.update_props()
		self.update_poses()

	#This method adds cube mesh volumes to preexisting bounding boxes
	#Adds an initial coordinate frame to the scene
	def create_box_scene(self, scene, boxes, extrinsics):
		coord_frame_mat = self.coord_frame_mat
		frame_to_add = o3d.geometry.TriangleMesh.create_coordinate_frame()
		scene.scene.add_geometry("coord_frame", frame_to_add, coord_frame_mat, False)
		for box in boxes:
			volume_to_add = self.add_volume((box[0], box[1], Quaternion(box[2]).rotation_matrix))
			volume_to_add = volume_to_add.rotate(Quaternion(extrinsics['rotation']).rotation_matrix, [0, 0, 0])
			volume_to_add = volume_to_add.translate(np.array(extrinsics['translation']))
			cube_id = "volume_" + str(self.box_count)
			self.volume_indices.append(cube_id)

			volume_to_add.compute_vertex_normals()
			self.volumes_in_scene.append(volume_to_add)
			self.scene_widget.scene.add_geometry(cube_id, volume_to_add, self.transparent_mat)
			self.box_count += 1

		self.point_cloud.post_redraw()

	#when something changes with a box, that means it is currently selected
	#update the properties in the property window
	def update_props(self):
		# Enables or disables boxes, depending on whether box is currently selected
		boxes = [self.annotation_class, self.trans_x, self.trans_y, self.trans_z, self.rot_x, self.rot_y, self.rot_z, self.scale_x, self.scale_y,
			self.scale_z, self.delete_annotation_button]
		enabled = False
		if self.box_selected is not None:
			enabled = True

		for i in boxes:
			i.enabled = enabled

		if not enabled:
			boxes[0].selected_index = 0
			for i in range(1,10):
				boxes[i].double_value = 0
			self.cw.post_redraw()
			return -1

		annot_type = gui.Horiz()
		annot_type.add_child(gui.Label("Type:"))
		annot_type.add_child(self.annotation_type)
		annot_class = gui.Horiz()
		annot_class.add_child(gui.Label("Class:"))
		annot_class.add_child(self.annotation_class)
		annot_vert = gui.Vert()
		annot_vert.add_child(annot_type)
		annot_vert.add_child(annot_class)
		current_box = self.previous_index
		box_object = self.boxes_in_scene[current_box]

		scaled_color = tuple(255*x for x in box_object.color)
		if scaled_color in self.color_map.values():
			self.annotation_type.text = "Ground Truth"
			selected = list(self.color_map.keys())[list(self.color_map.values()).index(scaled_color)]
			if self.annotation_class.get_item(0) != selected:
				self.annotation_class.clear_items()
				self.annotation_class.add_item(selected)
				for annotation in self.color_map:
					if annotation != selected:
						self.annotation_class.add_item(annotation)
		elif scaled_color in self.pred_color_map.values():
			self.annotation_type.text = "Prediction"
			selected = list(self.pred_color_map.keys())[list(self.pred_color_map.values()).index(scaled_color)]
			if self.annotation_class.get_item(0) != selected:
				self.annotation_class.clear_items()
				self.annotation_class.add_item(selected)
				for annotation in self.all_pred_annotations + self.new_annotation_types:
					if annotation != selected:
						self.annotation_class.add_item(annotation)

		box_center = box_object.center
		box_rotate = list(box_object.R)
		r = Rotation.from_matrix(box_rotate)
		euler_rotations = r.as_euler("xyz", False)
		box_scale = box_object.extent

		self.trans_x.double_value = box_center[0]
		self.trans_y.double_value = box_center[1]
		self.trans_z.double_value = box_center[2]

		self.rot_x.double_value = math.degrees(euler_rotations[0])
		self.rot_y.double_value = math.degrees(euler_rotations[1])
		self.rot_z.double_value = math.degrees(euler_rotations[2])

		self.scale_x.double_value = box_scale[0]
		self.scale_y.double_value = box_scale[1]
		self.scale_z.double_value = box_scale[2]

		#updates array of all properties to allow referencing previous values
		self.box_props_selected = [
			box_center[0], box_center[1], box_center[2],
			euler_rotations[0], euler_rotations[1], euler_rotations[2],
			box_scale[0], box_scale[1], box_scale[2]
		]
		#simulates reversing the extrinsic transform and rotation to get the correct location of the object according
		#to the boxes.json file
		box_to_rotate = o3d.geometry.OrientedBoundingBox(box_object) #copy box object to do transforms on
		reverse_extrinsic = Quaternion(self.frame_extrinsic['rotation']).inverse
		box_to_rotate.translate(-np.array(self.frame_extrinsic['translation']))
		box_to_rotate = box_to_rotate.rotate(reverse_extrinsic.rotation_matrix, [0,0,0])
		result = Quaternion(matrix=box_to_rotate.R)
		size = [box_scale[1], box_scale[0], box_scale[2]] #flip the x and y scale back
		if self.show_gt:
			current_temp_box = self.temp_boxes["boxes"][self.previous_index]
		else:
			current_temp_box = self.temp_pred_boxes["boxes"][self.previous_index]
		updated_box_metadata = self.create_box_metadata(box_to_rotate.center, size, result.elements, current_temp_box["annotation"], current_temp_box["confidence"], "", 0, {})
		if self.show_gt:
			self.temp_boxes['boxes'][self.previous_index] = updated_box_metadata
		else:
			self.temp_pred_boxes['boxes'][self.previous_index] = updated_box_metadata
		self.cw.post_redraw()

	#redirects on_value_changed events to appropriate box transformation function
	def property_change_handler(self, value, prop, axis):
		value_as_float = float(value)
		if math.isnan(value_as_float): #handles not a number inputs
			value_as_float = 0.0
		if prop == "trans":
			self.translate_box(axis, value_as_float)
		elif prop == "rot":
			self.rotate_box(axis, value_as_float)
		elif prop == "scale":
			self.scale_box(axis, value_as_float)

		self.update_poses()

	# on label change, changes temp_boxes value and color of current box
	def label_change_handler(self, label, pos):
		if self.show_gt:
			self.temp_boxes["boxes"][self.previous_index]["annotation"] = label
		else:
			self.temp_pred_boxes["boxes"][self.previous_index]["annotation"] = label
		current_box = self.boxes_in_scene[self.previous_index]
		box_name = self.box_indices[self.previous_index]
		if self.show_gt:
			box_data = self.temp_boxes["boxes"][self.previous_index]
		else:
			box_data = self.temp_pred_boxes["boxes"][self.previous_index]
		self.scene_widget.scene.remove_geometry(box_name)

		# changes color of box based on label selection
		new_color = None
		if label in self.color_map and box_data["confidence"] == 101:
			new_color = self.color_map[label]
		elif label in self.pred_color_map:
			new_color = self.pred_color_map[label]

		new_color = tuple(x/255 for x in new_color)
		current_box.color = new_color
		self.scene_widget.scene.add_geometry(box_name, current_box, self.line_mat_highlight)

		self.point_cloud.post_redraw()
		self.update_poses()

	#used by property fields to move box along specified axis to new position -> value
	def translate_box(self, axis, value):
		current_box = self.previous_index
		box_to_drag = self.boxes_in_scene[current_box]
		box_name = self.box_indices[current_box]
		volume_to_drag = self.volumes_in_scene[current_box]
		volume_name = self.volume_indices[current_box]

		self.scene_widget.scene.remove_geometry(box_name)
		self.scene_widget.scene.remove_geometry(volume_name)
		self.scene_widget.scene.remove_geometry("coord_frame")

		if axis == "x":
			diff = value - self.box_props_selected[0]
			box_to_drag.translate([diff, 0, 0])
			volume_to_drag.translate([diff, 0, 0])

		elif axis == "y":
			diff = value - self.box_props_selected[1]
			box_to_drag.translate([0, diff, 0])
			volume_to_drag.translate([0, diff, 0])

		else: #the axis is z no other inputs are able to be entered
			diff = value - self.box_props_selected[2]
			box_to_drag.translate([0, 0, diff])
			volume_to_drag.translate([0, 0, diff])

		coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(2.0, box_to_drag.center)

		self.scene_widget.scene.add_geometry(box_name, box_to_drag, self.line_mat_highlight)
		self.scene_widget.scene.add_geometry(volume_name, volume_to_drag, self.transparent_mat)
		self.scene_widget.scene.add_geometry("coord_frame", coord_frame, self.coord_frame_mat)
		self.update_props()
		self.point_cloud.post_redraw()

	#Handler function for rotating annotation boxes during property field updates
	#uses in place rotation methods to modify the geometries requiring no update to
	#volume and boxes _in_scene entries.
	def rotate_box(self, axis, value):
		current_box = self.previous_index
		box_to_drag = self.boxes_in_scene[current_box]
		box_name = self.box_indices[current_box]
		volume_to_drag = self.volumes_in_scene[current_box]
		volume_name = self.volume_indices[current_box]

		self.scene_widget.scene.remove_geometry(box_name)
		self.scene_widget.scene.remove_geometry(volume_name)
		current_rot = Quaternion(matrix=box_to_drag.R) #used to rotate the rotation axis for local rotation of boxes
		if axis == "x":
			rotation_axis = current_rot.rotate([1, 0, 0])
			rotation = Quaternion(axis=rotation_axis, degrees=value).rotation_matrix
			box_to_drag.rotate(rotation)
			volume_to_drag.rotate(rotation)
		elif axis == "y":
			rotation_axis = current_rot.rotate([0, 1, 0])
			rotation = Quaternion(axis=rotation_axis, degrees=value).rotation_matrix
			box_to_drag.rotate(rotation)
			volume_to_drag.rotate(rotation)
		else:
			rotation_axis = current_rot.rotate([0, 0, 1])
			rotation = Quaternion(axis=rotation_axis, degrees=value).rotation_matrix
			box_to_drag.rotate(rotation)
			volume_to_drag.rotate(rotation)

		self.scene_widget.scene.add_geometry(box_name, box_to_drag, self.line_mat_highlight)
		self.scene_widget.scene.add_geometry(volume_name, volume_to_drag, self.transparent_mat)
		self.update_props()
		self.point_cloud.post_redraw()

	#The .scale method multiplies all vectors by a single factor. To work around this, scale_box
	#deletes the geometry for volume, creates a brand new one with an updated scale and overwrites the previous
	#volume in self.volumes_in_scene
	def scale_box(self, axis, value):
		current_box = self.previous_index
		box_to_drag = self.boxes_in_scene[current_box]
		box_center = box_to_drag.center
		box_name = self.box_indices[current_box]
		volume_to_drag = self.volumes_in_scene[current_box]
		volume_name = self.volume_indices[current_box]

		self.scene_widget.scene.remove_geometry(box_name)
		self.scene_widget.scene.remove_geometry(volume_name)
		trans = [self.box_props_selected[0], self.box_props_selected[1], self.box_props_selected[2]]
		scale = []
		qrt = box_to_drag.R
		if axis == "x":
			scale = [value, self.box_props_selected[7], self.box_props_selected[8]]
			box_to_drag.extent = scale
			volume_to_drag = self.add_volume([trans, (scale[1], scale[0], scale[2]), qrt])

		elif axis == "y":
			scale = [self.box_props_selected[6], value, self.box_props_selected[8]]
			box_to_drag.extent = scale
			volume_to_drag = self.add_volume([trans, (scale[1], scale[0], scale[2]), qrt])

		else:
			scale = [self.box_props_selected[6], self.box_props_selected[7], value]
			box_to_drag.extent = scale
			volume_to_drag = self.add_volume([trans, (scale[1], scale[0], scale[2]), qrt])

		volume_to_drag.compute_vertex_normals()
		self.volumes_in_scene[self.previous_index] = volume_to_drag
		self.scene_widget.scene.add_geometry(box_name, box_to_drag, self.line_mat_highlight)
		self.scene_widget.scene.add_geometry(volume_name, volume_to_drag, self.transparent_mat)
		self.update_props()
		self.point_cloud.post_redraw()
	
	#general cube_mesh function to create cube mesh from bounding box information
	#positions cube mesh at center of bounding box allowing the boxes to be selectable
	#takes array of [origin, size, rotation_matrix]
	def add_volume(self, box):
		size = [0, 0, 0]
		size[0] = box[SIZE][1]
		size[1] = box[SIZE][0]
		size[2] = box[SIZE][2]

		cube_to_add = o3d.geometry.TriangleMesh.create_box(size[0], size[1], size[2], False, False)
		cube_to_add = cube_to_add.translate(np.array([0, 0, 0]), False) #false translates the mesh center to origin
		cube_to_add = cube_to_add.rotate(box[ROTATION], [0, 0, 0])
		cube_to_add = cube_to_add.translate(box[ORIGIN])

		return cube_to_add

	#Reads the Z values for the center of all boxes and averages them
	#Helper function for placing new boxes around the same level as the road
	def get_depth_average(self):
		z_total = 0
		for box in self.boxes_in_scene:
			box_origin = box.get_center()
			z_total += box_origin[2]
		return z_total/self.box_count

	#Extracts the current data for a selected bounding box
	#returns it as a json object for use in save and export functions
	def create_box_metadata(self, origin, size, rotation, label, confidence, ids, internal_pts, data):
		if isinstance(origin, np.ndarray):
			origin = origin.tolist()
		if isinstance(size, np.ndarray):
			size = size.tolist()
		if isinstance(rotation, np.ndarray):
			rotation = rotation.tolist()
		
		return {
			"origin": origin,
			"size": size,
			"rotation": rotation,
			"annotation": label,
			"confidence": confidence,
			"id": ids,
			"internal_pts": internal_pts,
			"data": data
		}

	#sets horizontal or vertical drag
	def toggle_axis(self, index, opt_name):
		if index == 0:
			self.z_drag = False
		else:
			self.z_drag = True

	def toggle_drag_operation(self):
		if self.drag_operation == True:
			self.current_tool.text = "Rotation"
		else:
			self.current_tool.text = "Translation"
		self.toggle_horiz.visible = not self.toggle_horiz.visible
		self.drag_operation = not self.drag_operation

		self.cw.post_redraw()

	#adapted from lct method, credit to Nicholas Revilla
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
			box = Box(b[0], b[1], Quaternion(b[2]), name=b[3], score=b[4],
					  velocity=(0, 0, 0))
			color = b[5]
			

			# Box is stored in vehicle frame, so transform it to RGB sensor frame
			box.translate(-np.array(self.image_extrinsic['translation']))
			box.rotate(Quaternion(self.image_extrinsic['rotation']).inverse)
			curr_index = self.boxes_to_render.index(b)
			line_weight = 2
			# Thank you to Oscar Beijbom for providing this box rendering algorithm at https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/utils/data_classes.py
			if box_in_image(box, np.asarray(self.image_intrinsic['matrix']), (self.image_w, self.image_h),
							BoxVisibility.ANY):
				# If the box is in view, then render it onto the PLT frame
				corners = view_points(box.corners(), np.asarray(self.image_intrinsic['matrix']), normalize=True)[:2, :]

				# If the box is in view and it is the currently selected box, highlight it
				if curr_index == self.previous_index:
						line_weight = 10
				else:
						line_weight = 2

				def draw_rect(selected_corners, c):
					prev = selected_corners[-1]
					for corner in selected_corners:
						cv2.line(self.image,
								 (int(prev[0]), int(prev[1])),
								 (int(corner[0]), int(corner[1])),
								 c, line_weight)
						prev = corner

				# if b["confidence"] >= 100: # If the box is a ground truth box
				# Draw the sides
				for i in range(4):
					cv2.line(self.image,
							(int(corners.T[i][0]), int(corners.T[i][1])),
							(int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
							color, line_weight)

				# Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
				draw_rect(corners.T[:4], color)
				draw_rect(corners.T[4:], color)
				
				# else: # If the box is a predicted box
				# 	# Draw the sides
				# 	for i in range(4):
				# 		cv2.line(self.image,
				# 				(int(corners.T[i][0]), int(corners.T[i][1])),
				# 				(int(corners.T[i + 4][0]), int(corners.T[i + 4][1])),
				# 				pred_color, line_weight)
						
				# 	# Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
				# 	draw_rect(corners.T[:4], pred_color)
				# 	draw_rect(corners.T[4:], pred_color)

				# Draw line indicating the front
				center_bottom_forward = np.mean(corners.T[2:4], axis=0)
				center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
				cv2.line(self.image,
						 (int(center_bottom[0]), int(center_bottom[1])),
						 (int(center_bottom_forward[0]), int(center_bottom_forward[1])),
						 color, line_weight)

				# might be useful later -- Only render confidence if this isnt at GT box
				# if b["confidence"] < 100 and self.show_score:
				# 	cv2.putText(self.image, str(b["confidence"]), (int(corners.T[0][0]), int(corners.T[1][1])),
				# 				cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

		new_image = o3d.geometry.Image(self.image)
		self.image_widget.update_image(new_image)

		# Force image widget to redraw
		# Post Redraw calls seem to crash the app on windows. Temporary workaround
		self.image_window.post_redraw()

	def update_cam_pos_pcd(self):

		self.scene_widget.scene.remove_geometry("RGB Line")
		# Add Line that indicates current RGB Camera View
		line = o3d.geometry.LineSet()
		line.points = o3d.utility.Vector3dVector([[0, 0, 0], [0, 0, 2]])
		line.lines = o3d.utility.Vector2iVector([[0, 1]])
		line.colors = o3d.utility.Vector3dVector([[1.0, 0, 0]])

		line.rotate(Quaternion(self.image_extrinsic['rotation']).rotation_matrix, [0, 0, 0])
		line.translate(self.image_extrinsic['translation'])

		line.rotate(Quaternion(self.frame_extrinsic['rotation']).rotation_matrix, [0, 0, 0])
		line.translate(self.frame_extrinsic['translation'])

		self.scene_widget.scene.add_geometry("RGB Line", line, self.line_mat)
		self.point_cloud.post_redraw()

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
		self.update_image_path()

	def update_poses(self):
		"""Extracts all the pose data when switching sensors, and or frames
            Args:
                self: window object
            Returns:
                None
                """
		# Pulling intrinsic and extrinsic data from LVT directory based on current selected frame and sensor
		self.image_intrinsic = json.load(
			open(os.path.join(self.lct_path, "cameras", self.rgb_sensor_name, "intrinsics.json")))
		self.image_extrinsic = json.load(
			open(os.path.join(self.lct_path, "cameras", self.rgb_sensor_name, "extrinsics.json")))
		self.frame_extrinsic = json.load(open(os.path.join(self.lct_path, "ego", str(self.frame_num) + ".json")))
		self.update_image()
		self.update_cam_pos_pcd()

	def update_image_path(self):
		"""This updates the image path based on current rgb sensor name and frame number
            Args:
                self: window object
            Returns:
                None
                """
		self.image_path = os.path.join(self.lct_path, "cameras", self.rgb_sensor_name, str(self.frame_num) + ".jpg")
		self.update_poses()

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
			self.frame_select.set_value(self.frame_num)
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
		# if not (self.show_pred and self.show_gt):
		if new_val == "Predicted" and self.pred_frames > 0:
			self.show_pred = True
			self.show_gt = False
			self.update()
		else: # switched to ground truth boxes
			self.show_pred = False
			self.show_gt = True
			self.update()
	
	def jump_to_vehicle(self):
		bounds = self.scene_widget.scene.bounding_box
		self.scene_widget.setup_camera(10, bounds, self.frame_extrinsic['translation'])
		eye = [0,0,0]
		eye[0] = self.frame_extrinsic['translation'][0]
		eye[1] = self.frame_extrinsic['translation'][1]
		eye[2] = 150.0
		self.scene_widget.scene.camera.look_at(self.frame_extrinsic['translation'], eye, [1, 0, 0])
		self.update()

	# deletes the currently selected annotation as well as all its associated data, else nothing happens
	def delete_annotation(self):
		if self.box_selected:
			current_box = self.previous_index
			box_name = self.box_indices[current_box]
			volume_name = self.volume_indices[current_box]

			if self.show_gt:
				self.temp_boxes["boxes"].pop(current_box)
			else:
				self.temp_pred_boxes["boxes"].pop(current_box)
			self.box_indices.pop(current_box)
			self.volume_indices.pop(current_box)
			self.boxes_in_scene.pop(current_box)
			self.volumes_in_scene.pop(current_box)
					
			rendering.Open3DScene.remove_geometry(self.scene_widget.scene, box_name)
			rendering.Open3DScene.remove_geometry(self.scene_widget.scene, volume_name)
			
			self.point_cloud.post_redraw()
			
			self.previous_index = -1
			self.box_selected = None
			self.update_props()
			self.update_poses()

	# creates popup allowing user to add new annotation type
	def add_new_annotation_type(self):
		dialog = gui.Dialog("Create New Annotation")
		em = self.cw.theme.font_size
		margin = gui.Margins(1* em, 1 * em, 1 * em, 1 * em)
		layout = gui.Vert(0, margin)
		button_layout = gui.Horiz()

		text_box_horiz = gui.Horiz()
		self.text_box = gui.TextEdit()
		self.text_box.placeholder_text = "New Label"
		text_box_horiz.add_child(self.text_box)

		buttons_horiz = gui.Horiz(0.50, gui.Margins(0.50, 0.25, 0.50, 0.25))
		submit_button = gui.Button("Submit")
		submit_button.set_on_clicked(self.new_annotation_confirmation)
		cancel_button = gui.Button("Cancel")
		cancel_button.set_on_clicked(self.cw.close_dialog)

		buttons_horiz.add_child(submit_button)
		buttons_horiz.add_fixed(5)
		buttons_horiz.add_child(cancel_button)

		layout.add_child(text_box_horiz)
		layout.add_fixed(10)
		layout.add_child(buttons_horiz)
		dialog.add_child(layout)
		self.cw.show_dialog(dialog)

	# on submit button for add_new_annotation_type, makes updates to combobox
	def new_annotation_confirmation(self):
		# if blank then do nothing
		if len(self.text_box.text_value) == 0:
			return 0
		self.annotation_class.add_item(self.text_box.text_value)
		self.new_annotation_types.append(self.text_box.text_value)
		color_to_add = (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

		self.color_map[self.text_box.text_value] = color_to_add
		self.pred_color_map[self.text_box.text_value] = color_to_add

		self.annotation_class.selected_index = self.annotation_class.number_of_items - 1

		# if a box is currently selected, it becomes the new type
		if self.box_selected is not None:

			self.label_change_handler(self.text_box.text_value, self.annotation_class.number_of_items - 1)
		self.cw.close_dialog()

	# overwrites currently open file with temp_boxes
	def save_changes_to_json(self):
		self.save_check = 1
		self.cw.close_dialog()
		# check current annotation type and save to appropriate folder
		if self.show_gt and not self.show_pred:
			path = os.path.join(self.lct_path ,"bounding", str(self.frame_num), "boxes.json")
			boxes_to_save = {"boxes": [box for box in self.temp_boxes["boxes"]]}
		elif self.show_pred and not self.show_gt:
			path = os.path.join(self.lct_path ,"pred_bounding", str(self.frame_num), "boxes.json")
			boxes_to_save = {"boxes": [box for box in self.temp_pred_boxes["boxes"]]}
		with open(path, "w") as outfile:
			outfile.write(json.dumps(boxes_to_save))

	def save_as(self):
		# opens a file browser to let user select place to save
		file_dialog = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save", self.cw.theme)
		file_dialog.add_filter(".json", "JSON file (.json)")
		file_dialog.set_on_cancel(self.cw.close_dialog)
		file_dialog.set_on_done(self.save_changes_to_json)
		self.cw.show_dialog(file_dialog)

	def save_and_propagate(self):
		# propagates changes to the next frame
		old_gt_boxes_path = os.path.join(self.lct_path ,"bounding", str(self.frame_num), "boxes.json")
		old_pred_boxes_path = os.path.join(self.lct_path ,"pred_bounding", str(self.frame_num), "boxes.json")

		old_pred_boxes = json.load(open(old_pred_boxes_path))
		old_gt_boxes = json.load(open(old_gt_boxes_path))

		new_gt_boxes = [box for box in self.temp_boxes["boxes"] if box not in old_gt_boxes["boxes"]]
		new_pred_boxes = [box for box in self.temp_pred_boxes["boxes"] if box not in old_pred_boxes["boxes"]]

		self.save_changes_to_json()

		self.propagated_gt_boxes = new_gt_boxes
		self.propagated_pred_boxes = new_pred_boxes

		print("propagated gt boxes: ", self.propagated_gt_boxes)
		print("propagated pred boxes: ", self.propagated_pred_boxes)

		new_val = self.frame_num + 1
		self.on_frame_switch(new_val)
		


	# restarts the program in order to exit
	def exit_annotation_mode(self):
		if (self.save_check == 0 and self.temp_boxes != self.old_boxes):
			dialog = gui.Dialog("Confirm Exit")
			em = self.cw.theme.font_size
			margin = gui.Margins(2* em, 1 * em, 2 * em, 2 * em)
			layout = gui.Vert(0, margin)
			button_layout = gui.Horiz()

			layout.add_child(gui.Label("Are you sure you want to exit annotation mode? You have unsaved changes."))
			layout.add_fixed(10)
			confirm_button = gui.Button("Exit")
			back_button = gui.Button("Go Back")

			confirm_button.set_on_clicked(self.confirm_exit)
			back_button.set_on_clicked(self.cw.close_dialog)
			button_layout.add_child(back_button)
			button_layout.add_fixed(5)
			button_layout.add_child(confirm_button)
			layout.add_child(button_layout)
			dialog.add_child(layout)
			self.cw.show_dialog(dialog)
		else:
			self.confirm_exit()


	def confirm_exit(self):
		# point_cloud.close() must be after Window() in order to work, cw.close doesn't matter
		Window(sys.argv[2], self.frame_num)
		self.point_cloud.close()
		self.image_window.close()
		self.cw.close()

	# getters and setters below
	def getCw(self):
		return self.cw
	
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

	def update_pointcloud(self):
		"""Takes new pointcloud data and converts it to global frame, 
			then renders the bounding boxes (Assuming the boxes are vehicle frame
			Args:
				self: window object
			Returns:
				None
				"""
		self.scene_widget.scene.clear_geometry()
		self.boxes_in_scene = []
		self.box_indices = []
		self.volumes_in_scene = []
		self.volume_indices = []
		# Add Pointcloud
		temp_points = np.empty((0,3))
		for label in self.label_list:
			self.scene_widget.remove_3d_label(label)

		self.label_list = []

		for i, pcd_path in enumerate(self.pcd_paths):
			temp_cloud = o3d.io.read_point_cloud(pcd_path)
			ego_rotation_matrix = Quaternion(self.frame_extrinsic['rotation']).rotation_matrix

			# Transform lidar points into global frame
			temp_cloud.rotate(ego_rotation_matrix, [0,0,0])
			temp_cloud.translate(np.array(self.frame_extrinsic['translation']))
			temp_points = np.concatenate((temp_points, np.asarray(temp_cloud.points)))

		self.pointcloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.asarray(temp_points)))
		# Add new global frame pointcloud to our 3D widget
		self.scene_widget.scene.add_geometry("Point Cloud", self.pointcloud, self.coord_frame_mat)
		self.scene_widget.scene.show_axes(True)
		i = 0
		mat = rendering.MaterialRecord()
		mat.shader = "unlitLine"
		mat.line_width = .25

		for box in self.boxes_to_render:
			size = [0,0,0]
			# Open3D wants sizes in L,W,H
			size[0] = box[SIZE][1]
			size[1] = box[SIZE][0]
			size[2] = box[SIZE][2]
			color = box[COLOR]
			bounding_box = o3d.geometry.OrientedBoundingBox(box[ORIGIN], Quaternion(box[ROTATION]).rotation_matrix, size)
			bounding_box.rotate(Quaternion(self.frame_extrinsic['rotation']).rotation_matrix, [0,0,0])
			bounding_box.translate(np.array(self.frame_extrinsic['translation']))
			hex = '#%02x%02x%02x' % color # bounding_box.color needs to be a tuple of floats (color is a tuple of ints)
			bounding_box.color = matplotlib.colors.to_rgb(hex)

			self.box_indices.append(box[ANNOTATION] + str(i)) #used to reference specific boxes in scene
			self.boxes_in_scene.append(bounding_box)

			# might be useful later
			# if box[CONFIDENCE] < 100 and self.show_score:
			# 	label = self.scene_widget.add_3d_label(bounding_box.center, str(box[CONFIDENCE]))
			# 	label.color = gui.Color(1.0,0.0,0.0)
			# 	self.label_list.append(label)

			self.scene_widget.scene.add_geometry(box[ANNOTATION] + str(i), bounding_box, mat)
			i += 1

		# update volumes in the scene
		self.create_box_scene(self.scene_widget, self.boxes_to_render, self.frame_extrinsic)


		#Add Line that indicates current RGB Camera View
		line = o3d.geometry.LineSet()
		line.points = o3d.utility.Vector3dVector([[0,0,0], [0,0,2]])
		line.lines =  o3d.utility.Vector2iVector([[0,1]])
		line.colors = o3d.utility.Vector3dVector([[1.0,0,0]])


		line.rotate(Quaternion(self.image_extrinsic['rotation']).rotation_matrix, [0,0,0])
		line.translate(self.image_extrinsic['translation'])


		line.rotate(Quaternion(self.frame_extrinsic['rotation']).rotation_matrix, [0,0,0])
		line.translate(self.frame_extrinsic['translation'])


		self.scene_widget.scene.add_geometry("RGB Line",line, mat)


		# Force our widgets to update
		self.scene_widget.force_redraw()
		#Post Redraw calls seem to crash the app on windows. Temporary workaround
		if OS_STRING != "Windows":
			self.point_cloud.post_redraw()
	
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
		self.temp_boxes = self.boxes.copy()
		self.temp_boxes["boxes"].extend(self.propagated_gt_boxes)
		self.pred_boxes = json.load(open(os.path.join(self.lct_path , "pred_bounding", str(self.frame_num), "boxes.json")))
		self.temp_pred_boxes = self.pred_boxes.copy()
		self.temp_pred_boxes["boxes"].extend(self.propagated_pred_boxes)

		self.propagated_gt_boxes = [] #reset propagated boxes
		self.propagated_pred_boxes = []
		
		# #If highlight_faults is False, then we just filter boxes
		
		# #If checked, add GT Boxes we should render
		if self.show_gt is True:
			for box in self.boxes['boxes']:
				if box['confidence'] >= self.min_confidence:
					bounding_box = [box['origin'], box['size'], box['rotation'], box['annotation'],
									box['confidence'], self.color_map[box['annotation']]]
					self.boxes_to_render.append(bounding_box)

		#Add Pred Boxes we should render
		if self.show_pred is True:
			if self.pred_frames > 0:
				for box in self.pred_boxes['boxes']:
					if box['confidence'] >= self.min_confidence:
						bounding_box = [box['origin'], box['size'], box['rotation'], box['annotation'], box['confidence'], self.pred_color_map[box['annotation']]]
						self.boxes_to_render.append(bounding_box)


		#Post Redraw calls seem to crash the app on windows. Temporary workaround
		if OS_STRING != "Windows":
			self.cw.post_redraw()

	def update(self):
		""" This updates the window object to reflect the current state
        Args:
            self: window object
        Returns:
            None
            """
		
		self.update_pcd_path()
		self.update_bounding()
		self.update_image_path()
		self.update_pointcloud()

