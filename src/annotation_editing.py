"""
	functions for editing, we'll see how useful this file is
"""

import open3d.visualization.gui as gui
import open3d.visualization.rendering as rendering
import open3d as o3d
import functools
import open3d as o3d
from nuscenes.utils.data_classes import Box
import open3d.visualization.rendering as rendering
import matplotlib.colors
import numpy as np
from pyquaternion import Quaternion
import random
import os
import sys
import json
import lct

ORIGIN = 0
SIZE = 1
ROTATION = 2
ANNOTATION = 3
CONFIDENCE = 4
COLOR = 5


class Annotation:
	# returns created window with all its buttons and whatnot
	def __init__(self, scene_widget, point_cloud, frame_extrinsic, boxes, boxes_to_render, boxes_in_scene, box_indices, annotation_types, path):
		self.cw = gui.Application.instance.create_window("LCT", 400, 800)
		self.scene_widget = scene_widget
		self.point_cloud = point_cloud
		self.all_pred_annotations = annotation_types
		self.boxes_to_render = boxes_to_render #list of box metadata in scene
		self.box_indices = box_indices #name references for bounding boxes in scene
		self.boxes_in_scene = boxes_in_scene #current bounding box objects in scene
		self.volume_indices = [] #name references for cube volumes in scene
		self.volumes_in_scene = [] #the current cube volume objects in scene
		
		self.box_selected = None
		self.previous_index = -1 #-1 denotes, no box selected
		#used to generate unique ids for boxes and volumes
		self.box_count = 0
		self.transparent_mat = rendering.Material() #invisible material for box volumes
		self.transparent_mat.shader = "defaultLitTransparency"
		self.transparent_mat.base_color = (0.0, 0.0, 0.0, 0.0)
		self.coord_frame = "coord_frame"
		self.z_drag = False
		
		# modify temp boxes in this file, then when it's time to save use them to overwrite existing json
		self.temp_boxes = boxes.copy()

		#initialize the scene with transparent volumes to allow mouse interactions with boxes
		self.create_box_scene(scene_widget, boxes_to_render, frame_extrinsic)

		# shamelessly stolen from lct setup, cuz their window looks nice
		em = self.cw.theme.font_size
		margin = gui.Margins(0.50 * em, 0.25 * em, 0.50 * em, 0.25 * em)
		layout = gui.Vert(0, margin)

		# button for adding a new bounding box
		add_box_horiz = gui.Horiz()
		add_box_button = gui.Button("Add New Bounding Box")
		toggle_axis_button = gui.Button("Toggle Vertical/Horizontal")
		add_box_button.set_on_clicked(self.add_bounding_box)
		toggle_axis_button.set_on_clicked(self.toggle_axis)
		add_box_horiz.add_child(add_box_button)
		add_box_horiz.add_child(toggle_axis_button)
		
		# buttons for saving/saving as annotation changes
		save_annotation_horiz = gui.Horiz()
		save_annotation_button = gui.Button("Save Changes")
		save_partial = functools.partial(self.save_changes_to_json, path=path)
		save_annotation_button.set_on_clicked(save_partial)
		
		save_as_button = gui.Button("Save As")
		save_as_button.set_on_clicked(self.save_as)
		
		save_annotation_horiz.add_child(save_annotation_button)
		save_annotation_horiz.add_child(save_as_button)

		# button for exiting annotation mode
		exit_annotation_horiz = gui.Horiz()
		exit_annotation_button = gui.Button("Exit Annotation Mode")
		exit_partial = functools.partial(self.exit_annotation_mode, widget=scene_widget)
		exit_annotation_button.set_on_clicked(exit_partial)
		exit_annotation_horiz.add_child(exit_annotation_button)
		
		# add selected box info tracking here?
		
		# deletes bounding box, should only be enabled if a bounding box is selected
		delete_annotation_horiz = gui.Horiz()
		delete_annotation_button = gui.Button("Delete Annotation")
		delete_annotation_button.set_on_clicked(self.delete_annotation)
		delete_annotation_horiz.add_child(delete_annotation_button)

		# empty horiz, just cuz i think exit_annotation looks better at the bottom
		empty_horiz = gui.Horiz()

		# adding all of the horiz to the vert, in order
		layout.add_child(add_box_horiz)
		layout.add_child(save_annotation_horiz)
		layout.add_child(delete_annotation_horiz)
		
		layout.add_child(empty_horiz)
		layout.add_child(exit_annotation_horiz)

		self.cw.add_child(layout)
		
		# Event handlers
		
		# sets up onclick box selection
		click_partial = functools.partial(self.get_point_depth, widget=scene_widget)
		scene_widget.set_on_mouse(click_partial)

		# sets up keyboard event handling
		key_partial = functools.partial(self.key_event_handler, widget=scene_widget)
		scene_widget.set_on_key(key_partial)


	# Intermediate helper function that allows a user to select an annotation type from a dropdown list
	# Picks both the bounding box color and the attached annotation type
	def add_bounding_box(self):

		dialog = gui.Dialog("Select Annotation")
		em = self.cw.theme.font_size
		margin = gui.Margins(0.50 * em, 0.25 * em, 0.50 * em, 0.25 * em)
		layout = gui.Vert(0, margin)
		layout.add_child(gui.Label("Select an Annotation Type:"))
		annotation_dropdown = gui.Combobox()
		for annotation in self.all_pred_annotations:
			annotation_dropdown.add_item(annotation)
		layout.add_child(annotation_dropdown)
		button_layout = gui.Horiz()
		select_button = gui.Button("Select")
		cancel_button = gui.Button("Cancel")

		select_button.set_on_clicked(self.place_bounding_box(annotation_dropdown.selected_text))
		cancel_button.set_on_clicked(self.cw.close_dialog())
		button_layout.add_child(select_button)
		button_layout.add_child(cancel_button)
		layout.add_child(button_layout)
		dialog.add_child(layout)
		self.cw.show_dialog(dialog)
		
	# function should:
	# close the exiting control panel
	# idk, restore the state as if the program just reopened
	# potential ideas:
	# -just restart the program (hey, it'll probably work)
	# -close control window, reopen any previously closed windows, and run update (maybe update done in lct)
	def exit_annotation_mode(self, widget):
		# os.execl(sys.executable, os.path.abspath(__file__), *sys.argv), this doesn't work but maybe it's an idea
		print("TODO")
		widget.set_on_mouse(self.enable_mouse)

	# onclick, places down a bounding box on the cursor, then reenables mouse functionality
	def place_bounding_box(self, annotation):
		# Random values are placeholders until we implement the desired values
		qtr = Quaternion([random.uniform(-0.1,1), random.uniform(-0.1,1), random.uniform(-0.1,1), random.uniform(0,1)]) #Randomized rotation of box
		origin = (self.scene_widget.center_of_rotation[0], self.scene_widget.center_of_rotation[1], self.get_depth_average())
		size = [random.randint(1,5),random.randint(1,5),random.randint(1,5)] #Random dimensions of box
		bbox_params = [origin, size, qtr] #create_volume uses box meta data to create mesh

		mat = rendering.Material()
		mat.shader = "unlitLine"
		mat.line_width = 0.25

		bounding_box = o3d.geometry.OrientedBoundingBox(origin, qtr.rotation_matrix, size) #Creates bounding box object
		bounding_box.color = matplotlib.colors.to_rgb((0.0,1.0,0)) #will select color from annotation type list
		bbox_name = "bbox_" + str(self.box_count)
		self.box_indices.append(bbox_name)
		self.boxes_in_scene.append(bounding_box)

		volume_to_add = self.add_volume(bbox_params)
		volume_name = "volume_" + str(self.box_count)
		self.volume_indices.append(volume_name)
		self.volumes_in_scene.append(volume_to_add)
		volume_to_add.compute_vertex_normals()

		self.scene_widget.scene.add_geometry(bbox_name, bounding_box, mat) #Adds the box to the scene
		self.scene_widget.scene.add_geometry(volume_name, volume_to_add, self.transparent_mat)#Adds the volume
		self.select_box(self.box_count) #make the new box the currently selected box

		self.scene_widget.force_redraw()
		self.point_cloud.post_redraw()
		print("clicked!")
		self.box_count += 1

	# disables current mouse functionality, ie dragging screen and stuff
	def disable_mouse(self, event):
		return gui.Widget.EventCallbackResult.CONSUMED

	# re-enables mouse functionality to their defaults
	def enable_mouse(self, event):
		return gui.Widget.EventCallbackResult.IGNORED

	#Takes the frame x and y coordinates and flattens the 3D scene into a 2D depth image
	#The X and Y coordinates select the depth value from the depth image and converts it into a depth value
	#After getting the coordinates, it automatically calls the closest distance function
	def get_point_depth(self, event, widget):
		if event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.CTRL):
			def get_depth(depth_image): #gets world coords from mouse click
				x = event.x - widget.frame.x
				y = event.y - widget.frame.y

				depth = np.asarray(depth_image)[y, x] #flatten image to depth image, get color value from x,y point
				world = widget.scene.camera.unproject(
					event.x, event.y, depth, widget.frame.width, widget.frame.height)
				output = "({:.3f}, {:.3f}, {:.3f})".format(
					world[0], world[1], world[2])
				#print(output)
				get_nearest(world)

			def get_nearest(world_coords): #searches boxes in the scene for shortest dist
				boxes = self.volumes_in_scene
				if len(boxes) != 0:
					smallest_dist = np.linalg.norm(world_coords - boxes[0].get_center())
					closest_box = boxes[0]
					for box in boxes:
						curr_dist = np.linalg.norm(world_coords - box.get_center())
						if curr_dist < smallest_dist:
							smallest_dist = curr_dist
							closest_box = box

					closest_index = boxes.index(closest_box) #get the array position of closest_box
					self.box_selected = self.box_indices[closest_index]
					self.select_box(closest_index) #select the nearest box

			widget.scene.scene.render_to_depth_image(get_depth)
			return gui.Widget.EventCallbackResult.HANDLED

		#So Open3D only passes one event to one event handler so the drag function gets moved here
		elif event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(gui.KeyModifier.SHIFT): 
			current_box = self.previous_index
			scene_camera = self.scene_widget.scene.camera
			box_to_drag = self.boxes_in_scene[current_box]
			box_name = self.box_indices[current_box]
			volume_to_drag = self.volumes_in_scene[current_box]
			volume_name = self.volume_indices[current_box]
			mat = rendering.Material()
			mat.shader = "unlitLine"
			curr_x = 0.0
			curr_y = 0.0
			scaling_factor = 0.001
			if event.is_button_down(
					gui.MouseButton.RIGHT) and event.type == gui.MouseEvent.Type.BUTTON_DOWN and event.is_modifier_down(
					gui.KeyModifier.SHIFT):
				curr_x = event.x
				curr_y = event.y
				print(curr_x, curr_y)

			elif event.is_button_down(
					gui.MouseButton.RIGHT) and event.type == gui.MouseEvent.Type.DRAG and event.is_modifier_down(
					gui.KeyModifier.SHIFT):
				rendering.Open3DScene.remove_geometry(self.scene_widget.scene, box_name)
				rendering.Open3DScene.remove_geometry(self.scene_widget.scene, volume_name)
				x_diff = (event.x - curr_x) * scaling_factor
				y_diff = (event.y - curr_y) * scaling_factor
				print(x_diff, y_diff)
				if self.z_drag:  # if z_drag is on
					box_to_drag.translate((0, 0, y_diff))
					volume_to_drag.translate((0, 0, y_diff))
				else:
					box_to_drag.translate((x_diff, y_diff, 0))
					volume_to_drag.translate((x_diff, y_diff, 0))
				self.scene_widget.scene.add_geometry(box_name, box_to_drag, mat)
				self.scene_widget.scene.add_geometry(volume_name, volume_to_drag, self.transparent_mat)
				curr_x = event.x
				curr_y = event.y
			return gui.Widget.EventCallbackResult.CONSUMED
		return gui.Widget.EventCallbackResult.IGNORED

	#select_box takes a box name (string) and checks to see if a previous box has been selected
	#then it modifies the appropriate line widths to select and deselect boxes
	#it also moves the coordinate frame to the selected box
	def select_box(self, box_index):
		if self.previous_index != -1:  # if not first box clicked "deselect" previous box
			prev_mat = rendering.Material()
			prev_mat.shader = "unlitLine"
			prev_mat.line_width = 0.25 #return line_width to normal
			rendering.Open3DScene.modify_geometry_material(self.scene_widget.scene, self.box_indices[self.previous_index],
														   prev_mat)

		rendering.Open3DScene.remove_geometry(self.scene_widget.scene, self.coord_frame)
		self.previous_index = box_index
		box = self.box_indices[box_index]
		origin = o3d.geometry.TriangleMesh.get_center(self.volumes_in_scene[box_index])
		frame = o3d.geometry.TriangleMesh.create_coordinate_frame(1.0, origin)
		frame_mat = rendering.Material()
		frame_mat.shader = "defaultLit"
		mat = rendering.Material()
		mat.shader = "unlitLine" #default linewidth is 1.0, makes box look highlighted
		rendering.Open3DScene.modify_geometry_material(self.scene_widget.scene, box, mat)
		self.scene_widget.scene.add_geometry("coord_frame", frame, frame_mat, True)

	#This method adds cube mesh volumes to preexisting bounding boxes
	#Adds an initial coordinate frame to the scene
	def create_box_scene(self, scene, boxes, extrinsics):
		coord_frame_mat = rendering.Material()
		coord_frame_mat.shader = "defaultLit"
		frame_to_add = o3d.geometry.TriangleMesh.create_coordinate_frame()
		scene.scene.add_geometry("coord_frame", frame_to_add, coord_frame_mat, False)
		for box in boxes:
			volume_to_add = self.add_volume(box)
			volume_to_add = volume_to_add.rotate(Quaternion(extrinsics['rotation']).rotation_matrix, [0, 0, 0])
			volume_to_add = volume_to_add.translate(np.array(extrinsics['translation']))
			cube_id = "volume_" + str(self.box_count)
			self.volume_indices.append(cube_id)

			volume_to_add.compute_vertex_normals()
			self.volumes_in_scene.append(volume_to_add)
			self.scene_widget.scene.add_geometry(cube_id, volume_to_add, self.transparent_mat)
			self.box_count += 1

		self.point_cloud.post_redraw()
		
	def key_event_handler(self, event, widget):
		# delete button handler
		if event.key == 127 and event.type == event.Type.DOWN:
			self.delete_annotation()
			return gui.Widget.EventCallbackResult.CONSUMED
		
		return gui.Widget.EventCallbackResult.IGNORED

	#general cube_mesh function to create cube mesh from bounding box information
	#positions cube mesh at center of bounding box allowing the boxes to be selectable
	def add_volume(self, box):
		size = [0, 0, 0]
		size[0] = box[SIZE][1]
		size[1] = box[SIZE][0]
		size[2] = box[SIZE][2]

		cube_to_add = o3d.geometry.TriangleMesh.create_box(size[0], size[1], size[2], False, False)
		cube_to_add = cube_to_add.translate(np.array([0, 0, 0]), False) #false translates the mesh center to origin
		cube_to_add = cube_to_add.rotate(Quaternion(box[ROTATION]).rotation_matrix, [0, 0, 0])
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
	def create_box_metadata(self, box):
		ret_val = ""
		return False

	#takes a drag event and converts it into a translation


	#toggles horizontal or vertical drag
	def toggle_axis(self):
		self.z_drag = not self.z_drag

	# deletes the currently selected annotation as well as all its associated data, else nothing happens
	def delete_annotation(self):
		if self.box_selected:
			current_box = self.previous_index
			box_name = self.box_indices[current_box]
			volume_name = self.volume_indices[current_box]

			self.temp_boxes["boxes"].pop(current_box)
			self.box_indices.pop(current_box)
			self.volume_indices.pop(current_box)
			self.boxes_in_scene.pop(current_box)
			self.box_indices.pop(current_box)
					
			rendering.Open3DScene.remove_geometry(self.scene_widget.scene, box_name)
			rendering.Open3DScene.remove_geometry(self.scene_widget.scene, volume_name)
			
			self.point_cloud.post_redraw()

	
	# overwrites currently open file with temp_boxes
	def save_changes_to_json(self, path):
		self.cw.close_dialog()
		with open(path, "w") as outfile:
			outfile.write(json.dumps(self.temp_boxes))

	def save_as(self):
		# pops open one of those nifty file browsers to let user select place to save
		file_dialog = gui.FileDialog(gui.FileDialog.SAVE, "Choose file to save", self.cw.theme)
		file_dialog.add_filter(".json", "JSON file (.json)")
		file_dialog.set_on_cancel(self.cw.close_dialog)
		file_dialog.set_on_done(self.save_changes_to_json)
		self.cw.show_dialog(file_dialog)

	# getters and setters below
	def getCw(self):
		return self.cw
		
	# need to discuss this one	
	def currentBox(self):
		return 0

