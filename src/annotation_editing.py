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
	def __init__(self, scene_widget, point_cloud, frame_extrinsic, boxes, boxes_to_render, boxes_in_scene, box_indices, path):
		self.cw = gui.Application.instance.create_window("LCT", 400, 800)
		self.scene_widget = scene_widget
		self.point_cloud = point_cloud
		self.boxes_to_render = boxes_to_render
		self.box_indices = box_indices #name references for bounding boxes in scene
		self.boxes_in_scene = boxes_in_scene
		self.volume_indices = [] #name references for cube volumes in scene
		self.volumes_in_scene = [] #the current cube volume objects in scene
		self.box_selected = ""
		self.previous_index = -1 #-1 denotes, no box selected
		self.box_count = 0
		self.transparent_mat = rendering.MaterialRecord() #invisible material for box volumes
		self.transparent_mat.shader = "defaultLitTransparency"
		self.transparent_mat.base_color = (0.0, 0.0, 0.0, 0.0)
		self.coord_frame = "coord_frame"
		# modify temp boxes in this file, then when it's time to save use them to overwrite existing json
		temp_boxes = boxes.copy()

		#initialize the scene with transparent volumes to allow mouse interactions with boxes
		self.create_box_scene(scene_widget, boxes_to_render, frame_extrinsic)

		# shamelessly stolen from lct setup, cuz their window looks nice
		em = self.cw.theme.font_size
		margin = gui.Margins(0.50 * em, 0.25 * em, 0.50 * em, 0.25 * em)
		layout = gui.Vert(0, margin)

		# button for adding a new bounding box
		add_box_horiz = gui.Horiz()
		add_box_button = gui.Button("Add New Bounding Box")
		#box_partial = functools.partial(self.add_bounding_box, widget=scene_widget, pw=pointcloud, fe=frame_extrinsic) # necessary to add args to functions
		#add_box_button.set_on_clicked(box_partial)
		add_box_button.set_on_clicked(self.place_bounding_box)
		add_box_horiz.add_child(add_box_button)

		# button for saving annotation changes
		# save as button canceled in the name of mvp
		save_annotation_horiz = gui.Horiz()
		save_annotation_button = gui.Button("Save Changes")
		save_partial = functools.partial(self.save_changes_to_json, temp_boxes=temp_boxes, path=path)
		save_annotation_button.set_on_clicked(save_partial)
		save_annotation_horiz.add_child(save_annotation_button)

		# button for exiting annotation mode
		exit_annotation_horiz = gui.Horiz()
		exit_annotation_button = gui.Button("Exit Annotation Mode")
		exit_partial = functools.partial(self.exit_annotation_mode, widget=scene_widget)
		exit_annotation_button.set_on_clicked(exit_partial)
		exit_annotation_horiz.add_child(exit_annotation_button)

		# add various metrics and number thingies used to display info about the current bbox
		click_partial = functools.partial(self.get_point_depth, widget=scene_widget)
		scene_widget.set_on_mouse(click_partial)

		# adding all of the horiz to the vert, in order
		layout.add_child(add_box_horiz)
		layout.add_child(save_annotation_horiz)
		layout.add_child(exit_annotation_horiz)

		self.cw.add_child(layout)
		
		
		
	# function should:
	# disable the current mouse functionality
	# onclick, adds a bounding box at the location of click (or maybe just center screen? idk)
	# highlight the current bounding box
	# re-enable the mouse functionality
	# return the new bounding box
	def add_bounding_box(self, widget, pw, fe):
		partial = functools.partial(self.place_bounding_box, widget=widget, pw=pw, fe=fe)
		widget.set_on_mouse(partial)

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
	def place_bounding_box(self):
		#if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
			# Random values are placeholders until we implement the desired values
			qtr = Quaternion([random.uniform(-0.1,1), random.uniform(-0.1,1), random.uniform(-0.1,1), random.uniform(0,1)]) #Randomized rotation of box
			origin = (self.scene_widget.center_of_rotation[0], self.scene_widget.center_of_rotation[1], self.get_depth_average())
			size = [random.randint(1,5),random.randint(1,5),random.randint(1,5)] #Random dimensions of box
			bbox_params = [origin, size, qtr]
			mat = rendering.MaterialRecord()
			mat.shader = "unlitLine"
			mat.line_width = 0.25
			bounding_box = o3d.geometry.OrientedBoundingBox(origin, qtr.rotation_matrix, size) #Creates bounding box object
			bounding_box.color = matplotlib.colors.to_rgb((0.0,1.0,0))
			volume_to_add = self.add_volume(bbox_params)
			bbox_name = "bbox_" + str(self.box_count)
			volume_name = "volume_" + str(self.box_count)
			self.box_indices.append(bbox_name)
			self.boxes_in_scene.append(bounding_box)
			self.volume_indices.append(volume_name)
			self.volumes_in_scene.append(volume_to_add)
			volume_to_add.compute_vertex_normals()
			self.scene_widget.scene.add_geometry(bbox_name, bounding_box, mat) #Generates a box
			self.scene_widget.scene.add_geometry(volume_name, volume_to_add, self.transparent_mat)

			self.select_box(self.box_count)
			self.scene_widget.force_redraw()
			self.point_cloud.post_redraw()
			print("clicked!")
			self.box_count += 1

			#widget.set_on_mouse(self.enable_mouse)
		#	return gui.Widget.EventCallbackResult.HANDLED

		#return gui.Widget.EventCallbackResult.IGNORED

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
			def get_depth(depth_image):
				x = event.x - widget.frame.x
				y = event.y - widget.frame.y

				depth = np.asarray(depth_image)[y, x]
				world = widget.scene.camera.unproject(
					event.x, event.y, depth, widget.frame.width, widget.frame.height)
				output = "({:.3f}, {:.3f}, {:.3f})".format(
					world[0], world[1], world[2])
				print(output)
				get_nearest(world)

			def get_nearest(world_coords):
				boxes = self.volumes_in_scene
				if len(boxes) != 0:
					smallest_dist = np.linalg.norm(world_coords - boxes[0].get_center())
					closest_box = boxes[0]
					for box in boxes:
						curr_dist = np.linalg.norm(world_coords - box.get_center())
						if curr_dist < smallest_dist:
							smallest_dist = curr_dist
							closest_box = box

					closest_index = boxes.index(closest_box)
					self.box_selected = self.box_indices[closest_index]
					self.select_box(closest_index)

			widget.scene.scene.render_to_depth_image(get_depth)
			return gui.Widget.EventCallbackResult.HANDLED
		return gui.Widget.EventCallbackResult.IGNORED

	#select_box takes a box name (string) and checks to see if a previous box has been selected
	#then it modifies the appropriate line widths to select and deselect boxes
	#it also moves the coordinate frame to the selected box
	def select_box(self, box_index):
		if self.previous_index != -1:  # if not first box clicked "deselect" previous box
			prev_mat = rendering.MaterialRecord()
			prev_mat.shader = "unlitLine"
			prev_mat.line_width = 0.25
			rendering.Open3DScene.modify_geometry_material(self.scene_widget.scene, self.box_indices[self.previous_index],
														   prev_mat)

		rendering.Open3DScene.remove_geometry(self.scene_widget.scene, self.coord_frame)
		self.previous_index = box_index
		box = self.box_indices[box_index]
		origin = o3d.geometry.TriangleMesh.get_center(self.volumes_in_scene[box_index])
		frame = o3d.geometry.TriangleMesh.create_coordinate_frame(1.0, origin)
		frame_mat = rendering.MaterialRecord()
		frame_mat.shader = "defaultLit"
		mat = rendering.MaterialRecord()
		mat.shader = "unlitLine"
		rendering.Open3DScene.modify_geometry_material(self.scene_widget.scene, box, mat)
		self.scene_widget.scene.add_geometry("coord_frame", frame, frame_mat, True)

	#This method adds cube mesh volumes to preexisting bounding boxes
	#Adds an initial coordinate frame to the scene (will probably remove, workaround for assumed bug)
	def create_box_scene(self, scene, boxes, extrinsics):
		coord_frame_mat = rendering.MaterialRecord()
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

	def add_volume(self, box):
		size = [0, 0, 0]
		size[0] = box[SIZE][1]
		size[1] = box[SIZE][0]
		size[2] = box[SIZE][2]

		cube_to_add = o3d.geometry.TriangleMesh.create_box(size[0], size[1], size[2], False, False)
		cube_to_add = cube_to_add.translate(np.array([0, 0, 0]), False)
		cube_to_add = cube_to_add.rotate(Quaternion(box[ROTATION]).rotation_matrix, [0, 0, 0])
		cube_to_add = cube_to_add.translate(box[ORIGIN])

		return cube_to_add
	# overwrites currently open file with temp_boxes
	def get_depth_average(self):
		z_total = 0
		for box in self.boxes_in_scene:
			box_origin = box.get_center()
			z_total += box_origin[2]
		return z_total/self.box_count
	def save_changes_to_json(self, temp_boxes, path):
		with open(path, "w") as outfile:
			outfile.write(json.dumps(temp_boxes))

	# getters and setters below
	def getCw(self):
		return self.cw
		
	# need to discuss this one	
	def currentBox(self):
		return 0

