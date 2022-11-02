"""
	functions for editing, we'll see how useful this file is
"""

import open3d.visualization.gui as gui
import functools
import open3d as o3d
from nuscenes.utils.data_classes import Box
import open3d.visualization.rendering as rendering
import matplotlib.colors
from pyquaternion import Quaternion
import numpy as np
import random

ORIGIN = 0
SIZE = 1
ROTATION = 2
ANNOTATION = 3
CONFIDENCE = 4
COLOR = 5

box_count = 1
# returns created window with all its buttons and whatnot
# maybe not be futureproofed, we'll have to see how nicely it plays with the rest of the functions
def setup_control_window(scene_widget, pointcloud, frame_extrinsic):
	cw = gui.Application.instance.create_window("LCT", 400, 800)
	
	# shamelessly stolen from lct setup, cuz their window looks nice
	em = cw.theme.font_size
	margin = gui.Margins(0.50 * em, 0.25 * em, 0.50 * em, 0.25 * em)
	layout = gui.Vert(0, margin)

	# button for adding a new bounding box
	add_box_horiz = gui.Horiz()
	add_box_button = gui.Button("Add New Bounding Box")
	box_partial = functools.partial(add_bounding_box, widget=scene_widget, pw=pointcloud, fe=frame_extrinsic) # necessary to add args to functions
	add_box_button.set_on_clicked(box_partial)
	add_box_horiz.add_child(add_box_button)

	# button for exiting annotation mode
	exit_annotation_horiz = gui.Horiz()
	exit_annotation_button = gui.Button("Exit Annotation Mode")
	exit_partial = functools.partial(exit_annotation_mode, widget=scene_widget)
	exit_annotation_button.set_on_clicked(exit_partial)
	exit_annotation_horiz.add_child(exit_annotation_button)

	# add various metrics and number thingies used to display info about the current bbox

	# adding all of the horiz to the vert, in order
	layout.add_child(add_box_horiz)
	layout.add_child(exit_annotation_horiz)
	
	cw.add_child(layout)
	
	return cw
	
# function should:
# disable the current mouse functionality
# onclick, adds a bounding box at the location of click (or maybe just center screen? idk)
# highlight the current bounding box
# re-enable the mouse functionality
# return the new bounding box

def add_bounding_box(widget, pw, fe):
	partial = functools.partial(place_bounding_box, widget=widget, pw=pw, fe=fe)
	widget.set_on_mouse(partial)

# function should:
# close the exiting control panel
# idk, restore the state as if the program just reopened
# potential ideas:
# -just restart the program (hey, it'll probably work)
# -close control window, reopen any previously closed windows, and run update (maybe update done in lct)
def exit_annotation_mode(widget):
	print("TODO")
	widget.set_on_mouse(enable_mouse)

# onclick, places down a bounding box on the cursor, then reenables mouse functionality
def place_bounding_box(event, widget, pw, fe):
	if event.type == gui.MouseEvent.Type.BUTTON_DOWN:
		# Random values are placeholders until we implement the desired values
		qtr = Quaternion([random.uniform(-0.1,1), random.uniform(-0.1,1), random.uniform(-0.1,1), random.uniform(0,1)]) #Randomized rotation of box
		origin = [random.randint(0,40),random.randint(0,10),random.randint(0,10)] #Random origin of box within existing pointcloud bounds
		size = [random.randint(1,5),random.randint(1,5),random.randint(1,5)] #Random dimensions of box
		mat = rendering.Material()
		
		bounding_box = o3d.geometry.OrientedBoundingBox(origin, qtr.rotation_matrix, size) #Creates bounding box object
		bounding_box.rotate(Quaternion(fe['rotation']).rotation_matrix, [0,0,0]) #Frame extrinsic data necessary to display boxes
		bounding_box.translate(np.array(fe['translation']))
		bounding_box.color = matplotlib.colors.to_rgb((0,0,1)) #Custom bounding boxes are blue to differentiate from the existing ones
		
		global box_count #Ensures that the scene treats each box differently, letting you add multiple
		widget.scene.add_geometry(str(box_count), bounding_box, mat) #Generates a box
		
		widget.force_redraw()
		pw.post_redraw()
		print("clicked!")
		box_count += 1
		
		widget.set_on_mouse(enable_mouse)
		return gui.Widget.EventCallbackResult.CONSUMED
	
	return gui.Widget.EventCallbackResult.CONSUMED

# disables current mouse functionality, ie dragging screen and stuff
def disable_mouse(event):
	return gui.Widget.EventCallbackResult.CONSUMED

# re-enables mouse functionality to their defaults
def enable_mouse(event):
	return gui.Widget.EventCallbackResult.IGNORED
	
