"""
	functions for editing, we'll see how useful this file is
"""

import open3d.visualization.gui as gui
import functools

# returns created window with all its buttons and whatnot
# maybe not be futureproofed, we'll have to see how nicely it plays with the rest of the functions
def setup_control_window(scene_widget):
	cw = gui.Application.instance.create_window("LCT", 400, 800)
	
	# shamelessly stolen from lct setup, cuz their window looks nice
	em = cw.theme.font_size
	margin = gui.Margins(0.50 * em, 0.25 * em, 0.50 * em, 0.25 * em)
	layout = gui.Vert(0, margin)

	# button for adding a new bounding box
	add_box_horiz = gui.Horiz()
	add_box_button = gui.Button("Add New Bounding Box")
	box_partial = functools.partial(add_bounding_box, widget=scene_widget) # necessary to add args to functions
	add_box_button.set_on_clicked(box_partial)
	add_box_horiz.add_child(add_box_button)

	# button for exiting annotation mode
	exit_annotation_horiz = gui.Horiz()
	exit_annotation_button = gui.Button("Exit Annotation Mode")
	exit_annotation_button.set_on_clicked(exit_annotation_mode)
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

def add_bounding_box(widget):
	widget.set_on_mouse(test)
	#disable_mouse(scene_widget)

# function should:
# close the exiting control panel
# idk, restore the state as if the program just reopened
# potential ideas:
# -just restart the program (hey, it'll probably work)
# -close control window, reopen any previously closed windows, and run update (maybe update done in lct)
def exit_annotation_mode():
	print("TODO")

# disables current mouse functionality, ie dragging screen and stuff
def disable_mouse(scene_widget):
	scene_widget.set_view_controls(scene_widget.FLY)
	print(scene_widget.Controls.ROTATE_CAMERA)
	
# reusable test function, delete for prod
def test(mouseEvent):
	print("test")
	return gui.Widget.EventCallbackResult.IGNORED
