"""
	Adding new functionality to the tool
	Allows users to add/edit existing annotations, then save
	
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
import copy
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, box_in_image, BoxVisibility

from utils import geometry_utils
from utils import testing
from operator import itemgetter
import platform

# new functionality, adding the ability to add/edit existing annotations in a new window
# idk where this is going
class annotations:
	def __init__(self, image):
		print("annotations object created")
		self.image = image
		self.blank_image = copy.deepcopy(self.image)
		self.image_widget = gui.ImageWidget()
		self.dist = 50
		self.width = 400
		self.height = 400
		
		self.ew = gui.Application.instance.create_window("Edit", 600, 600)
		self.cw = gui.Application.instance.create_window("Edit_Controls", 600, 600)
		
		self.ew.add_child(self.image_widget)
		
		# this draws a rectangle! arbitrary points, colors, etc
		self.corners = [[100,100], [500,100], [500,500], [100,500]]
		self.color = (200, 200, 200)
		self.draw_rect()
		
		dist_select = gui.NumberEdit(gui.NumberEdit.INT)
		dist_select.set_limits(1, 1000)
		dist_select.set_on_value_changed(self.update_dist)
		
		new_image = o3d.geometry.Image(self.image)
		self.image_widget.update_image(new_image)
		
		shift_panel = gui.Vert()
		top_horiz = gui.Horiz()
		center_horiz = gui.Horiz()
		bottom_horiz = gui.Horiz()
		shift_panel.add_child(gui.Label("Shift box position"))
		shift_panel.add_child(top_horiz)
		shift_panel.add_child(center_horiz)
		shift_panel.add_child(bottom_horiz)
		shift_panel.add_child(gui.Label(""))
		shift_panel.add_child(gui.Label("Adjust shift increment"))
		shift_panel.add_child(dist_select)
		
		size_panel = gui.Vert()
		size_panel.add_child(gui.Label("Adjust box dimensions"))
		
		width_select_frame = gui.Horiz()
		width_select = gui.NumberEdit(gui.NumberEdit.INT)
		width_select.set_limits(1, 1000)
		width_select.set_on_value_changed(self.update_width)
		width_select_frame.add_child(gui.Label("Box width: "))
		width_select_frame.add_child(width_select)
		
		height_select_frame = gui.Horiz()
		height_select = gui.NumberEdit(gui.NumberEdit.INT)
		height_select.set_limits(1, 1000)
		height_select.set_on_value_changed(self.update_height)
		height_select_frame.add_child(gui.Label("Box height: "))
		height_select_frame.add_child(height_select)
		
		size_panel.add_child(width_select_frame)
		size_panel.add_child(height_select_frame)
		
		control_layout = gui.Horiz()
		control_layout.add_child(shift_panel)
		control_layout.add_child(size_panel)

		self.cw.add_child(control_layout)
		
		left_button = gui.Button("<")
		up_button = gui.Button("^")
		right_button = gui.Button(">")
		down_button = gui.Button("v")
		
		left_button.set_on_clicked(self.shift_left)
		right_button.set_on_clicked(self.shift_right)
		up_button.set_on_clicked(self.shift_up)
		down_button.set_on_clicked(self.shift_down)
		
		center_horiz.add_fixed(20)
		center_horiz.add_child(left_button)
		top_horiz.add_fixed(35)
		top_horiz.add_child(up_button)
		center_horiz.add_child(right_button)
		bottom_horiz.add_fixed(35)
		bottom_horiz.add_child(down_button)
		
		# quit for convenience, doesn't quit the entire app tho
		self.ew.set_on_menu_item_activated(2, gui.Application.instance.quit)
		
	def draw_rect(self):
		self.image = copy.deepcopy(self.blank_image)
		
		prev = self.corners[-1]
		for corner in self.corners:
			cv2.line(self.image, 
			(int(prev[0]), int(prev[1])),
			(int(corner[0]), int(corner[1])), self.color, 2)
			prev = corner
		
		new_image = o3d.geometry.Image(self.image)
		self.image_widget.update_image(new_image)
	
	def shift_left(self):
		for corner in self.corners:
			corner[0] -= self.dist
		
		self.draw_rect()
	
	def shift_right(self):
		for corner in self.corners:
			corner[0] += self.dist
		
		self.draw_rect()
		
	def shift_up(self):
		for corner in self.corners:
			corner[1] -= self.dist
		
		self.draw_rect()
		
	def shift_down(self):
		for corner in self.corners:
			corner[1] += self.dist
		
		self.draw_rect()
	
	def update_dist(self, new_dist):
		if int(new_dist) >= 0 and int(new_dist) < 1000:
			self.dist = new_dist
	
	def update_width(self, new_width):
		if int(new_width) >= 0 and int(new_width) < 1000:
			self.width =  new_width
			
	def update_height(self, new_height):
		if int(new_height) >= 0 and int(new_height) < 1000:
			self.height =  new_height
