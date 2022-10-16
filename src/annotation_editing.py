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
		self.image_widget = gui.ImageWidget()
		
		self.ew = gui.Application.instance.create_window("Edit", 600, 600)
		#self.cw = gui.Application.instance.create_window("Edit_Controls", 600, 600)
		
		self.ew.add_child(self.image_widget)
		
		# this draws a rectangle! arbitrary points, colors, etc
		corners = [[100,100], [500,100], [500,500], [100,500]]
		color = (200, 200, 200)
		self.draw_rect(corners, color)
		
		new_image = o3d.geometry.Image(self.image)
		self.image_widget.update_image(new_image)
		
		# quit for convenience, doesn't quit the entire app tho
		self.ew.set_on_menu_item_activated(2, gui.Application.instance.quit)
		
	def draw_rect(self, selected_corners, c):
		prev = selected_corners[-1]
		for corner in selected_corners:
			cv2.line(self.image, 
			(int(prev[0]), int(prev[1])),
			(int(corner[0]), int(corner[1])), c, 2)
			prev = corner
