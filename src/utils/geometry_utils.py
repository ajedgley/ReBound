"""
geometry_utils.py

General use geometry utils
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import math
from pyquaternion import Quaternion

ORIGIN = 0
SIZE = 1
ROTATION = 2
ANNOTATION = 3
CONFIDENCE = 4
COLOR = 5

def translation_and_rotation(transform_matrix):
    """Converts tranformation matrix to a translation and rotation our conversion scripts can use
    Args:
        transform_matrix: 4x4 transform matrix as a numpy array
    
    Returns:
        translation: 1x3 translation vector
        rotation: 1x4 rotation quaternion
    """

    # Extract the first 3 entries in the last column as the translation
    translation = tuple(transform_matrix[:3, -1])

    # Extract the 3x3 rotation matrix from the upper left
    rotation_matrix = transform_matrix[:3, :3]

    # Convert rotation matrix to a quaternion
    quat = R.from_matrix(rotation_matrix)
    
    #Change quaternion from (x,y,z,w) to (w,x,y,z) which is what LVT wants
    rotation = (quat.as_quat()[3], quat.as_quat()[0], quat.as_quat()[1], quat.as_quat()[2])
    
    return translation, rotation

#Returns true if two axis-aligned 3D bounding boxes are overlapping
def is_overlapping(box1, box2):
    #Algorithm taken from https://developer.mozilla.org/en-US/docs/Games/Techniques/3D_collision_detection
    o1 = box1['origin']
    s1 = box1['size']
    o2 = box2['origin']
    s2 = box2['size']
    #loop through x, y, z axes
    for i in range(3):
        a_max = o1[i] + (s1[i]/2)
        a_min = o1[i] - (s1[i]/2)
        b_max = o2[i] + (s2[i]/2)
        b_min = o2[i] - (s2[i]/2)
        if not (a_min <= b_max and a_max >= b_min):
            return False
    return True

def box_dist(box1, box2):
    d_x = (box1['origin'][0] - box2['origin'][0]) ** 2
    d_y = (box1['origin'][1] - box2['origin'][1]) ** 2
    d_z = (box1['origin'][2] - box2['origin'][2]) ** 2
    return math.sqrt(d_x + d_y + d_z)


def compute_vertices(box):
    """Computes the vertex coordinates of a box
    Args:
        box: dictionary representing a bounding box
    """

    # compute the un-rotated displacements
    directions = np.array([
                [1, 1, 1],
                [1, -1, 1], 
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, 1],
                [-1, -1, 1],
                [-1, -1, -1],
                [-1, 1, -1],
            ])

    # swap length and width
    temp = box["size"]
    displacements = np.multiply(directions, np.asarray([temp[1], temp[0], temp[2]])/2)

    # convert the quaternion to a rotation matrix
    wxyz = box["rotation"]
    quat = Quaternion(wxyz)
    rotation = quat.rotation_matrix

    # rotate
    r_displacements = np.matmul(rotation, np.transpose(displacements))

    # compute
    origins = np.array([box["origin"]] * 8)
    vertices = origins + np.transpose(r_displacements)

    return vertices

def compute_interior_points(box, point_cloud):
    """Calculates the number of points interior to the box
    Args: 
        box: dictionary representing a bounding box
        point_cloud: (N, 3) Array representing points to be checked 
    Reference: https://math.stackexchange.com/questions/1472049/check-if-a-point-is-inside-a-rectangular-shaped-area-3d
    """

    # get vertex coords
    vertices = compute_vertices(box)

    # filter points
    min_xyz = np.min(vertices, axis = 0)
    max_xyz = np.max(vertices, axis = 0)

    f_points = []
    for point in point_cloud:
        if point[0] >= min_xyz[0] and point[1] >= min_xyz[1] and point[2] >= min_xyz[2] and point[0] <= max_xyz[0] and point[1] <= max_xyz[1] and point[2] <= max_xyz[2]:
            f_points.append(point)
    if len(f_points) == 0:
        return 0

    # get three corners
    three_corners = np.stack((vertices[6], vertices[3], vertices[1]))

    # choose reference corner
    ref_corner = vertices[2]

    # compute orthogonal edges
    uvw = ref_corner - three_corners

    # compute dot products
    uvw_ref = np.matmul(uvw, ref_corner)
    uvw_corners = np.matmul(uvw, np.transpose(three_corners))
    uvw_points = np.matmul(np.asarray(f_points), np.transpose(uvw))


    # count the points
    count = 0
    for p in uvw_points:
        conditions_met = 0
        for i in range(0, 3):
            if (p[i] >= uvw_ref[i] and p[i] <= uvw_corners[i, i]) or (p[i] <= uvw_ref[i] and p[i] >= uvw_corners[i, i]):
                conditions_met += 1
        if conditions_met == 3:
            count += 1
    return count




    