import os
import sys
import pandas
from utils import geometry_utils
from utils import dataformat_utils
from pyquaternion import Quaternion

class CurrentState:
    def __init__(self, cam_timestamp, ann_timestamp, frame_num):
        self.cam_timestamp = cam_timestamp
        self.ann_timestamp = ann_timestamp
        self.frame_num = frame_num


# This function will extract and convert the bounding boxes from Argoverse 2's annotations.feather file into the LVT format. 
def convert_annotations(annotations, scene_id, some_path):
    ''' Each row in the annotations.feather file in the Argoverse 2 dataset has the values of 

        Index: The current annotation

        timestamp_ns: The timestamp from when the annotation was created (This annotation is not synced with camera timestamps)

        track_uuid: idk

        category: can be any of "ANIMAL, ARTICULATED_BUS, BICYCLE, BICYCLIST, BOLLARD, BOX_TRUCK, BUS, CONSTRUCTION_BARREL, CONSTRUCTION_CONE, DOG, 
        LARGE_VEHICLE, MESSAGE_BOARD_TRAILER, MOBILE_PEDESTRIAN_CROSSING_SIGN, MOTORCYCLE, MOTORCYCLIS, OFFICIAL_SIGNALER, PEDESTRIAN, RAILED_VEHICLE, 
        REGULAR_VEHICLE, SCHOOL_BUS, SIGN, STOP_SIGN, STROLLER, TRAFFIC_LIGHT_TRAILER, TRUCK, TRUCK_CAB, VEHICULAR_TRAILER, WHEELCHAIR, WHEELED_DEVICE, 
        WHEELED_RIDER

        length_m: length of bounding box
        width_m: width of bounding box
        height_m: height of bounding box

        qw: quaternion w value
        qx: quaternion x value
        qy: quaternion y value
        qz: quaternion z value

        tx_m: translation x from car
        ty_m: translation y from car
        tz_m: translation z from car

        num_interior_pts: idk
    '''
    origins = []
    sizes = []
    rotations = []
    annotation_names = []
    confidences = []

    # Initializes an object containing an invalid timestamps and an initial index of 0
    state = CurrentState(-1, -1, 0)
    # Loops through every row in annotations.feather
    for annotation in annotations.itertuples():

        # New annotation timestamp which corresponds to a new frame that has been found
        # Constructs directory for prevous frame, and initializes new frame data
        if state.ann_timestamp != annotation.timestamp_ns:
            # If on start, nothing needs to be done (THIS CHECK CAN BE AVOIDED IF ANN_TIMESTAMP SET TO FIRST ANN TIMESTAMP, CAN BE DONE BEFORE FOR LOOP)
            if(state.ann_timestamp != -1):
                # Create bounding box directory for current frame
                ''' MAY WANT TO SOMEHOW SAVE TIMESTAMPS '''
                dataformat_utils.create_frame_bounding_directory(some_path, state.frame_num, origins, sizes, rotations, annotation_names, confidences)
                state.frame_num = state.frame_num + 1
                
            state.ann_timestamp = annotation.timestamp_ns
            # honestly not sure how 
            #state.cam_timestamp = get_closest_cam_timestamp(state.ann_timestamp, scene_id)

        ''' MAY NEED TO WORK WITH LENGTH, WIDTH, AND HEIGHT TO FIT PROPERLY'''
        origins.append([annotation.tx_m, annotation.ty_m, annotation.tz_m])
        ''' THESE SHOULD BE FINE'''
        sizes.append([annotation.width_m, annotation.length_m, annotation.height_m])
        ''' THESE SHOULD BE FINE'''
        annotation_names.append(annotation.category)
        ''' IDK HOW TO WORK WITH QUATERNIONS, MAY BE INCORRECT/CORRECT'''
        quat = Quaternion(axis=[annotation.qx, annotation.qy, annotation.qz], radians=annotation.qw)
        rotations.append(quat.q.tolist())
        ''' I BELIEVE THESE ANNOTATIONS ARE GROUND TRUTHS'''
        # Confidence set to 100 by default for ground truth data
        confidences.append(100)
        

#def get_closest_cam_timestamp(timestamp_query, scene_id):
    # For now it only returns the first cam timestamp, as a test
    #print("idk")


fn = sys.argv[1]
output_path = sys.argv[2]
if os.path.exists(fn):
    print(os.path.basename(fn))

    result = pandas.read_feather(fn) # This can probably be passed to multiple function
    convert_annotations(result, "a33a44fb-6008-3dc2-b7c5-2d27b70741e8", output_path)


    
    # Do Something For Each Row

    
'''def extract_bounding(frame, frame_num, lct_path):
    """Extracts the bounding data from a waymo frame and converts it into our intermediate format
    Args:
        frame: waymo frame
        frame_num: frame number
        lct_path: path to LCT directory
    Returns:
        None
        """

    origins = []
    sizes = []
    rotations = []
    annotation_names = []
    annotation_dict = {1: "Vehicle", 2: "Pedestrian", 3: "Sign", 4:"Cyclist"}
    confidences = []

    # Get annotation, rotation, confidence level, quaternion, center, and diminensions of each bounding box in frame
    for label in frame.laser_labels:
        origins.append([label.box.center_x, label.box.center_y, label.box.center_z])
        sizes.append([label.box.width, label.box.length, label.box.height])
        annotation_names.append(annotation_dict[label.type])
        quat = Quaternion(axis=[0.0, 0.0, 1.0], radians=label.box.heading)
        rotations.append(quat.q.tolist())
        # Confidence set to 100 by default for ground truth data
        confidences.append(100)
    dataformat_utils.create_frame_bounding_directory(lct_path, frame_num, origins, sizes, rotations, annotation_names, confidences)'''

'''def extract_bounding(nusc, sample, frame_num, output_path):
    """Extracts the bounding data from a nuScenes frame and converts it into our intermediate format
    Args:
        nusc: NuScenes API object used for obtaining data
        sample: Frame of nuScenes data
        frame_num: Number corresponding to sample
        output_path: Path to generic data format directory
    Returns:
        None
        """
    origins = []
    sizes = []
    rotations = []
    annotation_names = []
    confidences = []
    
    # Get translation, rotation, dimensions, and origins for bounding boxes for each annotation
    for i in range(0, len(sample['anns']) - 1):
        token = sample['anns'][i]
        annotation_metadata = nusc.get('sample_annotation', token)
        # Create nuscenes box object so we can easily transform this box to the vehicle frame that our dataset requires
        box = Box(annotation_metadata['translation'], annotation_metadata['size'], Quaternion(annotation_metadata['rotation']))

        # Get ego pose information. The LIDAR sensor has the ego information, so we can use that.
        sensor = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        poserecord = nusc.get('ego_pose', sensor['ego_pose_token'])
        
        #Transform the boxes from global frame to vehicle frame
        box.translate(-np.array(poserecord['translation']))
        box.rotate(Quaternion(poserecord['rotation']).inverse)

        # Store data obtained from annotation
        origins.append(box.center.tolist())
        sizes.append(annotation_metadata['size'])
        rotations.append(box.orientation.q.tolist())
        annotation_names.append(annotation_metadata['category_name'])

        # Confidence for ground truth data is always 100
        confidences.append(100)
        
    dataformat_utils.create_frame_bounding_directory(output_path, frame_num, origins, sizes, rotations, annotation_names, confidences)'''

'''def get_closest_cam_channel_timestamp(self, lidar_timestamp: int, log_id: str) -> Optional[int]:
        """Given a LiDAR timestamp, find the synchronized corresponding image timestamp for a particular camera.

        This image timestamp should have the closest absolute timestamp.

        Args:
            lidar_timestamp: integer
            cam_name: string, representing path to log directories
            log_id: string

        Returns:
            closest_cam_ch_timestamp: closest timestamp
        """
        if log_id not in self.per_log_cam_timestamps_index or cam_name not in self.per_log_cam_timestamps_index[log_id]:
            return None

        cam_timestamps = self.per_log_cam_timestamps_index[log_id][cam_name]
        # catch case if no files were loaded for a particular sensor
        if not cam_timestamps.tolist():
            return None

        closest_cam_ch_timestamp, timestamp_diff = find_closest_integer_in_ref_arr(lidar_timestamp, cam_timestamps)
        if timestamp_diff > self.MAX_LIDAR_RING_CAM_TIMESTAMP_DIFF and cam_name in list(RingCameras):
            # convert to nanoseconds->milliseconds for readability
            logger.warning(
                "No corresponding ring image at %s: %.1f > %s ms",
                lidar_timestamp,
                to_metric_time(ts=timestamp_diff, src=Nanosecond, dst=Millisecond),
                to_metric_time(ts=self.MAX_LIDAR_RING_CAM_TIMESTAMP_DIFF, src=Nanosecond, dst=Millisecond),
            )
            return None
        elif timestamp_diff > self.MAX_LIDAR_STEREO_CAM_TIMESTAMP_DIFF and cam_name in list(StereoCameras):
            # convert to nanoseconds->milliseconds for readability
            logger.warning(
                "No corresponding stereo image at %s: %.1f > %s ms",
                lidar_timestamp,
                to_metric_time(ts=timestamp_diff, src=Nanosecond, dst=Millisecond),
                to_metric_time(ts=self.MAX_LIDAR_STEREO_CAM_TIMESTAMP_DIFF, src=Nanosecond, dst=Millisecond),
            )
            return None
        return closest_cam_ch_timestamp'''













    

    
   

        
