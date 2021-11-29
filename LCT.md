# LiDAR Conversion Tool
This file documents the structure of the LCT Data Format and how to use the associated utility functions to convert a third-party dataset.


## LCT Format
Everything should be stored in vehicle frame.

**cameras** : contains the RGB images from different cameras
```
lct/
├── cameras/
│   ├── CAM_FRONT/
│   │   ├── 0.jpg
│   │   ├── 1.jpg
│   │   ├── ...
│   │   ├── extrinsics.json
│   │   └── intrinsics.json    
│   ├── CAM_BACK/
│   │   ├── 0.jpg
│   │   └── 1.jpg
│   │   ├── ...
│   │   ├── extrinsics.json
│   │   └── intrinsics.json   
│   └── ...
```
**pointcloud** : contains PCD files with LiDAR pointclouds for each sensor
```
lct/
├── pointcloud/
│   ├── LIDAR_TOP/
│   │   ├── 0.pcd
│   │   ├── 1.pcd
│   │   └── ...
│   ├── LIDAR_FRONT/
│   │   ├── 0.pcd
│   │   ├── 1.pcd
│   │   └── ...
│   └── ...
```
**bounding** : contains the ground truth bounding boxes as JSON files
```
lct/
├── bounding/
│   ├── 0/
│   │   ├── description.json
│   │   └── boxes.json
│   ├── 1/
│   │   ├── description.json
│   │   └── boxes.json
│   └── ...
```
**pred_bounding** : contains a model's predicted bounding boxes as JSON files
```
lct/
├── pred_bounding/
│   ├── 0/
│   │   ├── description.json
│   │   └── boxes.json
│   ├── 1/
│   │   ├── description.json
│   │   └── boxes.json
│   └── ...
```
**ego** : contains the translation and rotation for each frame as JSON files
```
lct/
├── ego/
│   ├── 0.json
│   ├── 1.json
│   └── ...
```

## Converting A Third-Party Dataset
Now, we'll go through how to convert a third-party dataset using the functions in `utils.py`.

1. **Create a top-level directory**
    
    You can use the function `create_lct_diretory()` to create empty directories. The function takes in:
    * a `path` for where to place the new directory
    * a `name` for the name for the top directory

2. **Import RGB Images**

    You can use `create_rgb_sensor_directory()` function to populate the top-level of the `cameras` directory. The function takes in:
    * a `path` to the top level of the LCT directory (i.e. `"\home\my_lct"`)
    * the `name` of the RGB sensor you want to import (i.e. `CAM_FRONT`)
    * the `translation` which is a (x, y, z) tuple representing the sensor's translation from the ego (i.e. `(1.5, -0.02, 2.11)`)
    * the `rotation` which is a (w, x, y, z) quaternion tuple representing the sensor's rotation (i.e. `(-0.5, 0.5, -0.5, 0.5)`)
    * the `intrinsic` data which is a 2D List representing 3x3 intrinsic matrix (i.e. `[[2060, 0, 975], [0, 2060, 630], [0, 0, 1]]`)

    The `translation` and `rotation` will be used to create an `extrinsics.json` file while an `intrinsics.json` file will be created using `intrinsic`. 

    If we ran this function for `CAM_FRONT`, the directory would look like this:
    ```
    lct/
    ├── cameras/
    │   ├── CAM_FRONT/
    │   │   ├── extrinsics.json
    │   │   └── intrinsics.json 
    ```
    Now we can populate the directory using the `add_rgb_frame_from_jpg()` or `add_rgb_frame()` functions. The only difference between these functions is how the RGB data is stored. `add_rgb_frame_from_jpg()` will move an existing JPG image to the correct directory, while `add_rgb_frame()` will create a JPG image from a buffer. These functions take in:
    * a `path` to the top level of the LCT directory (i.e. `"\home\my_lct"`)
    * the `name` of the RGB sensor you want to import (i.e. `CAM_FRONT`)
    * the `frame_num` which is the number corresponding to the current frame (i.e. `0`)
    * the image to import
        * `add_rgb_frame_from_jpg()` ~ the `input_path` where the JPG image is located (i.e. `/home/Downloads/my_image.jpg`)
        * `add_rgb_frame()` ~ the `image` which is a buffer containing a JPG image (i.e some `PIL Image` object)


    These functions will only add one image to the directory. This means that we'll need to loop through all of the frames to import every image. 

    If we continue the example from above, the directory looks like this:
    ```
    lct/
    ├── cameras/
    │   ├── CAM_FRONT/
    │   │   ├── 0.jpg
    │   │   ├── 1.jpg
    │   │   ├── ...    
    │   │   ├── extrinsics.json
    │   │   └── intrinsics.json 
    ```

    To complete the import, you would need to repeat this process with every camera which can easily be condensed to a loop.

3. **Import LiDAR Pointclouds**

    You can use the `create_lidar_sensor_directory()` function to create an empty directory for a specific sensor. The function takes in:
    * a `path` to the top level of the LCT directory (i.e. `"\home\my_lct"`)
    * the `name` of the LiDAR sensor you want to import (i.e. `LIDAR_TOP`)

    Like before, you can populate the using two different functions: `add_lidar_frame_from_pcd()` or `add_lidar_frame()`. `add_lidar_frame_from_pcd()` will move existing PCD files to the correct directory while `add_lidar_frame()` will convert a list of tuples into PCD files. The goal is to get all of the pointcloud data into PCD files. Both functions take in:

    * a `path` to the top level of the LCT directory (i.e. `"\home\my_lct"`)
    * the `name` of the LiDAR sensor you want to import (i.e. `LIDAR_TOP`)
    * the `frame_num` which is the number corresponding to the current frame (i.e. `0`)
    * the pointcloud data
        * `add_lidar_frame_from_pcd()` ~ the `input_path` where the PCD file is located (i.e. `/home/Downloads/my_pcd.pcd`)
        * `add_lidar_frame()`
            * the `points` which is a list of (x, y, z) tuples representing x,y,z coordinates (i.e. `[(1, 2, 5), (3, 4, 6), ...]`)
            * the `translation` which is a tuple representing the sensor's translation from the ego (i.e. `(1.5, -0.02, 2.11)`)
            * the `rotation` which is a quaternion tuple representing the sensor's rotation (i.e. `(-0.5, 0.5, -0.5, 0.5)`)
    
    As above, these functions will only add one PCD file to the directory, so you would need to repeat this for each frame and then for each sensor.

4. **Import Ground Truth Bounding Boxes**

    You can use the `create_frame_bounding_directory()` function to populate the `bounding` directory. The function takes in:
    * a `path` to the top level of the LCT directory (i.e. `"\home\my_lct"`)
    * the `frame_num` which is the number corresponding to the current frame (i.e. `0`)
    * the `origins` which is a list of (x, y, z) tuples that represent the center of each box (i.e. `[(1, 2, 5), (3, 4, 6), ...]`)
    * the `sizes` of each box as a list of (L, W, H) tuples that represent the width, length, and height (i.e. `[(1, 2, 5), (3, 4, 6), ...]`)
    * the `rotations` of each box with respect to (0, 0, 0) as a list of quaternion tuples (i.e. (`[(-0.5, 0.5, -0.5, 0.5), (-0.4, 0.4, -0.3, 0.4), ...]`))
    * the `annotation_names` for each boxe as a list of strings
    * the `confidence` which is a list of integers from 0-100 representing the confidence for each box --> should be 100 since this is ground truth data

5. **(Optional) Import Predicted Bounding Boxes**

    You reuse the `create_frame_bounding_directory()` function with an extra argument to import into the `pred_bounding` directory. The function takes in:
    * a `path` to the top level of the LCT directory (i.e. `"\home\my_lct"`)
    * the `frame_num` which is the number corresponding to the current frame (i.e. `0`)
    * the `origins` which is a list of (x, y, z) tuples that represent the center of each box (i.e. `[(1, 2, 5), (3, 4, 6), ...]`)
    * the `sizes` of each box as a list of (L, W, H) tuples that represent the width, length, and height (i.e. `[(1, 2, 5), (3, 4, 6), ...]`)
    * the `rotations` of each box with respect to (0, 0, 0) as a list of quaternion tuples (i.e. (`[(-0.5, 0.5, -0.5, 0.5), (-0.4, 0.4, -0.3, 0.4), ...]`))
    * the `annotation_names` for each boxe as a list of strings
    * the `confidence` which is a list of integers from 0-100 representing the confidence for each box
    * a `predicted` boolean that flags if this list of boxes should go into the pred_bounding directory

6. **Import Ego Data**

    The ego is the location of the car relative to the global frame. You can use the `create_ego_directory()`
    * a `path` to the top level of the LCT directory (i.e. `"\home\my_lct"`)
    * the `frame` which is the number corresponding to the current frame (i.e. `0`)
    * the `translation` which is a [x, y, z] list representing the sensor's translation from the ego (i.e. `[1.5, -0.02, 2.11]`)
    * the `rotation` which is a [w, x, y, z] quaternion list representing the sensor's rotation (i.e. `[-0.5, 0.5, -0.5, 0.5]`)

