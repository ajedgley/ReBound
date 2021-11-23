# LiDAR Conversion Tool
This file documents the structure of the LVT Data Format and how to use the associated utility functions to convert a third-party dataset.


## LVT Format
Everything should be stored in vehicle frame.

**cameras** : contains the RGB images from different cameras
```
lvt/
├── cameras/
│   ├── CAM_FRONT/
│   │   ├── 1.jpg
│   │   ├── 2.jpg
│   │   ├── ...
│   │   ├── extrinsics.json
│   │   └── intrinsics.json    
│   ├── CAM_BACK/
│   │   ├── 1.jpg
│   │   └── 2.jpg
│   │   ├── ...
│   │   ├── extrinsics.json
│   │   └── intrinsics.json   
│   └── ...
```
**pointcloud** : contains PCD files with LiDAR pointclouds for each sensor
```
lvt/
├── pointcloud/
│   ├── LIDAR_TOP/
│   │   ├── 1.pcd
│   │   ├── 2.pcd
│   │   └── ...
│   ├── LIDAR_FRONT/
│   │   ├── 1.pcd
│   │   ├── 2.pcd
│   │   └── ...
│   └── ...
```
**bounding** : contains the ground truth bounding boxes as JSON files
```
lvt/
├── bounding/
│   ├── 1/
│   │   ├── description.json
│   │   └── boxes.json
│   ├── 2/
│   │   ├── description.json
│   │   └── boxes.json
│   └── ...
```
**pred_bounding** : contains a model's predicted bounding boxes as JSON files
```
lvt/
├── pred_bounding/
│   ├── 1/
│   │   ├── description.json
│   │   └── boxes.json
│   ├── 2/
│   │   ├── description.json
│   │   └── boxes.json
│   └── ...
```
**ego** : contains the translation and rotation for each frame as JSON files
```
lvt/
├── ego/
│   ├── 1.json
│   ├── 2.json
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
    * a `path` to the top level of the LVT directory (i.e. `"\home\my_lvt"`)
    * the `name` of the RGB sensor you want to import (i.e. `CAM_FRONT`)
    * the `translation` which is a tuple representing the sensor's translation from the ego (i.e. `(1.5, -0.02, 2.11)`)
    * the `rotation` which is a quaternion tuple representing the sensor's rotation (i.e. `(-0.5, 0.5, -0.5, 0.5)`)
    * the `intrinsic` data which is a 2D List representing 3x3 intrinsic matrix (i.e. `[[2060, 0, 975], [0, 2060, 630], [0, 0, 1]]`)

    The `translation` and `rotation` will be used to create an `extrinsics.json` file while an `intrinsics.json` file will be created using `intrinsic`. 

    If we ran this function for `CAM_FRONT`, the directory would look like this:
    ```
    lvt/
    ├── cameras/
    │   ├── CAM_FRONT/
    │   │   ├── extrinsics.json
    │   │   └── intrinsics.json 
    ```
    Now we can populate the directory using the `add_rgb_frame_from_jpg()` or `add_rgb_frame()` functions. The only difference between these functions is how the RGB data is stored. `add_rgb_frame_from_jpg()` will move an existing JPG image to the correct directory, while `add_rgb_frame()` will create a JPG image from a buffer. These functions take in:
    * a `path` to the top level of the LVT directory (i.e. `"\home\my_lvt"`)
    * the `name` of the RGB sensor you want to import (i.e. `CAM_FRONT`)
    * the `frame_num` which is the number corresponding to the current frame (i.e. `1`)
    * the image to import
        * `add_rgb_frame_from_jpg()` ~ the `input_path` where the JPG image is located (i.e. `/home/Downloads/my_image.jpg`)
        * `add_rgb_frame()` ~ the `image` which is a buffer containing a JPG image (i.e some `PIL Image` object)


    These functions will only add one image to the directory. This means that we'll need to loop through all of the frames to import every image. 

    If we continue the example from above, the directory looks like this:
    ```
    lvt/
    ├── cameras/
    │   ├── CAM_FRONT/
    │   │   ├── 1.jpg
    │   │   ├── 2.jpg
    │   │   ├── ...    
    │   │   ├── extrinsics.json
    │   │   └── intrinsics.json 
    ```

    To complete the import, you would need to repeat this process with every camera which can easily be condensed to a loop.

3. **Import LiDAR Pointclouds**

4. **Import Ground Truth Bounding Boxes**





