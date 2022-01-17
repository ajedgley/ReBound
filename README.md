# LiDAR Visualization Tool

## Getting Started
### Prerequistes
LiDAR Visualization Tool runs on a Linux system, so you'll need to set up a system if you have a different operating system.
We have 2 ways for you to set up a system.

*You can skip this section if you already have Linux.*
1. **Dual-boot your system.**
    
    Here is a [tutorial](https://linuxconfig.org/how-to-install-ubuntu-20-04-alongside-windows-10-dual-boot) you can follow to set this up.
2. **Set-up a GUI-based virtual machine.**

    We used [VMware](https://www.vmware.com/) to set-up our virtual machine (**Note**: [VirtualBox](https://www.virtualbox.org/) will not be able to convert Waymo datasets due to lack of support for [Tensorflow](https://www.tensorflow.org/)). 
    Check out this [tutorial](https://unixcop.com/how-to-install-ubuntu-21-04-on-vmware-workstation-pro/) that can help with installation; however, here are some notes:
    * We used [Ubuntu 20.04](https://releases.ubuntu.com/20.04/) since it includes the correct verion of Python (Python 3.8)
    * We recommend allocating at least 30 GB of hard disk space and 4 GB of memory, but you may need more

### Cloning the Repository
**Note**: if any command doesn't exist, run `sudo apt-get install <command>` for that command

Execute these commands on your Linux system.
```sh
git clone https://para.cs.umd.edu/neehar/lidar.git
```

### Installing Dependencies
**Note**: if any command doesn't exist, run `sudo apt-get install <command>` for that command

As a good practice and to make sure you have the necessary dependencies, update your Linux system.
```sh
sudo apt update && sudo apt upgrade
```

Check which version of Python you have. Due to constraints by Open3D, you must have Python 3.8 (or older). 

```sh
python3 --version
```

**If you have a newer version, you'll have to change to the correct version.**

Move into the `src` folder
```sh
cd src
pip install pipenv
pipenv install
```
Add the following lines to the `~/.bashrc` file (replace `your_username` with your username)
```sh
export PATH=$PATH:/home/[your_username]/.local/bin/
```
Run this command to update the shell
```sh
. ~/.bashrc
```

## Usage
To use LiDAR Visualization Tool, simply execute the following commands in the terminal.

### Converting a nuScenes Dataset
`input_path` : a path to a nuScenes directory

`output_path` : a path to the output directory

`prediction_path (optional)` : a path to a JSON file with the predicted bounding boxes

```sh
pipenv run python3 nuscenes-ct.py -f <input_path> -o <output_path> -p <prediction_path>
```

### Converting a Waymo Dataset
`input_path` : a path to a Waymo TFRecord file

`output_path` : a path to the output directory

`r` : Specifies that the path given to input path is a directory that only contains TFRecord files to convert

```sh
pipenv run python3 waymo-ct.py -f <input_path> -o <output_path> -r
```

### Converting a Third-Party Dataset
Thus far, LiDAR Visualization Tool only supports conversion scripts for nuScenes and Waymo. However, users are welcome to create their own conversion scripts using the utility functions we've provided. You can find documentation for these in `LCT.md`.

### Visualizing Data
`input_path` : a path to a converted dataset in the our LVT format
```sh
pipenv run python3 lct.py -f <input_path>
```

### Controling the Visualization
Here is a quick guide on how to navigate the GUI.

* Switch the direction of the image by using the `Switch RGB Sensor` dropdown
* Advance the car's position using the `Switch Frame` plus and minus buttons (you can also type in the text box)
* Toggle between ground truth and predicted bounding boxes using the `Toggle Predicted or GT` dropdown
* Use the `Center Pointcloud View on Vehicle` button (the `Center` button) to snap the pointcloud view back to the car
* If you want to view specific bounding boxes, simply check the category(s) that you want to view
   * If you've selected a category from `Ground Truth Annotation Controls`, make sure the `Toggle Predicted or GT` is set to `Ground Truth`
   * If you've selected a category from `Predicted Annotation Controls`, make sure the `Toggle Predicted or GT` is set to `Predicted`
   * If you want to view ground truth and predicted boxes simultaneously, check the `Display Predicted and GT` checkbox under `Compare Predicted Data` and select the categories you want
* When viewing predicted data, you can also set the confidence threshold which means that all of the displayed boxes will have a confidence over the entered value. Do so by using the `Specify Confidence Threshold` plus and minus buttons (you can also type in the text box)
* If you want to search for a specific ground truth bounding box category in the image, you can check the category you want under `Ground Truth Annotation Controls` and then use the `Search Frames for Selected GT Boxes` buttons (the `Previous` and `Next` buttons) to search for the next occurence
