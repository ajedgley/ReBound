# LiDAR Visualization Tool

## Getting Started
### Prerequistes
LiDAR Visualization Tool runs on a Linux system, so you'll need to set up a system if you have a different operating system.
We have 2 ways for you to set up a system.
1. **Dual-boot your system.**
    
    Here is a [tutorial](https://linuxconfig.org/how-to-install-ubuntu-20-04-alongside-windows-10-dual-boot) you can follow to set this up.
2. **Set-up a GUI-based virtual machine.**

    We used [VMware](https://www.vmware.com/) to set-up our virtual machine(**Note**: [VirtualBox](https://www.virtualbox.org/) will not be able to convert Waymo datasets due to lack of support for tensorflow [Tensorflow](https://www.tensorflow.org/)). 
    Check out this [tutorial](https://unixcop.com/how-to-install-ubuntu-21-04-on-vmware-workstation-pro/) that can help with installation; however, here are some notes:
    * We used [Ubuntu 20.04](https://releases.ubuntu.com/20.04/) since it includes the correct verion of Python (Python 3.8)
    * We recommend allocating at least 30 GB of hard disk space and 4 GB of memory

### Cloning the Repository
**Note**: if any command doesn't exist, run `sudo apt-get install <command>` for that command

Execute these commands on your Linux system.
```sh
git clone https://para.cs.umd.edu/neehar/lidar.git
git checkout Development
```

### Installing Dependencies
**Note**: if any command doesn't exist, run `sudo apt-get install <command>` for that command

As a good practice and to make sure you have the necessary dependencies, update your Linux system.
```sh
sudo apt update && sudo apt upgrade
```

Check which version of Python you have. Due to constraints by Open3D, you must have Python 3.8. 

```sh
python3 --version
```

If you have a different version, you'll have to change to the correct version. We've included a tutorial for a Debian system.

---
Run these commands to get the necessary dependencies to install Python (run in your root directory).
```sh
# Install dependencies
sudo apt install build-essential zlib1g-dev libncurses5-dev libgdbm-dev libnss3-dev libssl-dev libsqlite3-dev libreadline-dev libffi-dev curl libbz2-dev
# download a Python 3.8 tarball
wget https://www.python.org/ftp/python/3.8.12/Python-3.8.12.tgz
# unzip the tarball
tar xzf Python-3.8.12.tgz
# make sure your system has the right configurations
cd Python-3.8.12
./configure --enable-optimizations
# make and install the Python binaries
make -j 2
sudo make altinstall
```
**Note**: you'll use `python3.8` wherever you see `python3` from now on

---

Move into the `src` folder
```sh
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
Thus far, LiDAR Visualization Tool only supports conversion scripts for nuScenes and Waymo. However, users are welcome to creating their own conversion scripts using the utility functions we've created. You can find documentation for these in LCT.md

### Visualizing Data
`input_path` : a path to a converted dataset in the our LVT format
```sh
pipenv run python3 lct.py -f <input_path>
```