#Conversion Script for waymo Dataset
import sys
import getopt
import os
from utils import create_lct_directory

#Get command line options
def parse_options():
    waymo_path = ""
    folder_name = ""
    custom_path = ""
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:o:p:", "help")
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("required: -f to specify the path to the Waymo file")
            print("required: -o to specify the name of the directory where the LVT format will go. Will be a folder in the current directory")
            print("optional: -p to specify a custom path where the LVT format dataset will go")
            sys.exit(2)
        elif opt == "-f":
            waymo_path = arg
        elif opt == "-o":
            folder_name = arg
        elif opt == '-p':
            custom_path = arg
        else:
            sys.exit(2)

    return (waymo_path, folder_name, custom_path)

if __name__ == "__main__":
    (waymo_path, folder_name, custom_path) = parse_options()

    #Check if path specified is a Waymo dataset file
    if os.path.splitext(waymo_path)[1] != ".tfrecord":
        print("The file specified is not a tfrecord")
        sys.exit(2)

    if len(custom_path) != 0:
        create_lct_directory(os.getcwd().join(custom_path), folder_name)
    else:
        create_lct_directory(os.getcwd(), folder_name)