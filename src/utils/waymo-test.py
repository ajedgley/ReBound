import unittest
import os
import sys
import getopt
import tensorflow as tf

# These tests are to be run after the waymo -> LVT -> waymo pipeline, 
# without any edits, to ensure data is retained

def parse_options():
    """Read in user command line input to get directory paths which will be used for input and output.
    Args:
        None
    Returns:
        input_path: Path to waymo dataset being read into LVT
        output_path: Path where user wants LVT to generate generic data format used in program
    """
    input_path = ""
    output_path = ""
    batch_processing = False

    # User is able to specify -h, -f, -o, and -r options
    # -h brings up help menu
    # -f is used to specify the path to the Waymo file you want to read in and requires one arg. If -r is specified then this arg
    # corresponds to a directory containing all the .tfrecord files you'd like to read in
    # -o is used to specify the path to the directory where the LVT format will go. If -r is specified then this folder will contain output
    # folders for each .tfrecord file read in
    # -r is used to specify the user is trying to batch process a set of files corresponding to the directory given with the -f flag
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:o:r", "help")
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("REQUIRED: -f to specify the path to the Waymo file")
            print("REQUIRED: -o to specify the name of the directory where the LVT format will go. Will be a folder in the current directory")
            print("OPTIONAL: -r to specify that the input to -f is a directory containing only tfrecord files to be converted")
            sys.exit(2)
        elif opt == "-f":
            input_path = arg
        elif opt == "-o":
            output_path = arg
        elif opt == "-r":
            # Indicates that the user is trying to run batch processing
            batch_processing = True
        else:
            sys.exit(2)

    return (input_path, output_path, batch_processing)

def count_frames(dataset):
    """counts frames in dataset to use for progress bar
    Args:
        dataset: waymo dataset
    Returns:
        frame_count: number of frames
        """
    frame_count = 0
    # dataset is a tfrecord, so no "length" exists.
    for frame in dataset:
        frame_count += 1
    return frame_count

class TestWaymoConversion(unittest.TestCase):
    def test_tfrecord(self):
        check = True
        str = ""
        for i in range(len(in_datasets)):
            in_list = []
            out_list = []
            in_data =  in_datasets[i]
            out_data = out_datasets[i]
        
            for frame in in_data:
                in_list.append(frame)
            for frame in out_data:
                out_list.append(frame)
    
            if len(in_list) == len(out_list):
                check = check and True
                for frame in in_list:
                    if frame in out_list:
                        out_list.remove(frame)
                    else:
                        check = check and False
                        str += "file " + str(i)
            else: check = check and False
        self.assertTrue(check, "")


if __name__ == '__main__':
    #input_path = "../../../segment-10212406498497081993_5300_000_5320_000_with_camera_labels.tfrecord"
    #output_path = "../../../segment-copy.tfrecord"
    (input_path, output_path, batch_processing) = parse_options()
    
    # This list will remain empty if we're not batch processing, but if we're batch processing then it will list all the items being
    # proccessed
    batch_items = []
    # Check if path specified is a Waymo dataset file if not batch processing
    if not batch_processing and os.path.splitext(input_path)[1] != ".tfrecord":
        print("The file specified is not a tfrecord")
        sys.exit(2)
    elif batch_processing:
        # Check if all files in directory specified correspond to Waymo dataset file.
        dir_contents = os.listdir(input_path)
        for item in dir_contents:
            item_name_details = os.path.splitext(item)
            # Fail if any item is not a .tfrecord
            if item_name_details[1] != ".tfrecord":
                print("The directory specified for batch processing has a file which is not a tfrecord")
                print("The file name is", item)
                sys.exit(2)
            batch_items.append(item_name_details[0])
    
    # Extract data from TFRecord File
    in_datasets = []
    out_datasets = []
    if not batch_processing:
        in_datasets.append(tf.data.TFRecordDataset(input_path, ''))
        out_datasets.append(tf.data.TFRecordDataset(output_path, ''))
    else:
        # Convert each tfrecord file at the input directory location if we are batch processing
        dir_contents = os.listdir(input_path)
        for item in dir_contents:
            in_datasets.append(tf.data.TFRecordDataset(input_path + "/" + item, ''))
            out_datasets.append(tf.data.TFRecordDataset(output_path + "/" + item, ''))

    sys.argv = sys.argv[:1]  
    unittest.main()