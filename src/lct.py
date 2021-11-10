"""
lct.py

Conversion tool to bring nuscenes dataset into LVT. 
"""

import getopt
import sys
import os
import utils
import numpy as np

# Parse CLI args and validate input
def parse_options():

    input_path = ""
    
    # Read in flags passed in with command line argument
    # Make sure that options which need an argument (namely -f for input file path and -o for output file path) have them
    try:
        opts, args = getopt.getopt(sys.argv[1:], "hf:", "help")
    except getopt.GetoptError as err:
        print(err)
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print("use -f to specify directory of LVT dataset")

            sys.exit(2)
        elif opt == "-f": #and len(opts) == 2:
            input_path = arg
        else:
            # Only reach here if you were passed in a single option; consider this invalid input since we need both file paths
            print("Invalid set of arguments entered. Please refer to -h flag for more information.")
            sys.exit(2)

    return input_path  
