

import os.path
import re

import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import warnings

warnings.filterwarnings("ignore")

def split_line(line):
    # Define the regular expression pattern
    pattern = r'(\w+)_\d+_\d+_(FaceTalk_\d+_\d+_\w+)\.npy'

    # Use re.match to find the groups in the line
    match = re.match(pattern, line)

    # Check if the line matches the pattern
    if match:
        # Extract the groups
        groups = match.groups()
        return list(groups)
    else:
        # Return None if the line does not match the pattern
        return None
def process_file(file_path):
    for filename in os.listdir(file_path):

        # Construct the full path to the file
        result = split_line(filename)

        # Print the result or handle it as needed
        if result is not None:
            print(result)
        else:
            print(f"Line does not match the pattern: {filename}")



# Replace 'your_file.txt' with the actual file path
file_path = r'/home/mirko/PycharmProjects/Tesi/Classification/datasetCOMA/'
process_file(file_path)
